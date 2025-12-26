import os
import jax
from huggingface_hub.constants import HF_HOME

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "hyperscaleescomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

import optax

import numpy as np

import hyperscalees as hs
from hyperscalees.models.llm.auto import get_model, models
from hyperscalees.models.llm.tokenizer import LegacyWorldTokenizer
from hyperscalees.models.common import simple_es_tree_key

from hyperscalees.environments.llm_bandits import all_tasks, validation_tasks

import tyro
from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path

from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.multihost_utils import process_allgather

from omegaconf import DictConfig, OmegaConf   
from hydra import initialize, compose, initialize_config_dir
from hydra.utils import instantiate


from .utils import build_generate_thread

import time

import tqdm

import operator

import wandb

@dataclass
class Args:
    seed: int = 0
    model_choice: Literal[tuple(models.keys())] =  "7g0.1B"
    wandb_directory: Optional[str] = "."

    rwkv_type: str = "AssociativeScanRWKV"
    dtype: Optional[str] = None

    parallel_generations_per_gpu: int = 128
    parallel_validations_per_gpu: int = 128
    generation_length: int = 101

    num_epochs: int = 100

    # lr_scale: float = 1.0
    lr: float = 1e-5
    sigma: float = 0.0#1e-3
    num_minibatches: int = 4
    clip_eps: float = 0.2
    train_temp: float = 1.0
    val_temperature: float = 0.0

    validate_every: int = 10
    parallel_validations: int = 128
    validation_iterations: int = 10

    task: Literal[tuple(all_tasks.keys())] = "fastzero"

    wandb_mode: Literal["online", "offline"] = "online"
    wandb_project: str = "HyperscaleExp"
    wandb_name: str = "full"
    tag: str = ""
    track: bool = False

    generations_per_prompt: int = 4

    coord_addr: Optional[str] = None
    num_procs: Optional[int] = None
    proc_id: Optional[int] = None


defaults = Args()
profile = os.getenv("PROFILE", "default")
CONFIG_DIR = (Path(__file__).resolve().parents[1] / "configs").as_posix()

if profile != "default":
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        user_cfg = compose(config_name=profile)
    user_overrides = OmegaConf.to_container(user_cfg, resolve=True)
    for k, v in user_overrides.items():
        if hasattr(defaults, k) and v is not None:
            setattr(defaults, k, v)

args = tyro.cli(Args, default=defaults)
print()
print(f"Using config: {profile}")
print()

print("starting distributed init")
if args.coord_addr is not None:
    jax.distributed.initialize(args.coord_addr, args.num_procs, args.proc_id)
else:
    print("NOT DISTRIBUTED CONTEXT")

master_key = jax.random.key(args.seed)

base_model_key = jax.random.fold_in(master_key, 0)
base_gen_key = jax.random.fold_in(master_key, 1)
base_valid_key = jax.random.fold_in(master_key, 2)

NOISER = hs.noiser.base_noiser.Noiser

total_num_devices = len(jax.devices())
print("global devices", jax.devices())
print("local devices", jax.local_devices())
print("process id", jax.process_index())
args.proc_id = jax.process_index()
args.total_parallel_generations = total_num_devices * args.parallel_generations_per_gpu
args.total_validation_generations = total_num_devices * args.parallel_validations_per_gpu

# args.lr = args.lr_scale * (args.sigma ** 2) * np.sqrt(args.total_parallel_generations)
mesh = jax.make_mesh((len(jax.devices()),), ('data',))

print()
print("per-device generations is", args.parallel_generations_per_gpu)
print("full number of generations is", args.total_parallel_generations)

RWKV, full_params, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
legacy_tokenizer = LegacyWorldTokenizer() if args.model_choice[0] == "7" else tokenizer

config, params, scan_map, es_map = full_params

args.prompts_per_epoch = args.total_parallel_generations // args.generations_per_prompt

Task = all_tasks[args.task](tokenizer, legacy_tokenizer, args.generation_length)

def replicate_matrix(x):
    return jax.make_array_from_single_device_arrays(x.shape, NamedSharding(mesh, P()), [jax.device_put(x, d) for d in jax.local_devices()])

params = jax.tree.map(replicate_matrix, params)

solver = optax.adam(args.lr)
# solver = optax.masked(solver, jax.tree.map(lambda x: x==1, es_map))
optimizer = solver.init(params)
frozen_noiser_params, noiser_params = NOISER.init_noiser(params, args.sigma, args.lr)
base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)


global_indices = replicate_matrix(np.arange(args.total_parallel_generations))
all_thread_idxes = jax.device_put(global_indices, NamedSharding(mesh, P('data')))

_generate_thread = build_generate_thread(RWKV, NOISER, frozen_noiser_params, config, base_evo_keys, base_gen_key, args.train_temp)

print("Compiling generate batch")
start_time = time.time()
generate_batch = jax.jit(shard_map(
    jax.vmap(_generate_thread, in_axes=(None, None, 0, 0, None)),
    mesh=mesh,
    in_specs=(P(), P(), P('data'), P('data'), P()),
    out_specs=P('data'),
    check_rep=False,
)).lower(noiser_params, params, jax.ShapeDtypeStruct((args.total_parallel_generations, args.generation_length), jnp.dtype('int32')), all_thread_idxes, 0).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(generate_batch.memory_analysis())


global_val_indices = replicate_matrix(np.arange(args.total_validation_generations))
all_thread_val_idxes = jax.device_put(global_val_indices, NamedSharding(mesh, P('data')))
# validate = build_validate(RWKV, config, params, base_evo_keys, base_valid_key, tokenizer, legacy_tokenizer, args, 0.0)
_generate_thread_val = build_generate_thread(
    RWKV, NOISER, frozen_noiser_params, config, base_evo_keys, base_valid_key, args.val_temperature
)
print("Compiling generate validation batch")
start_time = time.time()
generate_val_batch = jax.jit(shard_map(
    jax.vmap(_generate_thread_val, in_axes=(None, None, 0, 0, None)),
    mesh=mesh,
    in_specs=(P(), P(), P('data'), P('data'), P()),
    out_specs=P('data')
)).lower(noiser_params, params, jax.ShapeDtypeStruct((args.total_validation_generations, args.generation_length), jnp.dtype('int32')), all_thread_val_idxes, 0).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(generate_val_batch.memory_analysis())

"""
Input tokens: [1, 2, 3, 0, 0, 0, 0, 0]
Gener tokens: [1, 2, 3, 7, 7, 7, 7, 7]


"""

def single_example_loss(params, old_params, noiser_params, is_input_token, tokens, advantage):
    T = tokens.shape[0]
    token_padding = (16 - T % 16) % 16
    input_tokens = jnp.concatenate((tokens, jnp.zeros_like(tokens[:token_padding])))

    input_state = RWKV.default_state(params, config)
    
    pi, _ = RWKV.forward(NOISER, frozen_noiser_params, noiser_params, config, params, base_evo_keys, None, input_tokens, input_state)
    old_pi, _ = RWKV.forward(NOISER, frozen_noiser_params, noiser_params, config, old_params, base_evo_keys, None, input_tokens, input_state)

    pi_logprob = jax.nn.log_softmax(pi[:T-1])[jnp.arange(T-1), tokens[1:]]
    old_pi_logprob = jax.nn.log_softmax(old_pi[:T-1])[jnp.arange(T-1), tokens[1:]]
    ratio = jnp.exp(pi_logprob - old_pi_logprob)

    token_loss = -jnp.minimum(
        ratio * advantage,
        jnp.clip(ratio, 1-args.clip_eps, 1+args.clip_eps) * advantage
    )

    # pg_loss1 = -advantage * ratio
    # pg_loss2 = -advantage * jnp.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    # pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    return jnp.mean(jnp.where(is_input_token[1:], 0.0, token_loss))

def batch_loss(params, old_params, noiser_params, batch_is_input_token, batch_tokens, advantages):
    return jnp.mean(jax.vmap(single_example_loss, in_axes=(None, None, None, 0, 0, 0))(params, old_params, noiser_params, batch_is_input_token, batch_tokens, advantages))

fast_batch_grad = jax.value_and_grad(batch_loss)

def _do_update(optimizer, noiser_params, params, is_input_token, generations, raw_scores, lr):
    B_local = generations.shape[0]
    assert B_local % args.generations_per_prompt == 0
    assert B_local % args.num_minibatches == 0

    group_scores = raw_scores.reshape((-1, args.generations_per_prompt))
    true_scores = (group_scores - jnp.mean(group_scores, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)
    true_scores = true_scores.ravel()

    local_mb_size = B_local // args.num_minibatches

    def update_loop(state, x):
        new_params, optimizer = state
        is_input_batch, gen_batch, score_batch = x

        loss, grad = fast_batch_grad(
            new_params, params, noiser_params,
            is_input_batch, gen_batch, score_batch
        )

        grad = jax.tree.map(lambda g: jax.lax.pmean(g, 'data'), grad)

        updates, optimizer = solver.update(grad, optimizer, new_params)
        new_params = optax.apply_updates(new_params, updates)

        return (new_params, optimizer), loss

    (new_params, optimizer), losses = jax.lax.scan(
        update_loop,
        (params, optimizer),
        (
            is_input_token.reshape(args.num_minibatches, local_mb_size, -1),
            generations.reshape(args.num_minibatches, local_mb_size, -1),
            true_scores.reshape(args.num_minibatches, local_mb_size),
        )
    )

    param_diffs = jax.tree.map(
        lambda x, y: jnp.sqrt(jnp.mean((x - y) ** 2)),
        params, new_params
    )

    return optimizer, noiser_params, new_params, param_diffs

print()
print("Compiling do update")
start_time = time.time()
do_update = jax.jit(
    shard_map(
        _do_update,
        mesh=mesh,
        in_specs=(P(), P(), P(), P('data'), P('data'), P('data'), P()),
        out_specs=(P(), P(), P(), P()),
        check_rep=False,
    ),
    donate_argnums=(0, 1, 2),
).lower(
    optimizer,
    noiser_params,
    params,
    jnp.zeros((args.total_parallel_generations, args.generation_length), dtype=jnp.bool_),
    jnp.zeros((args.total_parallel_generations, args.generation_length), dtype=jnp.int32),
    jnp.zeros((args.total_parallel_generations,), dtype=jnp.float32),
    args.lr,
).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(do_update.memory_analysis())

true_train_fitness_sum = 0.0

FULL = 0
LORA = 1

full_name = f"{args.task}_grpo_{args.wandb_name}_lr={args.lr}_bs={args.total_parallel_generations}"
print("Run name", full_name)
if args.track and jax.process_index() == 0:
    print("Tracking run ...")
    if args.wandb_mode == "offline":
        print("Initializing wandb in offline mode")
        os.environ["WANDB_MODE"] = "offline" 
    wandb_dir = (Path(args.wandb_directory) / "wandb_runs").resolve()
    wandb_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        config=args,
        name=full_name,
        dir=str(wandb_dir),
    )

ValTask = validation_tasks[args.task](tokenizer, legacy_tokenizer, args.generation_length)

def validate(noiser_params, params, epoch):
    """Runs validation over args.validation_iterations * total_validation_generations samples."""
    total = 0.0
    count = 0

    for i in range(args.validation_iterations):
        # Per-iteration indices shaped/sharded exactly like the compile contract
        val_unique_indices = jax.device_put(
            replicate_matrix(jnp.arange(args.total_validation_generations)),
            NamedSharding(mesh, P('data'))
        ) + i * args.total_validation_generations

        # Build prompts per-shard to preserve sharding and avoid host OOM
        val_unique_prompts = jax.make_array_from_single_device_arrays(
            (args.total_validation_generations, args.generation_length),
            NamedSharding(mesh, P('data')),
            [ValTask.get_input(shard.data) for shard in val_unique_indices.addressable_shards]
        )

        # Generate on devices
        val_outputs = jax.block_until_ready(
            generate_val_batch(noiser_params, params, val_unique_prompts, val_unique_indices, epoch)
        )

        # Compute fitness per shard on host CPU, then place back onto the shard device
        _local_fitness = [
            jax.device_put(
                ValTask.get_batch_fitness(
                    jax.device_put(idx_shard.data, jax.local_devices(backend='cpu')[0]),
                    jax.device_put(out_shard.data, jax.local_devices(backend='cpu')[0])
                ),
                out_shard.device
            )
            for idx_shard, out_shard in zip(val_unique_indices.addressable_shards, val_outputs.addressable_shards)
        ]
        local_fitness = jax.make_array_from_single_device_arrays(
            (args.total_validation_generations,), NamedSharding(mesh, P('data')), _local_fitness
        )

        gathered = process_allgather(local_fitness, tiled=True)
        total += jnp.sum(gathered)
        count += gathered.size

    return (total / count).item()

def single_epoch(optimizer, noiser_params, params, true_train_fitness_sum, epoch):
    if epoch % args.validate_every == 0:
        print("VALIDATION")
        validation_score = validate(noiser_params, params, epoch)
        print("VALIDATION SCORE=", validation_score)
    else:
        validation_score = None
    # print("CURRENT MEMORY start of epoch", jax.local_devices()[0].memory_stats())
    start_time = time.time()
    unique_indices = jax.device_put(replicate_matrix(jnp.arange(args.prompts_per_epoch)), NamedSharding(mesh, P('data'))) + epoch * args.prompts_per_epoch
    indices = jnp.repeat(unique_indices, args.generations_per_prompt, axis=0)
    unique_prompts = jax.make_array_from_single_device_arrays((args.prompts_per_epoch, args.generation_length), NamedSharding(mesh, P('data')), [Task.get_input(shard.data) for shard in unique_indices.addressable_shards])
    # Task.get_input(unique_indices)
    batch_prompts = jnp.repeat(unique_prompts, args.generations_per_prompt, axis=0)
    ones = jnp.ones((args.total_parallel_generations, 1), dtype=jnp.bool_)
    ones = jax.device_put(ones, NamedSharding(mesh, P('data')))
    all_is_input_token = jnp.concatenate((ones, batch_prompts[:, 1:] != 0), axis=1)
    prompt_processing_time = time.time() - start_time

    # print("CURRENT MEMORY start of batch", jax.local_devices()[0].memory_stats())
    start_time = time.time()
    if epoch == 0:
        print("generating batch")
    output_batch = jax.block_until_ready(generate_batch(noiser_params, params, batch_prompts, all_thread_idxes, epoch))
    # all_generations = jax.make_array_from_single_device_arrays((args.total_parallel_generations, args.generation_length), NamedSharding(mesh, P('data')), output_batch)
    all_generations = output_batch
    token_generation_time = time.time() - start_time

    
    start_time = time.time()
    if epoch == 0:
        print("calculating fitness")
    # local_output_scores = jax.block_until_ready(Task.get_batch_fitness(indices, output_batch))
    _local_fitness = [jax.device_put(Task.get_batch_fitness(jax.device_put(shard1.data, jax.local_devices(backend='cpu')[0]), jax.device_put(shard2.data, jax.local_devices(backend='cpu')[0])), shard1.device) for shard1, shard2 in zip(indices.addressable_shards, output_batch.addressable_shards)]
    # for x in _local_fitness:
        # print(x.shape, x.device)
    local_fitness = jax.make_array_from_single_device_arrays((args.total_parallel_generations,), NamedSharding(mesh, P('data')), _local_fitness)

    fitness_time = time.time() - start_time

    # print("CURRENT MEMORY start of update", jax.local_devices()[0].memory_stats())
    start_time = time.time()
    if epoch == 0:
        print("gathering")
    output_scores = process_allgather(local_fitness, True)
    gather_time = time.time() - start_time

    start_time = time.time()
    if epoch == 0:
        print("updating params")
    optimizer, noiser_params, params, parameter_differences = jax.block_until_ready(do_update(optimizer, noiser_params, params,
                                                                                   all_is_input_token, all_generations,
                                                                                   local_fitness, args.lr))
    parameter_update_time = time.time() - start_time

    # print("CURRENT MEMORY start of stats", jax.local_devices()[0].memory_stats())
    # parameter_differences = jax.tree.map(lambda x, y:jnp.mean(jnp.abs(x-y)), params, updated_params)
    lora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == LORA else 0.0, parameter_differences, es_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == LORA else 0.0, es_map))
    nonlora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == FULL else 0.0, parameter_differences, es_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == FULL else 0.0, es_map))

    # params = updated_params

    true_train_fitness_sum += jnp.sum(output_scores).item()

    stats = {
        "avg_fitness": jnp.mean(output_scores),
        "std_fitness": jnp.std(output_scores),
        "max_fitness": jnp.max(output_scores),
        "min_fitness": jnp.min(output_scores),
        "median_fitness": jnp.median(output_scores),
        "lora_updates": lora_updates,
        "nonlora_updates": nonlora_updates,
        # "total_lora_updates": total_lora_updates,
        # "total_nonlora_updates": total_nonlora_updates,
        "prompt_preproc_time": prompt_processing_time,
        "token_gen_time": token_generation_time,
        "fitness_time": fitness_time,
        "gather_time": gather_time,
        "update_time": parameter_update_time,
        "true_train_avg_fitness": true_train_fitness_sum / ((epoch + 1) * args.total_parallel_generations)
    }

    if validation_score is not None:
        stats["validation_score"] = validation_score
    
    if args.track and jax.process_index() == 0:
        run.log(stats)
    else:
        print(f"Mean fitness: {jnp.mean(output_scores)}; std fitness: {jnp.std(output_scores)}; max fitness: {jnp.max(output_scores)}; min fitness: {jnp.min(output_scores)}; median fitness: {jnp.median(output_scores)}")
        print("mean parameter diffs")
        print("Lora modules:", lora_updates)
        print("Full modules:", nonlora_updates)
        print("Stats:")
        for k in stats:
            print(f"\t{k}: {stats[k]}")

    return optimizer, noiser_params, params, true_train_fitness_sum

for epoch in tqdm.trange(args.num_epochs):
    optimizer, noiser_params, params, true_train_fitness_sum = single_epoch(optimizer, noiser_params, params, true_train_fitness_sum, epoch)

if args.track:
    run.finish()
