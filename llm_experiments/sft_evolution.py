import os
import sys
import csv
import jax
from huggingface_hub.constants import HF_HOME

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "hyperscaleescomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

import numpy as np

import hyperscalees as hs
from hyperscalees.models.base_model import CommonInit
from hyperscalees.models.llm.auto import get_model, models, save, load
from hyperscalees.models.llm.tokenizer import LegacyWorldTokenizer
from hyperscalees.models.common import simple_es_tree_key

from hyperscalees.noiser import all_noisers
from hyperscalees.environments.llm_bandits import all_tasks, validation_tasks

import tyro
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal
from pathlib import Path

from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils as mu
from jax.experimental.multihost_utils import process_allgather

from omegaconf import DictConfig, OmegaConf   
from hydra import initialize, compose, initialize_config_dir
from hydra.utils import instantiate

from .utils import (
    build_generate_thread,
    build_generate_sft_thread,
    safe_decode,
    as_named
)

import time

import tqdm

import operator

@dataclass
class Args:
    seed: int = 0
    model_choice: str =  "7g0.1B"
    output_directory: Optional[str] = "."
    wandb_directory: Optional[str] = "."

    save_model: bool = True
    load_model: bool = False
    save_path: Optional[str] = "."
    load_path: Optional[str] = None

    rwkv_type: str = "BaseRWKV"
    dtype: Optional[str] = None

    parallel_generations_per_gpu: int = 1024
    parallel_validations_per_gpu: int = 128

    generation_length: int = 1024
    sub_sequence_length: int = 16

    num_epochs: int = 100
    log_output_every: int = 10

    lr_scale: float = 1.0
    sigma: float = 1e-3
    noise_reuse: int = 1
    freeze_nonlora: bool = True
    temperature: float = 0.0
    val_temperature: float = 0.0
    rank: int = 1

    validate_every: int = 10
    save_every: int = 100
    parallel_validations: int = 128
    validation_iterations: int = 10

    task: str = "fastzero"
    noiser: str = "eggroll"

    wandb_mode: Literal["online", "offline"] = "online" 
    wandb_project: str = "HyperscaleExp"
    wandb_name: str = "full"
    track: bool = False

    generations_per_prompt: int = 8

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

def _resolve_ckpt_path(load_path: str | Path) -> Path:
    p = Path(load_path)
    if p.is_dir():
        cand = p / "latest.model"
        if not cand.exists():
            raise FileNotFoundError(f"No latest.model in directory: {p}")
        return cand
    if p.suffix != ".model":
        raise ValueError(f"Expected a .model file or directory, got: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")
    return p

def _to_jnp_tree(x, dtype_str: Optional[str]):
    if dtype_str is None:
        return jax.tree.map(lambda y: jnp.asarray(y) if isinstance(y, np.ndarray) else y, x)
    target_dtype = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32
    return jax.tree.map(lambda y: jnp.asarray(y, dtype=target_dtype) if isinstance(y, np.ndarray) else y, x)

master_key = jax.random.key(args.seed)

base_model_key = jax.random.fold_in(master_key, 0)
base_gen_key = jax.random.fold_in(master_key, 1)
base_valid_key = jax.random.fold_in(master_key, 2)

NOISER = all_noisers[args.noiser]

total_num_devices = len(jax.devices())
print("global devices", jax.devices())
print("local devices", jax.local_devices())
print("process id", jax.process_index())
args.proc_id = jax.process_index()
args.total_parallel_generations = total_num_devices * args.parallel_generations_per_gpu
args.total_validation_generations = total_num_devices * args.parallel_validations_per_gpu

mesh = jax.make_mesh((len(jax.devices()),), ('data',))

print()
print("per-device generations is", args.parallel_generations_per_gpu)
print("full number of generations is", args.total_parallel_generations)

RWKV, full_params, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
legacy_tokenizer = LegacyWorldTokenizer() if args.model_choice[0] == "7" else tokenizer

config, params, scan_map, es_map = full_params

args.prompts_per_epoch = args.total_parallel_generations // args.generations_per_prompt
args.iterations_per_prompt = args.generation_length // args.sub_sequence_length

Task = all_tasks[args.task](tokenizer, legacy_tokenizer, args.generation_length)

def replicate_matrix(x):
    return jax.make_array_from_single_device_arrays(x.shape, NamedSharding(mesh, P()), [jax.device_put(x, d) for d in jax.local_devices()])

params = jax.tree.map(replicate_matrix, params)
frozen_noiser_params, noiser_params = NOISER.init_noiser(params, args.sigma, args.lr_scale, group_size=args.generations_per_prompt, freeze_nonlora=args.freeze_nonlora, noise_reuse=args.sub_sequence_length, rank=args.rank)
base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)

global_indices = replicate_matrix(np.arange(args.total_parallel_generations)) % args.generations_per_prompt
global_val_indices = replicate_matrix(np.arange(args.total_validation_generations))
all_thread_idxes = jax.device_put(global_indices, NamedSharding(mesh, P('data')))
all_thread_val_idxes = jax.device_put(global_val_indices, NamedSharding(mesh, P('data')))

_generate_thread = build_generate_sft_thread(RWKV, NOISER, frozen_noiser_params, config, base_evo_keys, base_gen_key, args.temperature)

state_ex = RWKV.default_state(params, config)
N = args.total_parallel_generations
T = args.sub_sequence_length
print("Compiling generate batch")
start_time = time.time()
generate_sft_batch = jax.jit(shard_map(
    jax.vmap(
        _generate_thread,
        # (noiser_params, params, inputs[T], targets[T], thread_idx, epoch, init_token, init_state)
        in_axes=(None, None, 0, 0, 0, None, 0, 0),
    ),
    mesh=mesh,
    in_specs=(P(), P(), P('data'), P('data'), P('data'), P(), P('data'), P('data')),
    out_specs=(P('data'), P('data'), P('data'), P('data'))  # (true_inputs, log_probs, last_state, last_tok)
)).lower(
    noiser_params,
    params,
    jax.ShapeDtypeStruct((N, T), jnp.int32),   # inputs
    jax.ShapeDtypeStruct((N, T), jnp.int32),   # targets
    all_thread_idxes,                          # [N]
    0,                                         # epoch (scalar)
    jax.ShapeDtypeStruct((N,), jnp.int32),     # init_token
    jax.tree.map(lambda x: jax.ShapeDtypeStruct((N,)+x.shape, x.dtype), state_ex)  # init_state batched
).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(generate_sft_batch.memory_analysis())

# validate = build_validate(RWKV, config, params, base_evo_keys, base_valid_key, tokenizer, legacy_tokenizer, args, args.temperature)
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


def _do_update(noiser_params, params, raw_scores, epoch_num):
    iterinfos = (jnp.full(raw_scores.size, epoch_num, dtype=jnp.int32), global_indices)

    fitnesses = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
    noiser_params, new_params = NOISER.do_updates(frozen_noiser_params, noiser_params, params, base_evo_keys, fitnesses, iterinfos, es_map)

    return noiser_params, new_params, jax.tree.map(lambda x, y: jnp.sqrt(jnp.mean((x - y) ** 2)), params, new_params)


print()
print("Compiling do update")
start_time = time.time()
do_update = jax.jit(shard_map(
    _do_update,
    mesh=mesh,
    in_specs=(P(), P(), P(), P()),
    out_specs=(P(), P(), P())
), donate_argnums=(0, 1)).lower(noiser_params, params, jnp.zeros(args.total_parallel_generations), 0).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(do_update.memory_analysis())

true_train_fitness_sum = 0.0

FULL = 0
LORA = 1

full_name = f"{args.task}_{args.noiser}_{args.wandb_name}_lr={args.lr_scale}_sigma={args.sigma:.2e}_bs={args.total_parallel_generations}"
experiment_id = f"{full_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

run_id = None

mu.sync_global_devices("pre-wandb-init")
print("Run name", full_name)
if jax.process_index() != 0:
    os.environ["WANDB_DISABLED"] = "true"
elif args.track and jax.process_index() == 0:
    import wandb 
    print("Initializing wandb ...")
    if args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline" 
    
    wandb_dir = (Path(args.wandb_directory) / "wandb_runs").resolve()
    wandb_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        config=args,
        name=full_name,
        dir=str(wandb_dir),
    )
    run_id = run.id
mu.sync_global_devices("post-wandb-init")

if jax.process_index() == 0:
    safe_run_id = run_id or os.environ.get("WANDB_RUN_ID") or experiment_id
    base_out_dir = Path(args.output_directory) if args.output_directory else (Path.cwd() / "outputs")
    run_out_dir = base_out_dir / f"{safe_run_id}"
    run_out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_model:
        ckpt_dir = (Path(args.save_path) / safe_run_id) if args.save_path else (run_out_dir / "checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "latest.model"

def _to_cpu_tree(x):
    """Bring a (possibly sharded/replicated) pytree to host numpy on proc 0."""
    x_host = jax.device_get(x)
    def _coerce(y):
        if isinstance(y, (list, tuple)) and len(y) > 0:
            y = y[0]
        if isinstance(y, jnp.ndarray):
            return np.asarray(y)
        return y
    return jax.tree.map(_coerce, x_host)

ValTask = validation_tasks[args.task](tokenizer, legacy_tokenizer, args.generation_length)

def validate(noiser_params, params, epoch):
    """Runs validation over args.validation_iterations * total_validation_generations samples."""
    total = 0.0
    count = 0

    for i in range(args.validation_iterations):

        val_unique_indices = jax.device_put(
            replicate_matrix(jnp.arange(args.total_validation_generations)),
            NamedSharding(mesh, P('data'))
        ) + i * args.total_validation_generations

        val_unique_prompts = jax.make_array_from_single_device_arrays(
            (args.total_validation_generations, args.generation_length),
            NamedSharding(mesh, P('data')),
            [ValTask.get_input(shard.data) for shard in val_unique_indices.addressable_shards]
        )

        val_outputs = jax.block_until_ready(
            generate_val_batch(noiser_params, params, val_unique_prompts, val_unique_indices, epoch)
        )

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

    if (
        args.track
        and args.log_output_every > 0
        and (epoch % args.log_output_every == 0)
        and jax.process_index() == 0
    ):
        # Take a small sample from the first local shard to minimize overhead
        K = min(8, args.total_parallel_generations)

        local_gen = np.array(val_outputs.addressable_shards[0].data)[:K]
        local_prompts = np.array(val_unique_prompts.addressable_shards[0].data)[:K]

        rows = []
        for i in range(local_gen.shape[0]):
            prompt_txt = safe_decode(local_prompts[i], tokenizer)
            gen_txt = safe_decode(local_gen[i], tokenizer)
            rows.append([epoch, i, prompt_txt, gen_txt])

        table = wandb.Table(columns=["epoch", "sample_id", "prompt", "generation"], rows=rows)
        wandb.log({"text_samples": table}, step=epoch)

    return (total / count).item()

def tile_state_for_batch(state_single, batch):
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape), state_single)

def single_epoch(noiser_params, params, true_train_fitness_sum, epoch):
    validation_processing_time = 0.0
    if epoch % args.validate_every == 0:
        print("VALIDATION")
        start_time = time.time()
        validation_score = validate(noiser_params, params, epoch)
        print("VALIDATION SCORE=", validation_score)
        validation_processing_time = time.time() - start_time
    else:
        validation_score = None

    start_time = time.time()
    unique_indices = jax.device_put(replicate_matrix(jnp.arange(args.prompts_per_epoch)), NamedSharding(mesh, P('data'))) + epoch * args.prompts_per_epoch
    indices = jnp.repeat(unique_indices, args.generations_per_prompt, axis=0)
    unique_prompts = jax.make_array_from_single_device_arrays((args.prompts_per_epoch, args.generation_length), NamedSharding(mesh, P('data')), [Task.get_input(shard.data) for shard in unique_indices.addressable_shards])
    batch_prompts = jnp.repeat(unique_prompts, args.generations_per_prompt, axis=0)

    # Mask all tokens except leading one at each subsequence start
    B, T = batch_prompts.shape
    L = args.sub_sequence_length
    shifted_inputs = jnp.concatenate(
        [jnp.zeros((B, 1), dtype=jnp.int32), batch_prompts[:, :-1]],
        axis=1
    )
    batch_inputs = shifted_inputs

    batch_prompts = as_named(batch_prompts, mesh, P('data'))
    batch_inputs = as_named(batch_inputs, mesh, P('data'))

    start_time = time.time()
    if epoch == 0:
        print("generating batch")
    
    init_state_batched = tile_state_for_batch(RWKV.default_state(params, config), args.total_parallel_generations)
    curr_state = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P('data'))), init_state_batched)
    curr_token = as_named(jnp.zeros((args.total_parallel_generations,), dtype=jnp.int32), mesh, P('data'))

    accum_local_scores = as_named(jnp.zeros((args.total_parallel_generations,), dtype=jnp.float32), mesh, P('data'))

    for i in range(args.iterations_per_prompt):
        start = i * L
        end = start + L
        sub_input_batch  = batch_inputs[:, start:end]
        sub_target_batch = batch_prompts[:, start:end]

        true_inputs, log_probs, curr_state, curr_token = jax.block_until_ready(
            generate_sft_batch(
                noiser_params, params,
                sub_input_batch, sub_target_batch,
                all_thread_idxes, epoch,
                curr_token, curr_state,
            )
        )
        if i ==0:
            token_generation_time = time.time() - start_time

        start_fitness_time = time.time()
        local_fitness = jnp.sum(log_probs.astype(jnp.float32), axis=1)
        fitness_time = time.time() - start_fitness_time

        accum_local_scores = accum_local_scores + local_fitness

        start_gather_time = time.time()
        if epoch == 0:
            print("gathering")
        output_scores = process_allgather(local_fitness, True)
        gather_time = time.time() - start_gather_time

        start_param_update_time = time.time()
        if epoch == 0:
            print("updating params")
        noiser_params, params, parameter_differences = jax.block_until_ready(do_update(noiser_params, params, output_scores, epoch))
        parameter_update_time = time.time() - start_param_update_time

        nonlora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == FULL else 0.0, parameter_differences, es_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == FULL else 0.0, es_map))

    full_prompt_sft_time = time.time() - start_time

    saving_time = 0.0
    if (
        args.save_model
        and (epoch % args.save_every == 0)
        and jax.process_index() == 0
    ):
        start_time = time.time()
        ckpt_path = ckpt_dir / f"epoch_{epoch:06d}.model"
        ckpt = CommonInit(
            frozen_params=_to_cpu_tree(config),
            params=_to_cpu_tree(params),
            scan_map=_to_cpu_tree(scan_map),
            es_map=_to_cpu_tree(es_map),
        )
        save(ckpt, ckpt_path, overwrite=True)
        saving_time = time.time() - start_time
    
    total_scores = process_allgather(accum_local_scores, tiled=True)
    true_train_fitness_sum += jnp.sum(total_scores).item()

    stats = {
        "avg_fitness": jnp.mean(total_scores),
        "std_fitness": jnp.std(total_scores),
        "max_fitness": jnp.max(total_scores),
        "min_fitness": jnp.min(total_scores),
        "median_fitness": jnp.median(total_scores),
        "nonlora_updates": nonlora_updates,
        "token_gen_time": token_generation_time,
        "prompt_processing_time": full_prompt_sft_time,
        "fitness_time": fitness_time,
        "gather_time": gather_time,
        "update_time": parameter_update_time,
        "validation_time": validation_processing_time,
        "true_train_avg_fitness": true_train_fitness_sum / ((epoch + 1) * args.total_parallel_generations)
    }

    if validation_score is not None:
        stats["validation_score"] = validation_score
    
    if args.track and jax.process_index() == 0:
        run.log(stats)
    else:
        print(f"Mean fitness: {jnp.mean(total_scores)}; std fitness: {jnp.std(total_scores)}; max fitness: {jnp.max(total_scores)}; min fitness: {jnp.min(total_scores)}; median fitness: {jnp.median(total_scores)}")
        print("mean parameter diffs")
        print("Full modules:", nonlora_updates)
        print("Stats:")
        for k in stats:
            print(f"\t{k}: {stats[k]}")

    return noiser_params, params, true_train_fitness_sum

for epoch in tqdm.trange(args.num_epochs):
    noiser_params, params, true_train_fitness_sum = single_epoch(noiser_params, params, true_train_fitness_sum, epoch)

mu.sync_global_devices("before-wandb-finish")
if args.track and jax.process_index() == 0:
    run.finish()
mu.sync_global_devices("after-wandb-finish")