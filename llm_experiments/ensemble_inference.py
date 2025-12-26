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
from hyperscalees.models.llm.auto import get_model, models
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
from jax.experimental.multihost_utils import process_allgather

from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose, initialize_config_dir
from hydra.utils import instantiate

from .utils import safe_decode

import time
import tqdm
import operator


@dataclass
class Args:
    seed: int = 0
    model_choice: Literal[tuple(models.keys())] = "7g0.1B"
    output_directory: Optional[str] = "."

    rwkv_type: str = "BaseRWKV"
    dtype: Optional[str] = None

    # Number of noisy models to sample
    num_noises: int = 64

    # Top-k models to select for ensemble
    topk: int = 8

    generation_length: int = 100
    thinking_length: int = 100
    answer_length: int = 100

    sigma: float = 1e-3
    noise_reuse: int = 1
    freeze_nonlora: bool = True
    temperature: float = 1.0

    # Ensemble temperature for final sampling
    ensemble_temperature: float = 1.0

    task: Literal[tuple(all_tasks.keys())] = "fastzero"
    noiser: Literal[tuple(all_noisers.keys())] = "eggroll"

    num_prompts: int = 1

    coord_addr: Optional[str] = None
    num_procs: Optional[int] = None
    proc_id: Optional[int] = None


args = tyro.cli(Args)
profile = os.getenv("PROFILE", "default")
CONFIG_DIR = (Path(__file__).resolve().parents[1] / "configs").as_posix()

if profile != "default":
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        user_cfg = compose(config_name=profile)

    user_overrides = OmegaConf.to_container(user_cfg, resolve=True)
    for k, v in user_overrides.items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)

print()
print(f"Using config: {profile}")
print()
args.generation_length = args.thinking_length + args.answer_length

master_key = jax.random.key(args.seed)

base_model_key = jax.random.fold_in(master_key, 0)
base_gen_key = jax.random.fold_in(master_key, 1)

NOISER = all_noisers[args.noiser]

print("starting distributed init")
if args.coord_addr is not None:
    jax.distributed.initialize(args.coord_addr, args.num_procs, args.proc_id)
else:
    print("NOT DISTRIBUTED CONTEXT")

total_num_devices = len(jax.devices())
print("global devices", jax.devices())
print("local devices", jax.local_devices())
print("process id", jax.process_index())
args.proc_id = jax.process_index()

mesh = jax.make_mesh((len(jax.devices()),), ('data',))

print()
print(f"Number of noises: {args.num_noises}")
print(f"Top-k for ensemble: {args.topk}")

RWKV, full_params, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
legacy_tokenizer = LegacyWorldTokenizer() if args.model_choice[0] == "7" else tokenizer

config, params, scan_map, es_map = full_params

Task = all_tasks[args.task](tokenizer, legacy_tokenizer, args.generation_length)

def replicate_matrix(x):
    return jax.make_array_from_single_device_arrays(x.shape, NamedSharding(mesh, P()), [jax.device_put(x, d) for d in jax.local_devices()])

params = jax.tree.map(replicate_matrix, params)
frozen_noiser_params, noiser_params = NOISER.init_noiser(params, args.sigma, 0.0, group_size=0, freeze_nonlora=args.freeze_nonlora, noise_reuse=args.noise_reuse)
base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)


def fold_in_helper(key, epoch, true_thread_idx):
    return jax.random.fold_in(jax.random.fold_in(key, epoch), true_thread_idx)


def build_generate_with_logits(MODEL, NOISER, frozen_noiser_params, config, base_evo_keys, master_gen_key, temperature=1.0):
    """
    Build a generate function that returns both tokens and logits for ensemble.
    """

    def forward_and_sample(noiser_params, params, input_token, input_state, generation_key, iterinfo):
        gen_key, _gen_key = jax.random.split(generation_key)
        generated_outs, generated_state = MODEL.forward(NOISER, frozen_noiser_params, noiser_params, config, params, base_evo_keys, iterinfo, input_token, input_state)
        logits = generated_outs[-1]
        if temperature != 0.0:
            sampled_tok = jax.random.categorical(_gen_key, logits / temperature)
        else:
            sampled_tok = jnp.argmax(logits)
        return sampled_tok, generated_state, gen_key, logits

    def generate_thread(noiser_params, params, prompt, thread_idx, epoch_num):
        """
        Generate tokens and collect logits at each position.
        Returns: (tokens, logits) where logits has shape (seq_len, vocab_size)
        """
        start_gen_key = fold_in_helper(master_gen_key, epoch_num, thread_idx)

        iterinfo = (epoch_num, thread_idx)

        def inner_scan(carry, input_token):
            tok, state, gen_key = carry
            true_input = jnp.where(input_token == 0, tok, input_token)
            tok, state, gen_key, logits = forward_and_sample(noiser_params, params, true_input, state, gen_key, iterinfo)
            return (tok, state, gen_key), (true_input, logits)

        init_token = 0
        init_state = MODEL.default_state(params, config)

        _, (out_tokens, out_logits) = jax.lax.scan(inner_scan, (init_token, init_state, start_gen_key), prompt)
        return out_tokens, out_logits

    return generate_thread


def build_ensemble_generate_with_logits(MODEL, NOISER, frozen_noiser_params, config, base_evo_keys, master_gen_key, temperature=1.0):
    """
    Build a generate function that runs N noisy models and returns both tokens and logits.
    This is more efficient as we collect logits during the first pass.
    """

    def forward_step(noiser_params, params, input_token, input_state, iterinfo):
        """Single forward step, returns logits and new state."""
        generated_outs, generated_state = MODEL.forward(NOISER, frozen_noiser_params, noiser_params, config, params, base_evo_keys, iterinfo, input_token, input_state)
        logits = generated_outs[-1]
        return logits, generated_state

    # Vectorize over noise samples (thread_idx)
    batched_forward_step = jax.vmap(
        forward_step,
        in_axes=(None, None, 0, 0, (None, 0))  # batch over input_token, state, and thread_idx in iterinfo
    )

    def ensemble_generate_with_logits(noiser_params, params, prompt, thread_idxes, epoch_num):
        """
        Generate with ensemble of noisy models, returning both tokens and logits.

        Args:
            noiser_params: Noiser parameters
            params: Model parameters
            prompt: Input prompt tokens (seq_len,)
            thread_idxes: Array of thread indices for different noise samples (N,)
            epoch_num: Epoch number for noise generation

        Returns:
            all_tokens: All generated tokens from all noisy models (N, seq_len)
            all_logits: All logits from all noisy models (N, seq_len, vocab_size)
        """
        N = thread_idxes.shape[0]
        seq_len = prompt.shape[0]

        # Initialize states for all N noisy models
        init_state = MODEL.default_state(params, config)
        init_states = jnp.broadcast_to(init_state[None, ...], (N,) + init_state.shape)

        # Initialize generation keys for each model
        gen_keys = jax.vmap(lambda idx: fold_in_helper(master_gen_key, epoch_num, idx))(thread_idxes)

        # Current tokens for each model
        current_tokens = jnp.zeros(N, dtype=jnp.int32)

        def scan_step(carry, prompt_token):
            current_tokens, states, gen_keys = carry

            # Use prompt token if available, else use generated token
            input_tokens = jnp.where(prompt_token == 0, current_tokens, prompt_token)

            # Get logits from all N models
            iterinfos = (epoch_num, thread_idxes)
            logits, new_states = batched_forward_step(noiser_params, params, input_tokens, states, iterinfos)
            # logits: (N, vocab_size), new_states: (N, ...)

            # Sample next token independently for each model
            gen_keys, sample_keys = jax.vmap(jax.random.split)(gen_keys)
            sample_keys = sample_keys[:, 0]  # shape (N,)

            if temperature != 0.0:
                next_tokens = jax.vmap(lambda key, logit: jax.random.categorical(key, logit / temperature))(sample_keys, logits)
            else:
                next_tokens = jnp.argmax(logits, axis=-1)

            return (next_tokens, new_states, gen_keys), (input_tokens, logits)

        # Run generation for all N models
        init_carry = (current_tokens, init_states, gen_keys)
        _, (all_generated_tokens, all_logits) = jax.lax.scan(
            scan_step, init_carry, prompt
        )
        # all_generated_tokens: (seq_len, N) -> transpose to (N, seq_len)
        # all_logits: (seq_len, N, vocab_size) -> transpose to (N, seq_len, vocab_size)
        all_generated_tokens = all_generated_tokens.T
        all_logits = jnp.transpose(all_logits, (1, 0, 2))

        return all_generated_tokens, all_logits

    return ensemble_generate_with_logits


# Build the ensemble generate function that returns both tokens and logits
_ensemble_generate_with_logits = build_ensemble_generate_with_logits(
    RWKV, NOISER, frozen_noiser_params, config, base_evo_keys, base_gen_key, args.temperature
)

# Build single-model generate for reference
_generate_with_logits = build_generate_with_logits(
    RWKV, NOISER, frozen_noiser_params, config, base_evo_keys, base_gen_key, args.temperature
)


def run_topk_ensemble(noiser_params, params, prompt, num_noises, topk, epoch=0):
    """
    Run the full top-k ensemble pipeline:
    1. Sample N noises and generate with each (collecting logits in one pass)
    2. Calculate rewards for each
    3. Select top-k models
    4. Do probability-level ensemble for final answer using pre-collected logits
    """
    thread_idxes = jnp.arange(num_noises)

    # Step 1: Generate with all N noisy models AND collect logits
    all_generated_tokens, all_logits = _ensemble_generate_with_logits(
        noiser_params, params, prompt, thread_idxes, epoch
    )
    # all_generated_tokens: (N, seq_len)
    # all_logits: (N, seq_len, vocab_size)

    # Step 2: Calculate rewards for each model
    prompt_indices = jnp.zeros(num_noises, dtype=jnp.int32)  # All use same prompt
    rewards = Task.get_batch_fitness(prompt_indices, all_generated_tokens)
    rewards = jnp.array(rewards)

    # Step 3: Select top-k models based on reward
    topk_indices = jnp.argsort(rewards)[-topk:]  # Get indices of top-k rewards
    topk_rewards = rewards[topk_indices]

    # Step 4: Probability-level ensemble using top-k models
    # Use pre-collected logits - no need to regenerate!
    topk_logits = all_logits[topk_indices]  # (topk, seq_len, vocab_size)

    # Average probabilities (not logits) for ensemble
    topk_probs = jax.nn.softmax(topk_logits / args.ensemble_temperature, axis=-1)
    avg_probs = jnp.mean(topk_probs, axis=0)  # (seq_len, vocab_size)

    # Alternative: Weighted average by reward (normalized)
    # weights = jax.nn.softmax(topk_rewards * 10)  # Scale factor for sharper weighting
    # avg_probs = jnp.einsum('k,ksv->sv', weights, topk_probs)

    # Sample final tokens from averaged probabilities
    final_gen_key = jax.random.fold_in(base_gen_key, 999999)
    final_keys = jax.random.split(final_gen_key, prompt.shape[0])

    if args.ensemble_temperature != 0.0:
        # Sample from the averaged distribution
        final_tokens = jax.vmap(
            lambda key, probs: jax.random.choice(key, jnp.arange(probs.shape[0]), p=probs)
        )(final_keys, avg_probs)
    else:
        # Greedy: take argmax
        final_tokens = jnp.argmax(avg_probs, axis=-1)

    # For positions where prompt is non-zero, use prompt token
    final_tokens = jnp.where(prompt != 0, prompt, final_tokens)

    return {
        'final_tokens': final_tokens,
        'all_generated_tokens': all_generated_tokens,
        'all_logits': all_logits,
        'rewards': rewards,
        'topk_indices': topk_indices,
        'topk_rewards': topk_rewards,
        'avg_probs': avg_probs,
    }


print("\n" + "="*60)
print("Top-K Ensemble Inference Configuration")
print("="*60)
print(f"  Number of noisy models (N): {args.num_noises}")
print(f"  Top-k for ensemble: {args.topk}")
print(f"  Noise sigma: {args.sigma}")
print(f"  Temperature: {args.temperature}")
print(f"  Ensemble temperature: {args.ensemble_temperature}")
print(f"  Generation length: {args.generation_length}")
print("="*60)

print("\nCompiling ensemble functions...")
start_time = time.time()

# Create a sample prompt for compilation
sample_indices = jnp.arange(1)
sample_prompt = Task.get_input(sample_indices)[0]  # Get single prompt

# JIT compile the ensemble function
run_topk_ensemble_jit = jax.jit(
    run_topk_ensemble,
    static_argnums=(3, 4, 5)  # num_noises, topk, epoch are static
)

# Warm-up compilation
print("Compiling (this may take a while)...")
_ = run_topk_ensemble_jit(noiser_params, params, sample_prompt, args.num_noises, args.topk, 0)
print(f"Compile time: {time.time() - start_time:.2f}s")

# Run inference
print("\n" + "="*60)
print("Running Top-K Ensemble Inference")
print("="*60)

for i in range(args.num_prompts):
    print(f"\n--- Prompt {i+1}/{args.num_prompts} ---")

    indices = jnp.array([i])
    prompt = Task.get_input(indices)[0]

    start_time = time.time()
    results = run_topk_ensemble_jit(noiser_params, params, prompt, args.num_noises, args.topk, 0)
    inference_time = time.time() - start_time

    rewards = results['rewards']
    print(f"\nReward Statistics (across {args.num_noises} noisy models):")
    print(f"  Mean: {float(jnp.mean(rewards)):.4f}")
    print(f"  Std:  {float(jnp.std(rewards)):.4f}")
    print(f"  Max:  {float(jnp.max(rewards)):.4f}")
    print(f"  Min:  {float(jnp.min(rewards)):.4f}")
    print(f"  Top-{args.topk} rewards: {[float(r) for r in results['topk_rewards']]}")

    print(f"\nInference time: {inference_time:.2f}s")

    # Decode and display results
    prompt_text = safe_decode(np.array(prompt), tokenizer)
    final_text = safe_decode(np.array(results['final_tokens']), tokenizer)

    print(f"\nPrompt: {prompt_text[:200]}...")
    print(f"\nEnsemble Output: {final_text[:500]}...")

    # Show some individual model outputs for comparison
    print(f"\nSample outputs from top-{args.topk} models:")
    for j, idx in enumerate(results['topk_indices'][:3]):  # Show first 3
        model_text = safe_decode(np.array(results['all_generated_tokens'][idx]), tokenizer)
        print(f"  Model {int(idx)} (reward={float(results['rewards'][idx]):.4f}): {model_text[:200]}...")

    # Check ensemble reward
    final_reward = Task.get_batch_fitness(jnp.array([0]), results['final_tokens'][None, :])[0]
    print(f"\n--- Results Summary ---")
    print(f"  Ensemble output reward: {float(final_reward):.4f}")
    print(f"  Best individual reward: {float(jnp.max(results['rewards'])):.4f}")
    print(f"  Mean reward: {float(jnp.mean(results['rewards'])):.4f}")

print("\n" + "="*60)
print("Done!")
print("="*60)
