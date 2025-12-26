import os
import jax
from huggingface_hub.constants import HF_HOME

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "hyperscaleescomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import optax
import jax.numpy as jnp

import hyperscalees as hs
from functools import partial

import operator

NOISER = hs.noiser.eggroll.EggRoll
# NOISER = hs.noiser.open_es.OpenES
MODEL = hs.models.common.MLP

sigma = 0.2
lr = 0.03
num_epochs = 10
num_envs = 64

in_dim = 3

key = jax.random.key(0)
model_key = jax.random.fold_in(key, 0)
es_key = jax.random.fold_in(key, 1)
data_key = jax.random.fold_in(key, 2)

frozen_params, params, scan_map, es_map = MODEL.rand_init(model_key, in_dim=in_dim, out_dim=1, hidden_dims=[16, 16], use_bias=True, activation="relu", dtype="float32")
es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)
frozen_noiser_params, noiser_params = NOISER.init_noiser(params, sigma, lr, solver=optax.adamw, solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=8)

# inputs are noiser_params, params, iterinfo, input
jit_forward = jax.jit(jax.vmap(lambda n, p, i, x: MODEL.forward(NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, i, x), in_axes=(None, None, 0, 0)))
# inputs are noiser_params, params, input
jit_forward_eval = jax.jit(jax.vmap(lambda n, p, x: MODEL.forward(NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, None, x), in_axes=(None, None, 0)))
# inputs are noiser_params, params, fitnesses, iterinfo
jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(frozen_noiser_params, n, p, es_tree_key, f, i, es_map))

@jax.jit
def batch_calculate_fitnesses(x):
    return -((x - 2) ** 2)[:, 0]

for epoch in range(num_epochs):
    data_key, _key = jax.random.split(data_key)
    input_batch = jax.random.normal(_key, (num_envs, in_dim))
    iterinfo = (jnp.full(num_envs, epoch, dtype=jnp.int32), jnp.arange(num_envs))
    
    validations_batch = jit_forward_eval(noiser_params, params, input_batch)
    raw_validation_scores = batch_calculate_fitnesses(validations_batch)
    print(f"Avg validation score is {jnp.mean(raw_validation_scores)}")

    outputs_batch = jit_forward(noiser_params, params, iterinfo, input_batch)
    raw_scores = batch_calculate_fitnesses(outputs_batch)
    print(f"\tAverage score is {jnp.mean(raw_scores)}; min score is {jnp.min(raw_scores)}; max score is {jnp.max(raw_scores)}")

    fitnesses = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
    noiser_params, new_params = jit_update(noiser_params, params, fitnesses, iterinfo)

    parameter_differences = jax.tree.map(lambda x, y: jnp.sqrt(jnp.mean((x-y) ** 2)), params, new_params)
    lora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == 1 else 0.0, parameter_differences, es_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == 1 else 0.0, es_map))
    nonlora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == 0 else 0.0, parameter_differences, es_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == 0 else 0.0, es_map))
    print(f"\tlora differences: {lora_updates}; nonlora differences: {nonlora_updates}")

    params = new_params
