import os
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

import time

import hyperscalees as hs

from hyperscalees.models.llm.auto import get_model
from hyperscalees.models.common import simple_es_tree_key



# NOISER = hs.noiser.base_noiser.Noiser
NOISER = hs.noiser.eggroll.EggRoll

base_model_key = jax.random.key(0)

RWKV, full_params, _ = get_model("7g0.1B")
tokenizer = hs.models.llm.tokenizer.LegacyWorldTokenizer()
config, params, scan_map, es_map = full_params
params = jax.device_put(params, jax.local_devices()[0])

frozen_noiser_params, noiser_params = NOISER.init_noiser(params, 0.001, None)
base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)
# print(jax.tree.map(lambda x, y, z: (x.shape, y, z), params, scan_map, es_map))
# print(base_evo_keys)


# print(jax.tree.map(lambda x: (jnp.max(jnp.abs(x)), jnp.mean(jnp.abs(x))), params))





context = "The Eiffel tower is in the city of"
answer = " Paris"
encoded = tokenizer.encode(context)
print(context)

init_state = RWKV.default_state(params, config)
print(init_state.shape)

forward = partial(RWKV.forward, NOISER, frozen_noiser_params, noiser_params, config)

start_time = time.time()
out, state = jax.block_until_ready(forward(params, base_evo_keys, (0, 1), encoded, init_state))
end_time = time.time()
print(f"Forward time: {end_time - start_time} seconds (note: much faster with jax.jit)")
out = out[len(encoded)-1]
soft_out = jax.nn.softmax(out)
values, indices = jax.lax.top_k(soft_out, 10)
for i in range(10):
    print(f"{values[i].item() * 100}%: {tokenizer.decode([indices[i].item()])}")

