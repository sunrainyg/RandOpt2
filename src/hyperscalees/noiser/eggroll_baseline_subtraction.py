import jax
import optax
import jax.numpy as jnp
from .base_noiser import Noiser

from functools import partial

import optax

def _is_zero_noise(thread_id, group_size):
    tid = (thread_id % group_size)
    return (tid == 0) | (tid == 1)

def get_nonlora_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    epoch, thread_id = iterinfo
    G = jnp.asarray(frozen_noiser_params["group_size"])
    mask = (jnp.mod(thread_id, G) >= 2).astype(param.dtype)

    true_epoch = jnp.where(frozen_noiser_params["noise_reuse"] == 0, 0, epoch // frozen_noiser_params["noise_reuse"])
    true_thread_idx = thread_id // 2
    sign = jnp.where((thread_id % 2) == 0, 1.0, -1.0)
    sigma = base_sigma * sign

    eps = jax.random.normal(
        jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx),
        param.shape, dtype=param.dtype
    )
    return eps * sigma * mask  # zeros out for dirs 0 and 1

def get_lora_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    epoch, thread_id = iterinfo
    G = jnp.asarray(frozen_noiser_params["group_size"])
    a, b = param.shape
    r = frozen_noiser_params["rank"]
    mask = (jnp.mod(thread_id, G) >= 2).astype(param.dtype)

    true_epoch = jnp.where(frozen_noiser_params["noise_reuse"] == 0, 0, epoch // frozen_noiser_params["noise_reuse"])
    true_thread_idx = thread_id // 2
    sign = jnp.where((thread_id % 2) == 0, 1.0, -1.0)
    sigma = base_sigma * sign

    lora_params = jax.random.normal(
        jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx),
        (a + b, r), dtype=param.dtype
    )
    B = lora_params[:b]             
    A = lora_params[b:] * mask      
    B = B * mask                      
    return A * sigma, B

def _simple_lora_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    A, B = jax.vmap(partial(get_lora_update_params, frozen_noiser_params), in_axes=(None, 0, None, None))(base_sigma / jnp.sqrt(frozen_noiser_params["rank"]), iterinfo, param, key)
    broadcasted_scores = jnp.reshape(scores, scores.shape + (1,1))
    A = broadcasted_scores * A # N x a x r for A vs N x b x r for B -> final update is just a x b
    num_envs = scores.shape[0]
    print("LORA UPDATE", A.shape, B.shape)
    # return A.T @ B / num_envs
    return jnp.einsum('nir,njr->ij', A, B) / num_envs

def _simple_full_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    if frozen_noiser_params["freeze_nonlora"]:
        return jnp.zeros_like(param)
    updates = jax.vmap(
        partial(get_nonlora_update_params, frozen_noiser_params),
        in_axes=(None, 0, None, None)
    )(base_sigma, iterinfo, param, key)
    broadcasted_scores = jnp.reshape(scores, scores.shape + (1,) * len(param.shape))
    return jnp.astype(jnp.mean(broadcasted_scores * updates, axis=0), param.dtype)

def _noop_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    return jnp.zeros_like(param)

class EggRollBS(Noiser):
    @classmethod
    def init_noiser(cls, params, sigma, lr, *args, solver=None, solver_kwargs=None, group_size=0, freeze_nonlora=False, noise_reuse=0, rank=1, trust_region_norm=1.0, **kwargs):
        """
        Return frozen_noiser_params and noiser_params
        """
        if solver is None:
            solver = optax.sgd
        if solver_kwargs is None:
            solver_kwargs = {}
        base_solver = solver(lr, **solver_kwargs)
        true_solver = optax.chain(
            optax.clip_by_global_norm(trust_region_norm),
            base_solver,
        )
        opt_state = true_solver.init(params)
        
        return {"group_size": group_size, "freeze_nonlora": freeze_nonlora, "noise_reuse": noise_reuse, "solver": true_solver, "rank": rank, "trust_region_norm": trust_region_norm}, {"sigma": sigma, "opt_state": opt_state}
    
    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        base_ans = x @ param.T
        if iterinfo is None:
            return base_ans
        A, B = get_lora_update_params(frozen_noiser_params, noiser_params["sigma"] / jnp.sqrt(frozen_noiser_params["rank"]), iterinfo, param, base_key)
        return base_ans + x @ B @ A.T

    @classmethod
    def do_Tmm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        base_ans = x @ param
        if iterinfo is None:
            return base_ans
        A, B = get_lora_update_params(frozen_noiser_params, noiser_params["sigma"] / jnp.sqrt(frozen_noiser_params["rank"]), iterinfo, param, base_key)
        return base_ans + x @ A @ B.T

    @classmethod
    def do_emb(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        # return param[x]
        raise NotImplementedError("Embedding is not implemented")

    @classmethod
    def get_noisy_standard(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo):
        if iterinfo is None or frozen_noiser_params["freeze_nonlora"]:
            return param
        return param + get_nonlora_update_params(frozen_noiser_params, noiser_params["sigma"], iterinfo, param, base_key)

    @classmethod
    def convert_fitnesses(cls, frozen_noiser_params, noiser_params, raw_scores, num_episodes_list=None):
        eps = 1e-8
        G = int(frozen_noiser_params["group_size"])
        if raw_scores.ndim == 1:
            assert raw_scores.size % G == 0
            Q = raw_scores.size // G
            S = raw_scores.reshape(Q, G)
        else:
            S = raw_scores
            assert S.shape[1] == G
        b = S[:, :1]
        Z = (S - b) / (jnp.std(S, axis=1, keepdims=True) + eps)
        Z = Z.at[:, 0].set(0.0)
        Z = Z.at[:, 1].set(0.0)
        per_dir_fitness = jnp.mean(Z, axis=0)
        return per_dir_fitness

    @classmethod
    def _do_update(cls, param, base_key, fitnesses, iterinfos, map_classification, sigma, frozen_noiser_params, **kwargs):
        update_fn = [_simple_full_update, _simple_lora_update, _noop_update, _noop_update][map_classification]

        if len(base_key.shape) == 0:
            new_grad = update_fn(sigma, param, base_key, fitnesses, iterinfos, frozen_noiser_params)
        else:
            new_grad = jax.lax.scan(lambda _, x: (0, update_fn(sigma, x[0], x[1], fitnesses, iterinfos, frozen_noiser_params)), 0, xs=(param, base_key))[1]

        # return (param + new_grad * lr * jnp.sqrt(fitnesses.size)).astype(param.dtype)
        return -(new_grad * jnp.sqrt(fitnesses.size)).astype(param.dtype)

    @classmethod
    def do_updates(cls, frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map):
        new_grad = jax.tree.map(lambda p, k, m: cls._do_update(p, k, fitnesses, iterinfos, m, noiser_params["sigma"], frozen_noiser_params), params, base_keys, es_map)
        updates, noiser_params["opt_state"] = frozen_noiser_params["solver"].update(new_grad, noiser_params["opt_state"], params)
        return noiser_params, optax.apply_updates(params, updates)
