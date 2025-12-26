import jax
import optax
import jax.numpy as jnp
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from hyperscalees.noiser.base_noiser import Noiser

from functools import partial

def get_sparse_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    epoch, thread_id = iterinfo

    true_epoch = 0 if frozen_noiser_params["noise_reuse"] == 0 else epoch // frozen_noiser_params["noise_reuse"]

    true_thread_idx = thread_id // 2
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    a, b = param.shape
    k = max(a,b)

    # Create deterministic key from epoch and thread, then split into multiple keys
    deterministic_key = jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx)
    key1, key2 = jax.random.split(deterministic_key, 2)

    # Create idx and sparse vector
    idxjoint = jnp.floor(jax.random.uniform(key1, (k,)) * (a * b)).astype(jnp.int32)
    idxa = idxjoint // b
    idxb = idxjoint % b
    sparse_vector = jax.random.normal(key2, shape=(k,), dtype=param.dtype)

    # Slower version:
    # key1, key2, key3 = jax.random.split(deterministic_key, 3)
    # idxa = jax.random.choice(key1, a, shape=(k,))
    # idxb = jax.random.choice(key2, b, shape=(k,))
    # sparse_vector = jax.random.normal(key3, shape=(k,), dtype=param.dtype)

    return sparse_vector * sigma, idxa, idxb

def get_lora_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    epoch, thread_id = iterinfo

    true_epoch = 0 if frozen_noiser_params["noise_reuse"] == 0 else epoch // frozen_noiser_params["noise_reuse"]

    true_thread_idx = thread_id // 2
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    a, b = param.shape
    lora_params = jax.random.normal(jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx), (a+b, frozen_noiser_params["rank"]), dtype=param.dtype) # (a+b, r)
    B = lora_params[:b] # b x r
    A = lora_params[b:] # a x r

    # update is A @ B.T
    return A * sigma, B

def get_nonlora_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    epoch, thread_id = iterinfo

    true_epoch = 0 if frozen_noiser_params["noise_reuse"] == 0 else epoch // frozen_noiser_params["noise_reuse"]

    true_thread_idx = thread_id // 2
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    updates = jax.random.normal(jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx), param.shape, dtype=param.dtype)
    return updates * sigma

def _simple_full_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    if frozen_noiser_params["freeze_nonlora"]:
        return jnp.zeros_like(param)
    _, thread_ids = iterinfo
    sigma = jnp.where(thread_ids % 2 == 0, base_sigma, -base_sigma)
    updates = jax.vmap(partial(get_nonlora_update_params, frozen_noiser_params), in_axes=(None, 0, None, None))(base_sigma, iterinfo, param, key) # N x a x b
    broadcasted_scores = jnp.reshape(scores, scores.shape + (1,) * len(param.shape)) # N x 1 x 1
    broadcasted_sigma = jnp.reshape(sigma, sigma.shape + (1,) * len(param.shape))
    # return jnp.astype(jnp.mean(broadcasted_scores * updates / broadcasted_sigma ** 2, axis=0), param.dtype)
    return jnp.astype(jnp.mean(broadcasted_scores * updates, axis=0), param.dtype) # (a, b)

def _simple_sparse_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    # sparse_vectors, idxa, idxb: (N, k)
    sparse_vectors, idxa, idxb = jax.vmap(partial(get_sparse_update_params, frozen_noiser_params), in_axes=(None, 0, None, None))(base_sigma, iterinfo, param, key)

    a, b = param.shape
    N, k = sparse_vectors.shape
    # To be verified: normalize by the proportion of non-zero elements in the sparse matrices
    q_normalizer = k / (a*b) # eg k = max(a,b) -> q = 1/min(a,b)
    # q_normalizer = 1
    
    broadcasted_scores = jnp.reshape(scores, (N, 1))  # (N, 1)
    weighted_vectors = broadcasted_scores * sparse_vectors  # (N, 1) * (N, k) = (N, k)
    weighted_matrices = jnp.zeros((N, a, b), dtype=param.dtype)
    weighted_matrices = weighted_matrices.at[jnp.arange(N)[:, None], idxa, idxb].add(weighted_vectors)
    
    # Alternative attempt // ignore this
    # sparse_matrices, = jax.vmap(partial(get_sparse_update_params, frozen_noiser_params), in_axes=(None, 0, None, None))(base_sigma, iterinfo, param, key)
    # a, b = param.shape
    # q = 1/min(a,b)
    # broadcasted_scores = jnp.reshape(scores, scores.shape + (1,1))  # (N, 1, 1)
    # weighted_matrices = broadcasted_scores * sparse_matrices # (N, a, b)
    
    return jnp.mean(weighted_matrices, axis=0) * q_normalizer  # (a, b)

def _simple_lora_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    num_envs = scores.shape[0] # (N,)
    A, B = jax.vmap(partial(get_lora_update_params, frozen_noiser_params), in_axes=(None, 0, None, None))(base_sigma / jnp.sqrt(frozen_noiser_params["rank"]), iterinfo, param, key) # (N, a, r), (N, b, r)
    broadcasted_scores = jnp.reshape(scores, scores.shape + (1,1)) # (N, 1, 1)
    A = broadcasted_scores * A # (N, 1, 1) * (N, a, r) -> (N, a, r)
    print("LORA UPDATE", A.shape, B.shape)
    # return A.T @ B / num_envs
    return jnp.einsum('nar,nbr->ab', A, B) / num_envs # (a, b)

def _noop_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    return jnp.zeros_like(param)

class Sparse(Noiser):
    @classmethod
    def init_noiser(cls, params, sigma, lr, *args, solver=None, solver_kwargs=None, group_size=0, freeze_nonlora=False, noise_reuse=0, rank=1, q_multiplier=1, **kwargs):
        """
        Return frozen_noiser_params and noiser_params
        """
        if solver is None:
            solver = optax.sgd
        if solver_kwargs is None:
            solver_kwargs = {}
        true_solver = solver(lr, **solver_kwargs)
        opt_state = true_solver.init(params)
        
        return {"group_size": group_size, "freeze_nonlora": freeze_nonlora, "noise_reuse": noise_reuse, "solver": true_solver, "q_multiplier": q_multiplier}, {"sigma": sigma, "opt_state": opt_state}
    
    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        # x has shape (batch_size, b) and param has shape (a, b)
        base_ans = x @ param.T  # (batch_size, a)
        if iterinfo is None:
            return base_ans
        sparse_vector, idxa, idxb = get_sparse_update_params(frozen_noiser_params, noiser_params["sigma"], iterinfo, param, base_key)
        # idx1, idx2, sparse_vector: (n,)
        x_prod_sparse = x[idxb] * sparse_vector  # (batch_size, n) * (n,) -> (batch_size, n)
        return base_ans.at[idxa].add(x_prod_sparse)  # (batch_size, a)

    @classmethod
    def do_Tmm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        # x has shape (batch_size, b) and param has shape (b, a)
        base_ans = x @ param  # (batch_size, a)
        if iterinfo is None:
            return base_ans
        sparse_vector, idxa, idxb = get_sparse_update_params(frozen_noiser_params, noiser_params["sigma"], iterinfo, param, base_key)
        # idx1, idx2, sparse_vector: (n,)
        x_prod_sparse = x[idxb] * sparse_vector  # (batch_size, n) * (n,) -> (batch_size, n)
        return base_ans.at[idxa].add(x_prod_sparse)  # (batch_size, a)

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
        group_size = frozen_noiser_params["group_size"]
        if group_size == 0:
            true_scores = (raw_scores - jnp.mean(raw_scores, keepdims=True)) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)
        else:
            group_scores = raw_scores.reshape((-1, group_size))
            true_scores = (group_scores - jnp.mean(group_scores, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)
            true_scores = true_scores.ravel()
        # fitness = jax.nn.softmax(true_scores)
        # return fitness * raw_scores.size
        return true_scores

    @classmethod
    def _do_update(cls, param, base_key, fitnesses, iterinfos, map_classification, sigma, frozen_noiser_params, **kwargs):
        update_fn = [_simple_full_update, _simple_sparse_update, _noop_update, _noop_update][map_classification]

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
