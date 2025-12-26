import jax
import jax.numpy as jnp

from functools import partial

from .llm import LLM
from ..base_model import Model, CommonParams
from ..common import PARAM, MM_PARAM, EMB_PARAM, EXCLUDED, Parameter, MM, TMM, Embedding, Linear, call_submodule


class Qwen2RMSNorm(Model):
    @classmethod
    def _forward(cls, common_params, x, eps=1e-6):
        hidden_states = x
        variance = jnp.mean(hidden_states ** 2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + eps)

        return call_submodule(Parameter, 'weight', common_params) * hidden_states


class Qwen2MLP(Model):
    @classmethod
    def _forward(cls, common_params, x):
        return call_submodule(Linear, 'down_proj', common_params, jax.nn.silu(call_submodule(Linear, 'gate_proj', common_params, x)) * call_submodule(Linear, 'up_proj', common_params, x))

class RWKV6Attention(Model):
    @classmethod
    def _forward(cls, common_params, x, state, length, new_starts, H, S, inner_loop):
        T, C = x.shape

        sx = jnp.concatenate([state[:1], x[:-1, :]], dtype=x.dtype)
        sx = jnp.where(new_starts[:, None], jnp.zeros_like(sx), sx)
        sx = sx - x
        xxx = x + sx * call_submodule(Parameter, 'time_maa_x', common_params)
        xxx = jnp.tanh(call_submodule(TMM, 'time_maa_w1', common_params, xxx)).reshape(T, 5, -1).transpose(1, 0, 2)
        xxx = jax.vmap(lambda x, y: x @ y)(xxx, common_params.params['time_maa_w2']).reshape(5, T, -1) # TODO: Fix
        mr, mk, mv, mw, mg = xxx

        xr = x + sx * (call_submodule(Parameter, 'time_maa_r', common_params) + mr)
        xk = x + sx * (call_submodule(Parameter, 'time_maa_k', common_params) + mk)
        xv = x + sx * (call_submodule(Parameter, 'time_maa_v', common_params) + mv)
        xw = x + sx * (call_submodule(Parameter, 'time_maa_w', common_params) + mw)
        xg = x + sx * (call_submodule(Parameter, 'time_maa_g', common_params) + mg)
        state = state.at[0].set(x[length-1])

        r = jnp.reshape(call_submodule(Linear, 'q_proj', common_params, xr), (T, H, S)) # query_states
        k = jnp.reshape(call_submodule(Linear, 'k_proj', common_params, xk), (T, -1, S)) # key_states
        v = jnp.reshape(call_submodule(Linear, 'v_proj', common_params, xv), (T, -1, S)) # value_states

        w_lora_result = call_submodule(Parameter, 'time_decay', common_params).reshape(1, H, S, 1) + call_submodule(TMM, 'time_decay_w2', common_params, jnp.tanh(call_submodule(TMM, 'time_decay_w1', common_params, xw))).reshape(T, H, S, 1) # decay_states
        
        g = jax.nn.sigmoid(call_submodule(Linear, 'gate', common_params, xg)) # gate_states

        num_kv_reps = r.shape[-2] // k.shape[-2]
        k = jnp.repeat(k, num_kv_reps, axis=-2)
        v = jnp.repeat(v, num_kv_reps, axis=-2)

        log_w = -jnp.exp(w_lora_result) # decay_states_log
        log_w = jnp.clip(log_w, min=-5.0)
        k = k * (1 - jnp.exp(log_w[:, :, :, 0]))

        # scale = r.shape[-1] ** -0.5
        s = jnp.reshape(state[1:, :],(H, S, S))
        state_new, out = inner_loop(r, k, v, log_w, None, s, length, new_starts)
        state = state.at[1:].set(state_new.reshape(S, -1))

        x = out.reshape(T, H*S) * g
        return call_submodule(Linear, 'o_proj', common_params, x), state

class BaseRWKV(LLM):
    @classmethod
    def transform_torch_model(cls, torch_model, dtype=jnp.bfloat16):
        import torch
        import re
        w = torch_model
        keys = list(w.keys())
        for k in keys:
            k_new = k.replace("model.", "").replace("layers.", "blocks.")
            if 'time_' in k_new:
                w[k] = w[k].squeeze()
            # if 'time_maa_w2' in k_new:
                # w[k] = w[k].reshape((-1, w[k].shape[-1]))
            if k_new != k:
                w[k_new] = w[k]
                del w[k]
        return w

    @classmethod
    def get_scan_map(cls, config):
        BS = (0,)
        NS = tuple()
        return {
            'blocks': {
                'input_layernorm': {'weight': BS},
                'mlp': {'down_proj': {'weight': BS}, 'gate_proj': {'weight': BS}, 'up_proj': {'weight': BS}},
                'post_attention_layernorm': {'weight': BS},
                'self_attn': {
                    'gate': {'weight': BS},
                    'k_proj': {'bias': BS, 'weight': BS},
                    'o_proj': {'weight': BS},
                    'q_proj': {'bias': BS, 'weight': BS},
                    'time_decay': BS,
                    'time_decay_w1': BS,
                    'time_decay_w2': BS,
                    'time_maa_g': BS,
                    'time_maa_k': BS,
                    'time_maa_r': BS,
                    'time_maa_v': BS,
                    'time_maa_w': BS,
                    'time_maa_w1': BS,
                    'time_maa_w2': (0, 1),
                    'time_maa_x': BS,
                    'v_proj': {'bias': BS, 'weight': BS}}
            },
            'embed_tokens': {'weight': NS},
            'lm_head': {'weight': NS},
            'norm': {'weight': NS}
        }
        # return {
        #     'blocks': {
        #         'att': {'a0': BS, 'a1': BS, 'a2': BS, 'g1': BS, 'g2': BS, 'k_a': BS, 'k_k': BS, 'key': {'weight': BS},
        #                 'ln_x': {'bias': BS, 'weight': BS}, 'output': {'weight': BS},
        #                 'r_k': BS, # BS EXCEPTION
        #                 'receptance': {'weight': BS},
        #                 'v0': BS, 'v1': BS, 'v2': BS,
        #                 'value': {'weight': BS},
        #                 'w0': BS, 'w1': BS, 'w2': BS, 'x_a': BS, 'x_g': BS, 'x_k': BS, 'x_r': BS, 'x_v': BS, 'x_w': BS},
        #         'ffn': {'key': {'weight': BS}, 'value': {'weight': BS}, 'x_k': BS},
        #         'ln1': {'bias': BS, 'weight': BS}, 'ln2': {'bias': BS, 'weight': BS}},
        #     'emb': {'weight': NS},
        #     'head': {'weight': NS},
        #     'ln0': {'bias': NS, 'weight': NS},
        #     'ln_out': {'bias': NS, 'weight': NS}
        # }

    @classmethod
    def get_es_map(cls, config):
        LORA = MM_PARAM
        FULL = PARAM
        return {
            'blocks': {
                'input_layernorm': {'weight': FULL},
                'mlp': {'down_proj': {'weight': LORA}, 'gate_proj': {'weight': LORA}, 'up_proj': {'weight': LORA}},
                'post_attention_layernorm': {'weight': FULL},
                'self_attn': {
                    'gate': {'weight': LORA},
                    'k_proj': {'bias': FULL, 'weight': LORA},
                    'o_proj': {'weight': LORA},
                    'q_proj': {'bias': FULL, 'weight': LORA},
                    'time_decay': FULL,
                    'time_decay_w1': LORA,
                    'time_decay_w2': LORA,
                    'time_maa_g': FULL,
                    'time_maa_k': FULL,
                    'time_maa_r': FULL,
                    'time_maa_v': FULL,
                    'time_maa_w': FULL,
                    'time_maa_w1': LORA,
                    'time_maa_w2': EXCLUDED, # TODO: FIX
                    'time_maa_x': FULL,
                    'v_proj': {'bias': FULL, 'weight': LORA}}
            },
            'embed_tokens': {'weight': EXCLUDED},
            'lm_head': {'weight': EXCLUDED},
            'norm': {'weight': FULL}
        }
        # return {
        #     'blocks': {
        #         'att': {'a0': FULL, 'a1': LORA, 'a2': LORA, 'g1': LORA, 'g2': LORA, 'k_a': FULL, 'k_k': FULL, 'key': {'weight': LORA},
        #                 'ln_x': {'bias': FULL, 'weight': FULL}, 'output': {'weight': LORA},
        #                 'r_k': FULL, # LORA EXCEPTION
        #                 'receptance': {'weight': LORA},
        #                 'v0': FULL, 'v1': LORA, 'v2': LORA,
        #                 'value': {'weight': LORA},
        #                 'w0': FULL, 'w1': LORA, 'w2': LORA, 'x_a': FULL, 'x_g': FULL, 'x_k': FULL, 'x_r': FULL, 'x_v': FULL, 'x_w': FULL},
        #         'ffn': {'key': {'weight': LORA}, 'value': {'weight': LORA}, 'x_k': FULL},
        #         'ln1': {'bias': FULL, 'weight': FULL}, 'ln2': {'bias': FULL, 'weight': FULL}},
        #     'emb': {'weight': EXCLUDED},
        #     'head': {'weight': EXCLUDED},
        #     'ln0': {'bias': FULL, 'weight': FULL},
        #     'ln_out': {'bias': FULL, 'weight': FULL}
        # }

    @classmethod
    def default_state(cls, params, config):
        n_embd = params['embed_tokens']['weight'].shape[1]
        n_layer = params['blocks']['input_layernorm']['weight'].shape[0]
        head_size = config["head_size"]
        n_head = n_embd // head_size
        return jnp.zeros((n_layer, (1 + head_size), n_embd), dtype=params['embed_tokens']['weight'].dtype)

    @classmethod
    def embed(cls, common_params, tokens):
        # TODO: Make this modifiable
        # return common_params.params['emb']['weight'][tokens.ravel()]
        return common_params.params['embed_tokens']['weight'][tokens.ravel()]
    
    @classmethod
    def outhead(cls, common_params, x):
        # TODO: Make this modifiable
        x = call_submodule(Qwen2RMSNorm, 'norm', common_params, x)
        return x @ common_params.params['lm_head']['weight'].T

    @classmethod
    def inner_loop(cls, r, k, v, w, time_first, s, length, new_starts):
        # q, k, v, gk, _, initial_state
        scale = r.shape[-1] ** -0.5
        w = jnp.exp(w)
        out = jnp.empty_like(r)
        out_s = s

        reset_s = jnp.zeros_like(s)
        for t in range(r.shape[0]):
            s = jax.lax.select(new_starts[t], reset_s, s)
            
            rt = jnp.expand_dims(r[t], 1) * scale
            kt = jnp.expand_dims(k[t], 2)
            vt = jnp.expand_dims(v[t], 1)
            at = kt*vt
            s = jnp.astype(at + w[t] * s, r.dtype)
            out = out.at[t].set((rt @ s).squeeze(1))
            out_s = jax.lax.select(t < length, s, out_s)

        return out_s, out

    @classmethod
    def forward_seq(cls, common_params, x, state, length, new_starts):
        params = common_params.params
        config = common_params.frozen_params
        n_embd = params['embed_tokens']['weight'].shape[1]
        n_layer = params['blocks']['input_layernorm']['weight'].shape[0]
        head_size = config["head_size"]
        n_head = n_embd // head_size

        @partial(jax.checkpoint,
                 policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        def block_loop(x, inputs):
            hidden_states = x
            params_i, es_tree_key_i, state = inputs
            block_i = common_params._replace(
                params=params_i,
                es_tree_key=es_tree_key_i
            )

            residual = hidden_states
            hidden_states = call_submodule(Qwen2RMSNorm, 'input_layernorm', block_i, hidden_states)
            hidden_states, state = call_submodule(RWKV6Attention, 'self_attn', block_i, hidden_states, state, length, new_starts, n_head, head_size, cls.inner_loop)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = call_submodule(Qwen2RMSNorm, 'post_attention_layernorm', block_i, hidden_states)
            hidden_states = call_submodule(Qwen2MLP, 'mlp', block_i, hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states, state

        x, state = jax.lax.scan(block_loop, x, (common_params.params['blocks'], common_params.es_tree_key['blocks'], state))

        return x, state


class FastRWKV(BaseRWKV):
    @classmethod
    def default_state(cls, params, config):
        n_embd = params['embed_tokens']['weight'].shape[1]
        n_layer = params['blocks']['input_layernorm']['weight'].shape[0]
        head_size = config["head_size"]
        n_head = n_embd // head_size
        return jnp.zeros((n_layer, (1 + head_size), n_embd), dtype=params['embed_tokens']['weight'].dtype)
        # return [jnp.zeros(((1 + head_size), n_embd), dtype=params['embed_tokens']['weight'].dtype)] * n_layer

    @classmethod
    def forward_seq(cls, common_params, x, state, length, new_starts):
        params = common_params.params
        config = common_params.frozen_params
        n_embd = params['embed_tokens']['weight'].shape[1]
        n_layer = params['blocks']['input_layernorm']['weight'].shape[0]
        head_size = config["head_size"]
        n_head = n_embd // head_size

        for i in range(n_layer):
            hidden_states = x
            # params_i, es_tree_key_i, state = inputs
            params_i = jax.tree.map(lambda a: a[i], common_params.params['blocks'])
            es_tree_key_i = jax.tree.map(lambda a: a[i], common_params.es_tree_key['blocks'])
            state_i = state[i]
            block_i = common_params._replace(
                params=params_i,
                es_tree_key=es_tree_key_i
            )

            residual = hidden_states
            hidden_states = call_submodule(Qwen2RMSNorm, 'input_layernorm', block_i, hidden_states)
            hidden_states, state_i = call_submodule(RWKV6Attention, 'self_attn', block_i, hidden_states, state_i, length, new_starts, n_head, head_size, cls.inner_loop)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = call_submodule(Qwen2RMSNorm, 'post_attention_layernorm', block_i, hidden_states)
            hidden_states = call_submodule(Qwen2MLP, 'mlp', block_i, hidden_states)
            hidden_states = residual + hidden_states

            x = hidden_states
            state = state.at[i].set(state_i)
            # state[i] = state_i

        return x, state
