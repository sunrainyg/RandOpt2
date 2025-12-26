import jax
import jax.numpy as jnp
import numpy as np

from .base_model import Model, CommonInit, CommonParams

from gymnax.environments import spaces
import distrax
from .common import Parameter, Embedding, MM, Linear, MLP, merge_inits, call_submodule, ACTIVATIONS

def get_space_defn(space):
    if isinstance(space, spaces.Discrete):
        return {
            'space': 'discrete',
            'start': 0,
            'n': int(space.n)
        }
    elif isinstance(space, spaces.Box) and space.dtype is jnp.float32:
        return {
            'space': 'continuous',
            'low': space.low if isinstance(space.low, jnp.ndarray) else space.low * jnp.ones(space.shape, dtype=space.dtype),
            'high': space.high if isinstance(space.high, jnp.ndarray) else space.high * jnp.ones(space.shape, dtype=space.dtype),
        }
    
    elif isinstance(space, spaces.Box):# and space.dtype.kind in ['i', 'u']:
        return {
            'space': 'boxdiscrete',
            'low': space.low if isinstance(space.low, jnp.ndarray) else space.low * jnp.ones(space.shape, dtype=space.dtype),
            'high': space.high if isinstance(space.high, jnp.ndarray) else space.high * jnp.ones(space.shape, dtype=space.dtype),
        }
    else:
        raise NotImplementedError(f'Unsupported space type {space}')

class InputProcessor(Model):
    @classmethod
    def rand_init(cls, key, n_embd, space_gymnax, dtype, *args, **kwargs):
        space = get_space_defn(space_gymnax)
        if space['space'] == 'discrete':
            return merge_inits(
                discrete=Embedding.rand_init(key, space['n'], n_embd, dtype)
            )
        elif space['space'] == 'continuous':
            in_size = np.prod(jnp.array(space['low']).shape)
            return merge_inits(
                continuous=MM.rand_init(key, in_size, n_embd, dtype)
            )
        else:
            raise NotImplementedError('Unsupported space type')

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        if 'discrete' in common_params.params:
            return call_submodule(Embedding, 'discrete', common_params, x)
        elif 'continuous' in common_params.params:
            return call_submodule(MM, 'continuous', common_params, x)
        else:
            raise NotImplementedError('Unsupported space type')

class OutputProcessor(Model):
    @classmethod
    def rand_init(cls, key, n_embd, space_gymnax, use_bias, dtype, *args, **kwargs):
        space = get_space_defn(space_gymnax)
        if space['space'] == 'discrete':
            out_size = space['n']
            return merge_inits(
                discrete=Linear.rand_init(key, n_embd, out_size, use_bias, dtype)
            )
        elif space['space'] == 'continuous':
            out_size = np.prod(jnp.array(space['low']).shape)
            return merge_inits(
                continuous=Linear.rand_init(key, n_embd, out_size, use_bias, dtype),
                actor_logtstd=Parameter.rand_init(key, None, None, jnp.zeros(out_size, dtype=dtype), dtype)
            )
        else:
            raise NotImplementedError('Unsupported space type')

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        # name = list(common_params.params.keys())[0]
        if 'discrete' in common_params.params:
            x = call_submodule(Linear, 'discrete', common_params, x)
            return distrax.Categorical(logits=x)
        elif 'continuous' in common_params.params:
            x = call_submodule(Linear, 'continuous', common_params, x)
            return distrax.MultivariateNormalDiag(x.astype(jnp.float32), jnp.exp(
                call_submodule(Parameter, 'actor_logtstd', common_params).astype(jnp.float32)
            ))
        else:
            raise NotImplementedError('Unsupported space type')
        
    
class ActorCriticMLP(Model):
    @classmethod
    def rand_init(cls, key, n_embd, obs_space, act_space, n_layers, use_bias, activation, have_critic, dtype, *args, **kwargs):
        obs_key, act_key, mlp_key, crit_key = jax.random.split(key, 4)
        if have_critic:
            return merge_inits(
                obs_embed=InputProcessor.rand_init(obs_key, n_embd, obs_space, dtype),
                act_head=OutputProcessor.rand_init(act_key, n_embd, act_space, use_bias, dtype),
                mlp=MLP.rand_init(mlp_key, n_embd, n_embd, [n_embd] * (n_layers - 1), use_bias, activation, dtype),
                critic_head=Linear.rand_init(crit_key, n_embd, 1, use_bias, dtype)
            )
        else:
            return merge_inits(
                obs_embed=InputProcessor.rand_init(obs_key, n_embd, obs_space, dtype),
                act_head=OutputProcessor.rand_init(act_key, n_embd, act_space, use_bias, dtype),
                mlp=MLP.rand_init(mlp_key, n_embd, n_embd, [n_embd] * (n_layers - 1), use_bias, activation, dtype),
            )

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        x = call_submodule(InputProcessor, 'obs_embed', common_params, x)
        x = ACTIVATIONS[common_params.frozen_params['mlp']['activation']](x)
        x = call_submodule(MLP, 'mlp', common_params, x)
        x = ACTIVATIONS[common_params.frozen_params['mlp']['activation']](x)
        pi = call_submodule(OutputProcessor, 'act_head', common_params, x)
        if 'critic_head' in common_params.params:
            critic = call_submodule(Linear, 'critic_head', common_params, x)
            return pi, critic
        return pi
        
