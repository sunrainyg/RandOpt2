class Noiser:
    @classmethod
    def init_noiser(cls, params, sigma, lr, *args, solver=None, solver_kwargs=None, **kwargs):
        """
        params: parameters of the model
        sigma: initial sigma of noiser
        lr: learning rate of optimization process
        solver (optional, keyword arg): the optax solver to use (i.e. optax.adamw)
        solver_kwargs (optional, keyword arg): the optax solver's keyword arguments (excluding learning rate)
        
        Return frozen_noiser_params and noiser_params
        """
        return {}, {}
    
    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        return x @ param.T

    @classmethod
    def do_Tmm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        return x @ param

    @classmethod
    def do_emb(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        return param[x]

    @classmethod
    def get_noisy_standard(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo):
        return param

    @classmethod
    def convert_fitnesses(cls, frozen_noiser_params, noiser_params, raw_scores, num_episodes_list=None):
        return raw_scores

    @classmethod
    def do_updates(cls, frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map):
        return noiser_params, params
