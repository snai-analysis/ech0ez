from pathlib import Path
from typing import ClassVar

import arviz as az
import numpy as np
import pandas as pd
import xarray as xa
from clipppy.contrib.sbi_helper import SBIHelper, SavedProperty, PickledProperty
from frozendict import frozendict
from jax import tree
from jax.numpy import stack
from jax.random import PRNGKey, split
from numpyro import infer
from numpyro.contrib.funsor import config_enumerate
from numpyro.infer.util import initialize_model

from .model_numpyro import Model


class Ech0ezHelper(SBIHelper):
    labels: ClassVar = {
        'M0': '$M_0$', 'sigma_res': r'$\sigma_0$', 'alpha': r'$\alpha$', 'beta': r'$\beta$', 'alpha_c': r'$\alpha_c$',
        'j0': '$j_0$', 'q0': '$q_0$',
        'mean_x1': r'$\bar{x}^{(0)}$', 'sigma_x1': r'$\sigma_x^{(0)}$',
        'mean_x2': r'$\bar{x}^{(1)}$', 'sigma_x2': r'$\sigma_x^{(1)}$',
        'xfrac': r'$P_0$',
        'mean_x': r'$\bar{x}$', 'sigma_x': r'$\sigma_x$', 'mean_c': r'$\bar{c}$', 'sigma_c': r'$\sigma_c$',
        'mean_R': r'$\bar{R}$', 'sigma_R': r'$\sigma_R$', 'R_V': '$R_V$', 'tau': r'$\tau_A$',
        'x_int': r'$x_{\rm int}$', 'c_int': r'$c_{\rm int}$', 'A_V': r'$A_V$', 'deltaM': r'$\delta M$',
    }

    model = None

    base_datadir: ClassVar = Path('data/pkg')
    data = SavedProperty(lambda self: self.base_datadir / f'{self.basename}.npz', np.load, None)
    hmc_res: xa.Dataset = PickledProperty(lambda self: self.base_resdir / self.basename / f'{self.model.name}.pickle')

    def set_model(self, **kwargs):
        data = self.data
        self.model = Model(
            obs=data['obs'], cov=data['cov'],
            z_hf_mean=data['z'], z_hf_cov=data['z_cov'],
            calib_idx=data['calib_idx'],
            distmod_calib_mean=data['distmod_calib_mean'],
            distmod_calib_cov=data['distmod_calib_cov'],
            calib_mask=~data['hf_mask'], hf_mask=data['hf_mask'],
            logmass=data['logmass'],
            distmod_hf_err=data['distmod_err'],
            **kwargs
        )
        return self

    def hmc_run(
        self, sample_keys, init_values=frozendict(), *,
        adapt_step_size=True, step_size=1e-2, target_accept_prob=0.95,
        adapt_mass_matrix=True,
        max_tree_depth=10, num_warmup=1000, num_samples=1000, num_chains=4,
        nuts_kwargs=frozendict(), mcmc_kwargs=frozendict(),
        rng_seed=0
    ):
        cmodel = config_enumerate(self.model)

        if not isinstance(init_values, list):
            init_values = num_chains * [init_values]

        key, *init_keys = split(PRNGKey(rng_seed), 1+num_chains)

        init_params = tree.map(lambda *zs: stack(zs), *(initialize_model(
            init_key, cmodel, init_strategy=infer.init_to_value(values={key: vals[key] for key in sample_keys})
        ).param_info.z for init_key, vals in zip(init_keys, init_values)))

        hmc = infer.MCMC(
            infer.NUTS(
                cmodel,
                adapt_step_size=adapt_step_size, step_size=step_size, target_accept_prob=target_accept_prob,
                adapt_mass_matrix=adapt_mass_matrix,
                dense_mass=[tuple(sorted(self.model.init.keys() & sample_keys))],
                # init_strategy=infer.init_to_value(values={key: init_values[key] for key in sample_keys}),
                max_tree_depth=max_tree_depth,
                **nuts_kwargs
            ),
            num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
            **mcmc_kwargs,
        )
        hmc.run(key, init_params=init_params)
        return hmc

    @staticmethod
    def hmc_get_samples(hmc):
        return xa.Dataset({
            key: xa.DataArray(val, dims=('chain', 'draw', ('hf' if key.endswith('_hf') else 'calib' if key.endswith('_calib') else 'SN'))[:len(val.shape)])
            for key, val in hmc.get_samples(group_by_chain=True).items()
            if not key.endswith('_')
        })

    @staticmethod
    def stats_global(ds, keys):
        return pd.DataFrame({
            'ess': az.ess(ds, var_names=keys).to_pandas(),
            'rhat': 100 * (az.rhat(ds, var_names=keys).to_pandas()-1),
            'mean': ds[keys].mean(('chain', 'draw')).to_pandas(),
            'std': ds[keys].std(('chain', 'draw')).to_pandas(),
        }).rename_axis(index='param')
