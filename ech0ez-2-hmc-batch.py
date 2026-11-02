def main(basename, s, num_chains=None):
    import numpy as np
    import numpyro
    numpyro.set_host_device_count(num_chains)

    import jax
    jax.config.update('jax_platforms', 'cpu')

    from libech0ez import ztf_offz, ztf_offerr
    from libech0ez.helper import Ech0ezHelper

    h = Ech0ezHelper(basename=basename).set_model(**(dict(
        photcalib_zs=ztf_offz[:3],
        photcalib_dms_mean=0*ztf_offerr[:3],
        photcalib_dms_cov=ztf_offerr[:3]**2 * np.eye(3),
    ) if '-sm' in basename else {}), **s)

    if not h.hmc_res_name.exists():
        sample_keys, init_values = h.model.sim_init(x_int=h.data['obs'][:, 1])
        hmc = h.hmc_run(
            sample_keys, init_values,
            num_warmup=4000//num_chains, num_samples=4000//num_chains, num_chains=num_chains,
        )
        h.hmc_res = h.hmc_get_samples(hmc)


if __name__ == '__main__':
    import argparse
    from functools import partial
    from itertools import product
    from multiprocessing import Pool, set_start_method
    from pathlib import Path


    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='num_chains', type=int, default=4)
    parser.add_argument('-b', dest='baseline', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('fname', nargs='*')
    args = parser.parse_args()

    set_start_method('spawn')

    if args.baseline:
        variants = [{
            'cosmo_type': 'hqj',
            'z_mode': 'lat',
            'RV_mode': 'dist',
            'use_AV': True,
            'use_2x': False,
            'dostep': False
        }]
    else:
        settings = {
            'cosmo_type': ('hqj',),
            'z_mode': ('lat',),
            'RV_mode': ('dist', 'global'),
            'use_AV': (True,),
            'use_2x': (True, False),
            'dostep': (True, False)
        }
        variants = [dict(zip(settings.keys(), vals)) for vals in product(*settings.values())]

    basenames = [Path(f).stem for f in args.fname]

    print(len(basenames), basenames)
    print(len(variants), variants)
    with Pool(16, maxtasksperchild=4) as p:
        p.starmap(partial(main, num_chains=args.num_chains), product(basenames, variants))

