import numpy as np
import torch
from frozendict import frozendict
from pandas import DataFrame

# Lacroix+2025v1, fig. 12. This changed in arXiv's v2...
ztf_offz, ztf_off, ztf_offerr = np.transpose([
    [0.013298835705045278, 0.019660194174757284, 0.07427184466019418],
    [0.03358344113842174, 0.08446601941747572, 0.09320388349514562],
    [0.05404915912031048, 0.08446601941747572, 0.09174757281553397],
    [0.07433376455368694, 0.09466019417475728, 0.10631067961165049],
    [0.09461836998706338, 0.09320388349514562, 0.10048543689320388],
    [0.11490297542043985, 0.09466019417475728, 0.12087378640776698],
    [0.1353686934023286, 0.10558252427184465, 0.12233009708737863],
    [0.15601552393272966, 0.07645631067961164, 0.10412621359223301],
    [0.17611901681759382, 0.15364077669902912, 0.11067961165048543]
])
ztf_offerr = abs(ztf_offerr - ztf_off)


_data_names = {'z': 'zHD', 'z_err': 'zHD_err', 'logmass': 'HOST_LOGMASS', 'logmass_err': 'HOST_LOGMASS_ERR'}


def fitres_to_data(df: DataFrame, is_hf, names=frozendict(), cosmo=None, extra_keys=()):
    data = {
        name: df[key].to_numpy()
        for name in ('z', 'z_err', 'm', 'x1', 'c', 'logmass', 'logmass_err', *extra_keys)
        for key in [names.get(name, _data_names.get(name, name))]
    }

    data['obs'] = np.stack([data[key] for key in ('m', 'x1', 'c')], -1)
    data['hf_mask'] = np.full_like(data['z'], is_hf, dtype=bool)


    if cosmo is not None:
        _z = torch.tensor(data['z'], requires_grad=True, dtype=torch.get_default_dtype())
        data['distmod_err'] = data['z_err'] * torch.autograd.grad(cosmo.distmod(_z).sum(), _z)[0].numpy(force=True).astype(data['z_err'].dtype)

    x0_to_m0 = -2.5 / np.log(10) / df['x0']
    data['cov'] = np.moveaxis(np.array([
        [(x0_to_m0 * df['x0_err'])**2, x0_to_m0 * df['cov_x0_x1'], x0_to_m0 * df['cov_x0_c']],
        [x0_to_m0 * df['cov_x0_x1'], df['x1_err']**2, df['cov_x1_c']],
        [x0_to_m0 * df['cov_x0_c'], df['cov_x1_c'], df['c_err']**2]
    ]), -1, 0)

    return data
