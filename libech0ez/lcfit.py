from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Container, Mapping, Any, TypedDict, cast, Sequence, Optional
from warnings import catch_warnings, filterwarnings

import numpy as np
import sncosmo
from astropy.table import Table
from pandas import DataFrame
from scipy.stats import chi2
from sncosmo.fitting import DataQualityError
from sncosmo.utils import Result
from tqdm.auto import tqdm

lin_to_mag = -2.5/np.log(10)


def fitres_to_dict(fitres: Result):
    return dict(zip(fitres.param_names, fitres.parameters)) | {
        f'{p}_err': v for p, v in fitres.errors.items()
    } | {
        f'cov_{p1}_{p2}': c.item() for (p1, p2), c in zip(
            combinations_with_replacement(fitres.vparam_names, 2),
            fitres.covariance[np.triu_indices(len(fitres.vparam_names))])
    } | {key: getattr(fitres, key) for key in ('success', 'chisq', 'ndof', 'ncall')}


class HeadRowT(TypedDict, total=True):
    t0: float
    zHEL: float
    zHD: float
    MWEBV: float


class SNANAPhotT(TypedDict, total=True):
    MJD: Sequence[float]
    FLUXCAL: Sequence[float]
    FLUXCALERR: Sequence[float]
    FLT: Sequence[str]


def phot_from_snana(phot: SNANAPhotT, get_band=lambda b, meta: b) -> Table:
    ret = Table({
        'mjd': phot['MJD'], 'flux': phot['FLUXCAL'], 'fluxerr': phot['FLUXCALERR'],
        'band': [get_band(b, phot.meta if isinstance(phot, Table) else None)
                 for b in dict(phot).get('FLT', dict(phot).get('FILTER', None))]})
    ret['zp'] = 27.5
    ret['magsys'] = [b.split('::')[1] for b in ret['band']]
    return ret


@dataclass
class LCFitter:
    modelcov: bool = True
    exclude: Container[str] = frozenset()

    saltver: str = 'T21'
    saltzp: Optional[float] = None
    saltzp_band: str = 'bessellb'
    saltzp_magsys: str = 'Vega'

    mwdust: type[sncosmo.PropagationEffect] = sncosmo.F99Dust

    def model(self, t0, z, mwebv, **kwargs):
        # model = sncosmo.Model(
        #     source=self.src,
        #     effects=[sncosmo.F99Dust(), sncosmo.models.C11()],
        #     effect_names=['mw', 'C11'], effect_frames=['obs', 'rest']
        # )
        model = sncosmo.Model(
            source=self.src,
            effects=[self.mwdust()],
            effect_names=['mw'], effect_frames=['obs']
        )
        model.set(t0=t0, z=z, mwebv=mwebv, **kwargs)
        return model

    @property
    def src(self):
        return sncosmo.get_source('salt2', self.saltver)

    @property
    def salt_zp(self):
        return self.saltzp if self.saltzp is not None else self.src.bandmag(self.saltzp_band, self.saltzp_magsys, 0)

    def __call__(self, head: DataFrame, phots: Mapping[Any, Table], fix_t0=False, **kwargs) -> tuple[DataFrame, Mapping[Any, Result]]:
        results = {}
        for snid, phot in tqdm(phots.items()):
            row = cast(HeadRowT, head.loc[snid])

            data = phot[[b not in self.exclude for b in phot['band']]]

            if not data:
                continue

            model = self.model(t0=row['t0'], z=row['zHEL'], mwebv=row['MWEBV'])

            with catch_warnings():
                filterwarnings('ignore', 'Dropping following bands')

                try:
                    result, fitted_model = sncosmo.fit_lc(
                        data, model, ([] if fix_t0 else ['t0']) + ['x0', 'x1', 'c'],
                        modelcov=self.modelcov,
                        bounds={'x1': (-5, 5), 'c': (-1, 1)},
                        **kwargs
                    )
                except DataQualityError:
                    continue
            results[snid] = result


        df_res = DataFrame({snid: fitres_to_dict(res) for snid, res in results.items()}).T.convert_dtypes()
        if 'zHD' in head:
            df_res = df_res.assign(zHD=head['zHD'])

        df_res['fitprob'] = chi2(df_res['ndof']).sf(df_res['chisq'])

        df_res['m'] = self.salt_zp + (-2.5 * np.log10(df_res['x0']))
        df_res['m_err'] = abs(lin_to_mag * df_res['x0_err'] / df_res['x0'])

        return df_res, results


def plot_fit_comparison(df_ref, df_res, salt_zp=None, bad_idx=None):
    from matplotlib import pyplot as plt

    idx = df_res[df_res['success']].index
    # bad_idx = df_res[abs(df_res['t0'] - df_ref['t0']) > 1].index

    fig, axs = plt.subplots(1, 4, figsize=(13, 4))

    for ax, (key, kwargs) in zip(axs, (
        ('t0', dict(lbl='$t_0$')),
        ('m', dict(lbl='$m$', xlim=(13, 21))),
        ('x1', dict(lbl='$x_1$', xlim=(-4, 4))),
        ('c', dict(lbl='$c$', xlim=(-0.4, 1))),
    )):
        if 'xlim' in kwargs:
            ax.plot(*2*(kwargs['xlim'],), 'k--')

        ax.errorbar(
            x=df_ref.loc[idx, key], xerr=np.clip(df_ref.loc[idx, f'{key}_err'], 0, float('inf')),
            y=df_res.loc[idx, key], yerr=np.clip(df_res.loc[idx, f'{key}_err'], 0, float('inf')),
            ls='none', capthick=0, lw=1, alpha=0.5, color='k'
        )
        ax.plot(df_ref.loc[idx, key], df_res.loc[idx, key], '.', ms=5, mew=0, alpha=0.5)
        if bad_idx is not None:
            ax.plot(df_ref.loc[bad_idx, key], df_res.loc[bad_idx, key], 'rx', ms=5, mew=0.5)

        ax.set(
            title=kwargs['lbl'],
            xlabel=f'ref. {kwargs.pop("lbl")}',
            # ylabel=f'their {kwargs.pop("lbl")}',
            ylim=kwargs.get('xlim', None),
            **{**dict(aspect='equal'), **kwargs})

    # axs[0].legend(loc='upper left')

    if salt_zp is not None:
        axs[1].text(0.02, 0.98, fr'$\mathrm{{ZP}}={salt_zp:.2f}$', ha='left', va='top', transform=axs[1].transAxes)

    return fig, axs
