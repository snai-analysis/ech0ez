from typing import TypedDict, TypeAlias

import numpy as np
from astropy import units as u
from astropy.constants import c as speed_of_light
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from pandas import DataFrame
from scipy.interpolate import RegularGridInterpolator


class _RedshiftsT(TypedDict, total=False):
    ra: np.ndarray
    dec: np.ndarray
    zHEL: np.ndarray
    zCMB: np.ndarray
    zPEC: np.ndarray
    zHD: np.ndarray


RedshiftsT: TypeAlias = _RedshiftsT | DataFrame


def correct_cmb(df: RedshiftsT):
    coords = SkyCoord(ra=df['ra_host'], dec=df['dec_host'], unit='deg')
    vSUN = 369.82 / 2.99792458e5 * np.cos(
        SkyCoord(l=264.021, b=48.253, unit='deg', frame='galactic').separation(coords)).to('').value
    zSUN = ((1 - vSUN) / (1 + vSUN)) ** 0.5 - 1

    return df.assign(
        zCMB=(1 + df['zHEL']) / (1 + zSUN) - 1,
        zSUN=zSUN
    )


def correct_vpec(df: RedshiftsT, mapname: str):
    coords = SkyCoord(ra=df['ra_host'], dec=df['dec_host'], unit='deg')

    twompp = np.load(mapname)
    cosmo = FlatLambdaCDM(Om0=0.3, H0=100)
    z_p = 0

    z_ps = []
    for _ in range(50):
        z_hd = (1 + df['zCMB']) / (1 + z_p) - 1

        coords_gcen = SkyCoord(coords, distance=cosmo.comoving_distance(z_hd)).transform_to('galactocentric')
        xyz = np.moveaxis(coords_gcen.cartesian.xyz.to('Mpc').value, 0, -1)

        v_x, v_y, v_z = RegularGridInterpolator(3 * [np.linspace(-200, 200, 257)], np.moveaxis(twompp, 0, -1),
                                                bounds_error=False, fill_value=float('nan'))(xyz).T * u.km / u.s
        z_p = (SkyCoord(coords_gcen, v_x=v_x, v_y=v_y, v_z=v_z).radial_velocity / speed_of_light).to('').value
        z_ps.append(z_p)
    z_ps = np.stack(z_ps)

    return df.assign(
        zPEC=z_ps[-1],
        zHD=(1 + df['zCMB']) / (1 + z_ps[-1]) - 1
    ), z_ps


def vpec_cov(ra, dec, dc, h=0.7, f=0.478, sigma8=0.8, omega_c=0.1109, omega_b=0.02258, n_s=0.963):
    from flip.covariance import CovMatrix
    from flip.power_spectra import compute_power_spectra

    ps_k, ps_P_mm, ps_P_mt, ps_P_tt, ps_fid = compute_power_spectra(
        'pyccl_engine', {
            'h': h,
            'Omega_c': omega_c / h**2,
            'Omega_b': omega_b / h**2,
            'sigma8': sigma8,
            'n_s': n_s,
            'transfer_function': 'boltzmann_camb',
        },
        normalization_power_spectrum='growth_amplitude',
        power_spectrum_model='linear',
        power_spectrum_non_linear_model=None,
        redshift=0,
        minimal_wavenumber=1e-4,
        maximal_wavenumber=0.2,
        number_points=1000, logspace=True,
        save_path=None,
    )

    return (f*sigma8)**2 * np.array(CovMatrix.init_from_flip(
        model_name='carreres23', model_kind='velocity',
        power_spectrum_dict={'vv': [[ps_k, ps_P_tt]]},
        coordinates_velocity=(ra, dec, dc*h),
        size_batch=1000, number_worker=1
    ).covariance_dict['vv'][0])
