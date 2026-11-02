from jax.numpy import log10


def hubble_distance_Mpc(h):
    return 2997.92458/h


def inv_efunc_qj(z, h, q0, j0):
    return 1 - z * (1+q0) + z**2 * (1 + 2*q0 + 1.5*q0**2 - 0.5*j0)


def comoving_distance_qj(z, h, q0, j0):
    return hubble_distance_Mpc(h) * z * (1 - z * ((1+q0)/2) + z**2 * ((2+4*q0+3*q0**2-j0)/6))


def differential_comoving_volume_qj(z, h, q0, j0):
    return comoving_distance_qj(z, h, q0, j0)**2 * hubble_distance_Mpc(h) * inv_efunc_qj(z, h, q0, j0)


def distmod_qj(z, h, q0, j0):
    return 5 * (log10(299792458) + log10(z/h) + log10(1 + (1-q0)/2 * z - (1-q0-3*q0**2+j0)/6 * z**2))
