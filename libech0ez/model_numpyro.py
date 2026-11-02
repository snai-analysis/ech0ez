from dataclasses import dataclass, fields
from typing import Literal, get_args, get_origin, ClassVar, Mapping, Any, Optional

from jax import Array
from jax.numpy import array, stack, log, where, expm1, log1p, exp, linspace, sqrt, broadcast_arrays, interp, empty, zeros, reshape
from jax.numpy.linalg import cholesky
from jax.scipy.special import erfc
from numpyro import sample, deterministic, plate, handlers
from numpyro._typing import Message
from numpyro.distributions import Uniform, Normal, Exponential, MultivariateNormal, InverseGamma, Categorical
from numpyro.primitives import Messenger

from .cosmax import distmod_qj


def log1mexp(x):
    return where(x < log(0.5), log1p(-exp(x)), log(-expm1(x)))


def logsubexp(a, b):
    return a + log1mexp(b-a)


def exponnorm_pdf(x, mean, std, rate):
    ls2 = rate * std**2
    return (
        rate/2
        * exp(rate/2 * (2*mean + ls2 - 2*x))
        * erfc((mean + ls2 - x)/(sqrt(2.) * std))
    )


def sample_mvn(name, mean, scale_tril):
    with plate(f'plate_{len(mean)}', len(mean)):
        return deterministic(name, mean + scale_tril @ sample(name+'_', Normal(0, 1)))


class SamplingContext(Messenger):
    def __enter__(self):
        self.ctx = {}
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        self.ctx = None
        return super().__exit__(*args, **kwargs)

    def process_message(self, msg: Message) -> None:
        if msg.get('name', None) in self.ctx and ((msg['type'] == 'sample' and msg['value'] is None) or msg['type'] == 'deterministic'):
            msg['value'] = self.ctx[msg['name']]
            msg['stop'] = True

    def postprocess_message(self, msg: Message) -> None:
        if msg['type'] in ('sample', 'deterministic'):
            self.ctx[msg['name']] = msg['value']


class NumpyroSample:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return sample(self.name, self.fn)


class NumpyroModel:
    _prior: ClassVar[Mapping[str, Any]] = {}

    def __init__(self):
        for key, fn in self._priors.items():
            setattr(self, key, NumpyroSample(key, fn))

    def __getattribute__(self, name):
        ret = super().__getattribute__(name)
        return ret.__get__(self, type(self)) if isinstance(ret, NumpyroSample) else ret

    def forward(self, *args, **kwargs): ...

    def __call__(self, *args, **kwargs):
        with SamplingContext():
            return self.forward(*args, **kwargs)


@dataclass
class Model(NumpyroModel):
    obs: Array
    cov: Array

    z_hf_mean: Array
    z_hf_cov: Array

    calib_idx: Array
    distmod_calib_mean: Array
    distmod_calib_cov: Array

    calib_mask: Array
    hf_mask: Array

    cosmo_type: Literal['fid', 'h', 'hqj'] = 'hqj'
    use_2x: bool = False
    RV_mode: Literal['global', 'dist'] = 'dist'
    use_AV: bool = True
    z_mode: Literal['lin', 'lat'] = 'lat'

    dodust: bool = True
    dostep: bool = False

    sel_zmax: float = 0.06
    sel_zbins: Optional[int] = None
    sel_mbins: int = 8000

    distmod_hf_err: Optional[Array] = None

    logmass: Optional[Array] = None

    photcalib_zs: Optional[Array] = None
    photcalib_dms_mean: Optional[Array] = None
    photcalib_dms_cov: Optional[Array] = None

    @classmethod
    def settings(cls):
        return {
            f.name: get_args(f.type) or (True, False)
            for f in fields(cls) if f.type is bool or get_origin(f.type) is Literal
        }

    _priors: ClassVar = {
        'h': Uniform(0.5, 0.9), 'q0': Uniform(-2, 1), 'j0': Uniform(-1, 3),
        'rate_exp': Uniform(-2, 2), 'sel_m0': Uniform(17, 20),
        'M0': Uniform(-20, -18), 'M0_1': Uniform(-20, -18), 'M0_2': Uniform(-20, -18),
        'dfinv_res': Uniform(0.01, 1),
        'alpha': Uniform(-0.55, 0.25),
        'alpha_1': Uniform(-0.55, 0.25), 'alpha_2': Uniform(-0.55, 0.25),
        'alpha_c': Uniform(-0.03, 0.03), 'beta': Uniform(-4, 4),
        'gamma': Uniform(-0.5, 0.5),
        'mean_x': Uniform(-3, 3), 'sigma_x': Uniform(0.01, 3),
        'xfrac': Uniform(0., 1.), 'xfrac_calib': Uniform(0., 1.), 'xfrac_hf': Uniform(0., 1.),
        # 'mean_x1': Uniform(-3, 3), 'mean_x2': Uniform(-3, 3),
        'sigma_x1': Uniform(0.01, 3), 'sigma_x2': Uniform(0.01, 3),
        'mean_c': Uniform(-0.3, 0.3), 'sigma_c': Uniform(0.001, 0.3),
        'mean_R': Uniform(1.2, 5), 'sigma_R': Uniform(0.01, 2), 'tau': Uniform(0.001, 0.5)
    }

    @property
    def sigma_res(self):
        var_res_ = sample('var_res_', InverseGamma(3e-3, 3e-3))
        return deterministic('sigma_res', var_res_**0.5)

    @property
    def mean_x1(self):
        mean_x1_ = sample('mean_x1_', Uniform(0, 1))
        return deterministic('mean_x1', -3 + 6 * (1 - mean_x1_**0.5))

    @property
    def mean_x2(self):
        return sample('mean_x2', Uniform(self.mean_x1, 3))


    def __post_init__(self):
        super().__init__()

        self.cov = array(self.cov)
        if self.z_mode == 'lin':
            self.cov = self.cov.at[self.hf_mask, 0, 0].set(
                self.cov[self.hf_mask, 0, 0]
                + self.distmod_hf_err**2
            )
        self.scale_tril = cholesky(self.cov)

        self.z_hf_scale_tril = cholesky(self.z_hf_cov)
        self.distmod_calib_scale_tril = cholesky(self.distmod_calib_cov)
        if self.photcalib_dms_cov is not None:
            self.photcalib_dm_scale_tril = cholesky(self.photcalib_dms_cov)

        self.x_int = self.x_int_2x if self.use_2x else self.x_int_1x
        self.R_V = self.R_V_global if self.RV_mode == 'global' else self.R_V_dist

        self.zgrid = linspace(0, self.sel_zmax, (self.sel_zbins or int(round(1000*self.sel_zmax, 0)))+1)
        self.zgrid = (self.zgrid[1:] + self.zgrid[:-1]) / 2

        self.mgrid = linspace(13, 21, self.sel_mbins+1)
        self.mgrid = (self.mgrid[1:] + self.mgrid[:-1]) / 2

        self.imkwargs = dict(extent=(0, 0.06, 13, 21), origin='lower', aspect='auto')

        self.mcov = self.cov.take(array([1, 2, 0]), axis=-2).take(array([1, 2, 0]), axis=-1)

    @property
    def name(self):
        return '-'.join(filter(bool, (
            self.cosmo_type,
            f'{self.z_mode}z',
            f'{self.RV_mode}R',
            ('AV' if self.use_AV else 'E'),
            ('2x' if self.use_2x else '1x'),
            ('step' if self.dostep else ''),
        )))

    global_params: ClassVar = {
        'h', 'q0', 'j0',
        'rate_exp', 'sel_m0',
        'M0', 'M0_1', 'M0_2', 'sigma_res', 'dfinv_res',
        'alpha', 'alpha_1', 'alpha_2', 'alpha_c', 'beta', 'gamma',
        'mean_x', 'sigma_x', 'xfrac', 'mean_x1', 'sigma_x1', 'mean_x2', 'sigma_x2',
        'mean_c', 'sigma_c',
        'R_V', 'mean_R', 'sigma_R', 'tau'
    }

    @property
    def init(self):
        return dict(
            h=0.7, q0=-0.5, j0=1,
            rate_exp=-1., sel_m0=18.8,
            M0=-19.2, M0_1=-19.2, M0_2=-19.2,
            sigma_res=0.1, var_res_=0.1**2, dfinv_res=0.5,
            alpha=-0.14, alpha_1=-0.14, alpha_2=-0.14, alpha_c=0., beta=2.2, gamma=0.,
            mean_x=0., sigma_x=1.,
            xfrac=0.6, xfrac_calib=0.9, xfrac_hf=0.6,
            mean_x1_=0.4, mean_x1=-0.8, sigma_x1=0.9, mean_x2=0.5, sigma_x2=0.45,
            mean_c=0., sigma_c=0.05,
            mean_R=3, sigma_R=0.5, tau=0.2 if self.use_AV else 0.05
        ) | (dict(R_V=3.1) if self.RV_mode == 'global' else dict())

    def sim_init(self, rng_seed=42426, *args, **kwargs):
        with handlers.seed(rng_seed=rng_seed), handlers.trace() as trace:
            handlers.condition(self, data=self.init | kwargs)(*args)
        sim = {key: val['value'] for key, val in trace.items() if not key.startswith('_')}
        sample_keys = {key for key in sim.keys() if trace[key]['type'] == 'sample'}

        return sample_keys, sim

    @property
    def cosmo(self):
        h = self.h if 'h' in self.cosmo_type else 0.73
        q0 = self.q0 if 'q' in self.cosmo_type else -0.5
        j0 = self.j0 if 'j' in self.cosmo_type else 1

        return h, q0, j0

    @property
    def z_hf(self):
        return self.z_hf_mean if self.z_mode == 'lin' else sample_mvn('z_hf', self.z_hf_mean, self.z_hf_scale_tril)

    def distmod(self):
        return deterministic('distmod', empty(len(self.hf_mask)).at[self.calib_mask].set(
            sample_mvn('distmod_calib', self.distmod_calib_mean, self.distmod_calib_scale_tril)[self.calib_idx]
        ).at[self.hf_mask].set(deterministic('distmod_hf', distmod_qj(self.z_hf, *self.cosmo))))

    def x_int_1x(self):
        mean_x = self.mean_x
        sigma_x = self.sigma_x

        with self.plate:
            return sample('x_int', Normal(mean_x, sigma_x))

    @property
    def compid(self):
        xfrac = self.xfrac
        # xfrac = where(self.calib_mask, self.xfrac_calib, self.xfrac_hf)
        with self.plate:
            return sample('compid', Categorical(stack([xfrac, 1-xfrac], -1)))

    def x_int_2x(self):
        mean_x1 = self.mean_x1
        sigma_x1 = self.sigma_x1
        mean_x2 = self.mean_x2
        sigma_x2 = self.sigma_x2

        compid = self.compid

        with self.plate:
            return sample('x_int', Normal(
                where(compid == 0, mean_x1, mean_x2),
                where(compid == 0, sigma_x1, sigma_x2)
            ))

    def c_int(self, x_int):
        mean_c = self.mean_c
        sigma_c = self.sigma_c
        alpha_c = self.alpha_c

        with self.plate:
            c_int_ = sample('c_int_', Normal(0, 1))
            return deterministic('c_int', mean_c + alpha_c*x_int + sigma_c*c_int_)

    def deltaM_(self):
        with self.plate:
            return sample('deltaM_', Normal(0., 1.))

    @property
    def deltaM(self):
        off = where(self.logmass > 10, self.gamma, 0.) if self.dostep else 0.
        sigma_res = self.sigma_res
        deltaM_ = self.deltaM_()

        with self.plate:
            return deterministic('deltaM', off + sigma_res * deltaM_)

    def M_int(self, x_int, c_int):
        return (where(
            self.compid == 0,
            self.M0_1 + self.alpha_1 * x_int,
            self.M0_2 + self.alpha_2 * x_int,
        ) if self.use_2x else (self.M0 + self.alpha * x_int)) + self.beta*c_int + self.deltaM

    @staticmethod
    def R_V_global():
        return sample('R_V', Uniform(1.2, 5))

    def R_V_dist(self):
        mean_R = self.mean_R
        sigma_R = self.sigma_R

        with self.plate:
            R_V_ = sample('R_V_', Uniform(0., 1.))
            distR_V = Normal(mean_R, sigma_R)
            prob_low = distR_V.cdf(1.2)
            return deterministic('R_V', distR_V.icdf(prob_low + (1-prob_low) * R_V_))
            # R_V = deterministic('R_V', TruncatedNormal(mean_R, sigma_R, low=1.2).icdf(R_V_))
            # R_V = sample('R_V', TruncatedNormal(mean_R, sigma_R, low=1.2))

    def E(self, R_V):
        tau = self.tau

        with self.plate:
            dust = tau * sample('dust_', Exponential(1.))
            if self.use_AV:
                A_V = deterministic('A_V', dust)
                return deterministic('E', A_V / R_V)
            else:
                return deterministic('E', dust)


    def photcalib_scatter(self):
        if self.photcalib_zs is None:
            return 0

        return deterministic('photcalib_scatter_', zeros(len(self.scale_tril)).at[self.hf_mask].set(
            interp(self.z_hf_mean, self.photcalib_zs, sample_mvn('photcalib_dms_', self.photcalib_dms_mean, self.photcalib_dm_scale_tril))
        ))


    def forward_latent(self):
        self.plate = plate('_SN', len(self.scale_tril))

        x_int = self.x_int()
        c_int = self.c_int(x_int)
        M_int = self.M_int(x_int, c_int)

        if self.dodust:
            R_V = self.R_V()
            E = self.E(R_V)
        else:
            R_V = 1.
            E = 0.

        x_ext = x_int
        c_ext = c_int + E
        M_ext = M_int + E * (R_V+1)

        distmod = self.distmod()

        m = M_ext + distmod + self.photcalib_scatter()

        with self.plate:
            return sample('obs', MultivariateNormal(
                stack(broadcast_arrays(m, x_ext, c_ext), -1),
                scale_tril=self.scale_tril
            ), obs=self.obs)

    def prior_int(self):
        if self.use_2x:
            compid = self.compid
            mean_x = where(compid==0, self.mean_x1, self.mean_x2)
            sigma_x = where(compid==0, self.sigma_x1, self.sigma_x2)
            alpha = where(compid==0, self.alpha_1, self.alpha_2)
            M0 = where(compid==0, self.M0_1, self.M0_2)
        else:
            mean_x = self.mean_x
            sigma_x = self.sigma_x
            alpha = self.alpha
            M0 = self.M0
        mean_c = self.mean_c + self.alpha_c*mean_x
        mean_M = M0 + alpha*mean_x + self.beta*mean_c + (where(self.logmass > 10, self.gamma, 0.) if self.dostep else 0.)

        a, b, d = sigma_x, sigma_x*self.alpha_c, self.sigma_c
        e, f, g = sigma_x*(self.alpha_c*self.beta + alpha), self.sigma_c*self.beta, self.sigma_res
        t00 = sqrt(a**2 + self.mcov[..., 0, 0])
        t10 = (a*b + self.mcov[..., 1, 0]) / t00
        t20 = (a*e + self.mcov[..., 2, 0]) / t00
        t11 = sqrt(b**2 + d**2 + self.mcov[..., 1, 1] - t10**2)
        t21 = (b*e + d*f + self.mcov[..., 2, 1] - t20*t10) / t11
        t22 = sqrt(e**2 + f**2 + g**2 + self.mcov[..., 2, 2] - t20**2 - t21**2)
        _scale_tril = stack(broadcast_arrays(t00, 0, 0, t10, t11, 0, t20, t21, t22), -1)
        _scale_tril = reshape(_scale_tril, (*_scale_tril.shape[:-1], 3, 3))

        return stack(broadcast_arrays(mean_x, mean_c, mean_M), -1), _scale_tril

    def forward_marginal(self):
        self.plate = plate('_SN', len(self.scale_tril))

        if self.dodust:
            R_V = self.R_V()
            E = self.E(R_V)
        else:
            R_V = 1.
            E = 0.

        distmod = self.distmod()
        photcalib_scatter = self.photcalib_scatter()

        _mean, _scale_tril = self.prior_int()

        with self.plate:
            return sample('obs', MultivariateNormal(_mean, scale_tril=_scale_tril), obs=stack(broadcast_arrays(
                self.obs[:, 1],
                self.obs[:, 2] - E,
                self.obs[:, 0] - E * (R_V + 1) - distmod - photcalib_scatter
            ), -1))

    def forward(self):
        return (self.forward_marginal if self.use_2x else self.forward_latent)()
