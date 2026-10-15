"""Reparameterized latent VI and exact conditional readout uncertainty."""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.special import logsumexp
from . import blr


def as_tensor(value, device="cpu"):
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def adapt_q(model, x, y, noise_var, steps=15, lr=.25, samples=6, seed=0, mode="full", differentiable=False):
    mask = model.mask(mode)
    prior_ls = model.z_logstd
    if mode == "fixed":
        return torch.zeros_like(prior_ls), torch.full_like(prior_ls, -30), mask
    mu = torch.zeros_like(prior_ls, requires_grad=True)
    ls = prior_ls.clone() if differentiable else prior_ls.detach().clone().requires_grad_(True)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    eps = torch.randn(samples, model.latent_dim, generator=generator, dtype=x.dtype, device=x.device)
    point = mode == "point"
    for _ in range(steps):
        zs = (mu[None, :]+eps*ls.exp()[None, :])*mask if not point else (mu*mask)[None, :]
        ll = torch.stack([blr.marginal_loglik(model(x, z), y, model.readout_mean, model.prior_var, noise_var) for z in zs]).mean()
        penalty = .5*(mu.square()*torch.exp(-2*prior_ls)*mask).sum() if point else blr.gaussian_kl(mu, ls, prior_ls, mask)
        # Divide the WHOLE objective, preserving the relative likelihood/KL
        # weight; this changes optimization step scale, not the VI target.
        objective = (-ll+penalty)/max(y.numel(), 1)
        variables = (mu,) if point else (mu, ls)
        grads = torch.autograd.grad(objective, variables, create_graph=differentiable)
        mu = (mu-lr*grads[0].clamp(-5, 5)).clamp(-4, 4)
        if not point:
            ls = (ls-lr*grads[1].clamp(-5, 5)).clamp(-4.5, 1.)
        if not differentiable:
            mu = mu.detach().requires_grad_(True)
            ls = ls.detach().requires_grad_(True)
    if point:
        ls = torch.full_like(mu, -30)
    return mu, ls, mask


@dataclass
class GaussianMixturePrediction:
    # Flattening convention: (point, output), so simultaneous vector-valued
    # observations at one location are adjacent entries.
    means: np.ndarray           # particles x (n*outputs)
    covariances: np.ndarray     # particles x (n*outputs) x (n*outputs)
    weights: np.ndarray
    output_dim: int
    noise_var: float
    # Diagnostic only: the adapted latent mean/log-std that produced this
    # prediction (None for GP and linearized predictions). Never used by
    # acquisition or scoring; recorded so span residuals can be computed.
    latent_mean: np.ndarray = None
    latent_logstd: np.ndarray = None

    def moments(self):
        mean = self.weights@self.means
        centred = self.means-mean
        within = np.einsum("s,sij->ij", self.weights, self.covariances)
        between = np.einsum("s,si,sj->ij", self.weights, centred, centred)
        covariance = (within+between)
        return mean, (covariance+covariance.T)*.5, within, between

    def sample_functions(self, count, seed=0, include_observation_noise=False):
        """Coherent draws over every location and output, not pointwise draws."""
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.weights), size=count, p=self.weights)
        draws = np.empty((count, self.means.shape[1]))
        for component in np.unique(indices):
            selected = np.flatnonzero(indices == component)
            covariance = self.covariances[component].copy()
            if include_observation_noise:
                covariance += self.noise_var*np.eye(len(covariance))
            vals, vecs = np.linalg.eigh((covariance+covariance.T)*.5)
            if vals.min() < -1e-7:
                raise ValueError("Predictive covariance is not positive semidefinite.")
            factor = vecs*np.sqrt(np.maximum(vals, 0))[None, :]
            draws[selected] = self.means[component] + rng.normal(size=(len(selected), len(vals)))@factor.T
        return draws.reshape(count, -1, self.output_dim)

    def interval(self, inflation=1., draws=1024, seed=1729):
        """Empirical 95% mixture-quantile interval with component std scaled by `inflation`."""
        variances = np.maximum(np.diagonal(self.covariances, axis1=1, axis2=2), 1e-12)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.weights), size=draws, p=self.weights)
        sample = self.means[indices] + rng.normal(size=(draws, self.means.shape[1]))*np.sqrt(variances[indices])*inflation
        return np.quantile(sample, [.025, .975], axis=0)

    def marginal_scores(self, truth, inflation=None):
        truth = np.asarray(truth).reshape(-1)
        variances = np.maximum(np.diagonal(self.covariances, axis1=1, axis2=2), 1e-12)
        mean, cov, within, between = self.moments()
        logpdf = -.5*(np.log(2*np.pi*variances)+(truth[None, :]-self.means)**2/variances)
        latent_nll = -np.mean(logsumexp(np.log(self.weights[:, None])+logpdf, axis=0))
        # Intervals are empirical Gaussian-mixture quantiles, not fitted normal
        # intervals. Deterministic seeded draws make scoring reproducible.
        lower, upper = self.interval()
        scores = {"mse": float(np.mean((mean-truth)**2)), "rmse": float(np.sqrt(np.mean((mean-truth)**2))),
                "latent_nll": float(latent_nll), "coverage95": float(np.mean((truth>=lower)&(truth<=upper))),
                "width95": float(np.mean(upper-lower)), "within_variance": float(np.diag(within).mean()),
                "between_variance": float(np.diag(between).mean())}
        if inflation is not None:
            # Post-hoc variance inflation chosen on VALIDATION tasks only; a
            # reported correction, not evidence that the posterior is calibrated.
            lo, hi = self.interval(inflation)
            scores.update({"calibration_factor": float(inflation),
                           "coverage95_calibrated": float(np.mean((truth>=lo)&(truth<=hi))),
                           "width95_calibrated": float(np.mean(hi-lo))})
        return scores


def geometric_predict(model, x_c, y_c, x_pred, noise_var, config, seed=0, mode="full", fixed_z=None):
    device = model.base_centres.device
    xc, yc, xp = [as_tensor(v, device) for v in (x_c, y_c, x_pred)]
    if fixed_z is not None:
        # Oracle-geometry reference: the latent state is supplied externally
        # (from the truth-fitted z*), only the readout is conditioned on data.
        mu = as_tensor(fixed_z, device); ls = torch.full_like(mu, -30); mask = torch.ones_like(mu)
        mode = "point"
    else:
        mu, ls, mask = adapt_q(model, xc, yc, noise_var, steps=config["steps"], lr=config["lr"],
                              samples=config["samples"], seed=seed, mode=mode)
    generator = torch.Generator(device=device).manual_seed(seed+10007)
    count = 1 if mode in ("fixed", "point") else config["particles"]
    eps = torch.randn(count, model.latent_dim, generator=generator, dtype=xp.dtype, device=device)
    zs = (mu[None, :]+eps*ls.exp()[None, :])*mask
    means, covs = [], []
    with torch.no_grad():
        for z in zs:
            pc, pp = model(xc, z), model(xp, z)
            mw, sw = blr.posterior(pc, yc, model.readout_mean, model.prior_var, noise_var)
            mean = pp@mw
            base = pp@sw@pp.T
            cov = torch.kron(base.contiguous(), torch.eye(model.output_dim, dtype=base.dtype, device=device))
            if mode == "no_readout_variance":
                cov = torch.zeros_like(cov)
            means.append(mean.flatten().cpu().numpy()); covs.append(cov.cpu().numpy())
    return GaussianMixturePrediction(np.array(means), np.array(covs), np.full(count, 1/count), model.output_dim, noise_var,
                                     latent_mean=(mu*mask).detach().cpu().numpy(), latent_logstd=ls.detach().cpu().numpy())


def linearized_predict(model, x_c, y_c, x_pred, noise_var, expansion="conditional_mean"):
    """Our joint first-order Gaussian ablation, NOT a reproduction of LUMA.

    The latent function f(x; z, w) = phi(x; z)^T w is linearized in z at z=0.
    The z-block of the design matrix is the Jacobian of phi(x; z)^T w_lin,
    where w_lin is the readout at which the expansion is taken:

    expansion="conditional_mean" (default): w_lin = m_C(0), the exact readout
    posterior mean given the context under fixed features. This is one
    iteration of "linearize around the current solution", the analogue of a
    Laplace/LUMA-style update.

    expansion="prior_mean": w_lin = m_0, the meta-learned readout prior mean.
    In the delivered checkpoints m_0 has norm 0.11 to 0.21 across seeds, so
    the z-block nearly vanishes and this variant collapses onto the fixed
    method (recorded 2026-09-24; kept under the label `linearized_prior`).

    The resulting Gaussian prior contains independent w blocks and a SHARED
    latent z block, so real/imaginary dependence induced by z is retained.
    """
    device = model.base_centres.device
    xc, yc, xp = [as_tensor(v, device) for v in (x_c, y_c, x_pred)]
    z0 = torch.zeros(model.latent_dim, dtype=torch.float64, device=device, requires_grad=True)
    with torch.no_grad():
        if expansion == "conditional_mean":
            w_lin, _ = blr.posterior(model(xc, z0), yc, model.readout_mean, model.prior_var, noise_var)
        elif expansion == "prior_mean":
            w_lin = model.readout_mean.detach().clone()
        else:
            raise ValueError(expansion)
    def design(x):
        phi = model(x, z0)
        jac = torch.autograd.functional.jacobian(lambda z: (model(x, z)@w_lin).flatten(), z0)
        return torch.cat((torch.kron(phi.contiguous(), torch.eye(model.output_dim, dtype=x.dtype, device=device)), jac), dim=1)
    dc, dp = design(xc).detach(), design(xp).detach()
    with torch.no_grad():
        pm = torch.cat((model.readout_mean.flatten(), torch.zeros_like(z0)))[:, None]
        pv = torch.cat((model.prior_var.repeat_interleave(model.output_dim), torch.exp(2*model.z_logstd)))
        mean, cov = blr.posterior(dc, yc.reshape(-1, 1), pm, pv, noise_var)
        mp, cp = (dp@mean).flatten(), dp@cov@dp.T
        z_lin = mean.flatten()[-model.latent_dim:]
    return GaussianMixturePrediction(mp.cpu().numpy()[None, :], cp.cpu().numpy()[None, :, :], np.ones(1), model.output_dim, noise_var,
                                     latent_mean=z_lin.cpu().numpy())


def span_residual(model, x_eval, truth, z, ridge=1e-10):
    """Proposition 2 quantity: mean squared residual of the true function
    outside the span of the feature matrix Phi(x_eval; z), uniform weights.

    `truth` has shape (n_eval, output_dim) in normalized units and is the
    noiseless latent function held by the evaluator. This is a diagnostic of
    the representation only; no readout prior or context data enters it.
    """
    device = model.base_centres.device
    x = as_tensor(x_eval, device); f = as_tensor(truth, device).reshape(len(x), -1)
    with torch.no_grad():
        phi = model(x, as_tensor(z, device))
        gram = phi.T@phi + ridge*torch.eye(phi.shape[1], dtype=phi.dtype, device=device)
        coef = torch.linalg.solve(gram, phi.T@f)
        resid = f - phi@coef
    return float(resid.square().mean())


def oracle_geometry(model, x_eval, truth, restarts=6, steps=200, lr=.1, seed=0):
    """Fit z* by minimizing the span residual against the TRUE function.

    This is an oracle reference, not a method: it reads held-out truth to
    locate the best achievable feature geometry, giving (i) the Prop. 2 floor
    attainable by any z and (ii) the `oracle_z` upper-reference method whose
    readout is still conditioned on queried data only. Multi-start Adam over
    the bounded latent box [-4, 4]; the best restart is returned.
    """
    device = model.base_centres.device
    x = as_tensor(x_eval, device); f = as_tensor(truth, device).reshape(len(x), -1)
    generator = torch.Generator(device=device).manual_seed(seed)
    starts = [torch.zeros(model.latent_dim, dtype=torch.float64, device=device)]
    starts += list(torch.randn(restarts-1, model.latent_dim, generator=generator, dtype=torch.float64, device=device)*1.5)
    best_z, best_val = None, float("inf")
    eye = torch.eye(model.hidden+2, dtype=torch.float64, device=device)
    for start in starts:
        z = start.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            phi = model(x, z)
            coef = torch.linalg.solve(phi.T@phi + 1e-8*eye, phi.T@f)
            loss = (f-phi@coef).square().mean()
            loss.backward(); opt.step()
            with torch.no_grad():
                z.clamp_(-4, 4)
        with torch.no_grad():
            phi = model(x, z); coef = torch.linalg.solve(phi.T@phi + 1e-8*eye, phi.T@f)
            val = float((f-phi@coef).square().mean())
        if val < best_val:
            best_val, best_z = val, z.detach().clone()
    return best_z.cpu().numpy(), best_val
