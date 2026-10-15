"""Exact Gaussian linear readout conditional on the geometric latent state."""
import math
import torch


def chol_spd(a):
    a = (a+a.mT)*.5
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    for jitter in (0., 1e-10, 1e-8, 1e-6):
        factor, info = torch.linalg.cholesky_ex(a+jitter*eye)
        if int(info.max()) == 0:
            return factor
    raise RuntimeError("Cholesky failed after bounded numerical jitter.")


def posterior(phi, y, prior_mean, prior_var, noise_var):
    m = phi.shape[-1]
    precision = torch.diag(prior_var.reciprocal()) + phi.mT@phi/noise_var
    chol = chol_spd(precision)
    cov = torch.cholesky_inverse(chol)
    rhs = prior_mean/prior_var[:, None] + phi.mT@y/noise_var
    mean = torch.cholesky_solve(rhs, chol)
    return mean, cov


def marginal_loglik(phi, y, prior_mean, prior_var, noise_var):
    n, out = y.shape
    cov = (phi*prior_var)@phi.mT + noise_var*torch.eye(n, dtype=phi.dtype, device=phi.device)
    chol = chol_spd(cov)
    delta = y-phi@prior_mean
    quad = (delta*torch.cholesky_solve(delta, chol)).sum()
    return -.5*(quad + out*2*torch.log(torch.diagonal(chol)).sum() + n*out*math.log(2*math.pi))


def conditional_loglik(phi_c, y_c, phi_q, y_q, prior_mean, prior_var, noise_var):
    mean, cov = posterior(phi_c, y_c, prior_mean, prior_var, noise_var)
    qcov = phi_q@cov@phi_q.mT + noise_var*torch.eye(len(phi_q), dtype=phi_q.dtype, device=phi_q.device)
    chol = chol_spd(qcov); delta = y_q-phi_q@mean
    out = y_q.shape[1]
    return -.5*((delta*torch.cholesky_solve(delta, chol)).sum()
                + out*2*torch.log(torch.diagonal(chol)).sum()+delta.numel()*math.log(2*math.pi))


def gaussian_kl(mean, logstd, prior_logstd, mask=None):
    if mask is None:
        mask = torch.ones_like(mean)
    return (.5*(torch.exp(2*(logstd-prior_logstd)) + mean.square()*torch.exp(-2*prior_logstd)
                   - 1 + 2*(prior_logstd-logstd))*mask).sum()

