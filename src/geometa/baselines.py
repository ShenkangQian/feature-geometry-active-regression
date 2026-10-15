"""Trained two-length-scale Matern-5/2 GP, exact Gaussian conditioning."""
import numpy as np
import torch
from torch import nn
from .blr import chol_spd
from .inference import GaussianMixturePrediction, as_tensor


class MultiscaleGP(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_lengths = nn.Parameter(torch.tensor([-2.3, -.6], dtype=torch.float64))
        self.log_amplitude = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.logits = nn.Parameter(torch.zeros(2, dtype=torch.float64))

    def kernel(self, x, y):
        dist = torch.abs(x.reshape(-1, 1)-y.reshape(1, -1))
        r = 5**.5*dist[:, :, None]/self.log_lengths.clamp(-4.6, 1.6).exp()[None, None, :]
        ks = (1+r+r*r/3)*torch.exp(-r)
        return self.log_amplitude.clamp(-5, 5).exp()*(ks*torch.softmax(self.logits, dim=0)).sum(dim=-1)

    def loss(self, x, y, noise_var):
        cov = self.kernel(x, x)+noise_var*torch.eye(len(x), dtype=x.dtype)
        chol = chol_spd(cov)
        return .5*((y*torch.cholesky_solve(y, chol)).sum()+y.shape[1]*2*torch.log(torch.diagonal(chol)).sum())/y.numel()

    def predict(self, x_c, y_c, x_pred, noise_var):
        xc, yc, xp = [as_tensor(v) for v in (x_c, y_c, x_pred)]
        with torch.no_grad():
            chol = chol_spd(self.kernel(xc, xc)+noise_var*torch.eye(len(xc), dtype=xc.dtype))
            cross = self.kernel(xp, xc)
            mean = cross@torch.cholesky_solve(yc, chol)
            cov = self.kernel(xp, xp)-cross@torch.cholesky_solve(cross.T, chol)
            cov = (cov+cov.T)*.5
            joint = torch.kron(cov.contiguous(), torch.eye(yc.shape[1], dtype=yc.dtype))
        return GaussianMixturePrediction(mean.flatten().numpy()[None], joint.numpy()[None], np.ones(1), yc.shape[1], noise_var)



class BasisGP(nn.Module):
    """Semiparametric GP: meta-learned fixed features as explicit basis plus a
    two-scale Matern-5/2 residual process (Rasmussen & Williams, Sec. 2.7).

        f(x) = phi(x; z=0)^T w + g(x),  w ~ N(m, diag(v)),  g ~ GP(0, k)

    phi is the frozen meta-trained feature map at z=0, so this baseline has
    exactly the representation of the `fixed` method plus a nonparametric
    residual. Its kernel parameters, basis prior mean m and variance v are
    trained on the same training tasks as the other baselines. It answers
    whether any gain of geometric adaptation over a GP is explained merely by
    the GP lacking a meta-learned mean/basis. Not a published method.
    """
    def __init__(self, feature_model):
        super().__init__()
        self.features = feature_model
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        d = feature_model.hidden+2
        self.gp = MultiscaleGP()
        self.basis_mean = nn.Parameter(torch.zeros(d, feature_model.output_dim, dtype=torch.float64))
        self.basis_logvar = nn.Parameter(torch.full((d,), 0., dtype=torch.float64))

    def _phi(self, x):
        z0 = torch.zeros(self.features.latent_dim, dtype=torch.float64, device=x.device)
        return self.features(x, z0)

    def kernel(self, x, y):
        return self.gp.kernel(x, y) + (self._phi(x)*self.basis_logvar.clamp(-6, 5).exp())@self._phi(y).T

    def mean(self, x):
        return self._phi(x)@self.basis_mean

    def loss(self, x, y, noise_var):
        cov = self.kernel(x, x)+noise_var*torch.eye(len(x), dtype=x.dtype)
        chol = chol_spd(cov); delta = y-self.mean(x)
        return .5*((delta*torch.cholesky_solve(delta, chol)).sum()+y.shape[1]*2*torch.log(torch.diagonal(chol)).sum())/y.numel()

    def predict(self, x_c, y_c, x_pred, noise_var):
        xc, yc, xp = [as_tensor(v) for v in (x_c, y_c, x_pred)]
        with torch.no_grad():
            chol = chol_spd(self.kernel(xc, xc)+noise_var*torch.eye(len(xc), dtype=xc.dtype))
            cross = self.kernel(xp, xc)
            mean = self.mean(xp)+cross@torch.cholesky_solve(yc-self.mean(xc), chol)
            cov = self.kernel(xp, xp)-cross@torch.cholesky_solve(cross.T, chol)
            cov = (cov+cov.T)*.5
            joint = torch.kron(cov.contiguous(), torch.eye(yc.shape[1], dtype=yc.dtype))
        return GaussianMixturePrediction(mean.flatten().numpy()[None], joint.numpy()[None], np.ones(1), yc.shape[1], noise_var)
