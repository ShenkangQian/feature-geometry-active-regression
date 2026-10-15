import math
import torch
from torch import nn


class GeometricFeatures(nn.Module):
    """Shared tanh geometry + task-local nonlinear centre/width adaptation.

    A single hidden layer is intentional. This is neither a Neural Process
    reproduction nor the LUMA architecture. All continuous parameters here are
    actually optimized in cross-task context/query episodes.
    """
    def __init__(self, hidden=12, output_dim=1, position_dim=2, width_dim=2):
        super().__init__()
        self.hidden, self.output_dim = hidden, output_dim
        self.position_dim, self.width_dim = position_dim, width_dim
        self.latent_dim = position_dim+width_dim
        self.base_centres = nn.Parameter(torch.linspace(-.85, .85, hidden))
        self.base_logwidth = nn.Parameter(torch.full((hidden,), math.log(.17)))
        self.position_loadings = nn.Parameter(torch.randn(hidden, position_dim))
        self.width_loadings = nn.Parameter(torch.randn(hidden, width_dim))
        self.readout_mean = nn.Parameter(torch.zeros(hidden+2, output_dim))
        self.readout_logvar = nn.Parameter(torch.full((hidden+2,), math.log(3.)))
        self.latent_logstd = nn.Parameter(torch.full((self.latent_dim,), -.35))
        self.double()

    @staticmethod
    def normalized_loadings(value):
        return value / (value.square().mean(dim=0, keepdim=True)+1e-6).sqrt()

    def forward(self, x, z):
        x = x.reshape(-1)
        if z.ndim != 1:
            raise ValueError("Use one latent state per feature matrix.")
        c = self.base_centres + .25*(self.normalized_loadings(self.position_loadings)@z[:self.position_dim])
        logwidth = self.base_logwidth + .45*(self.normalized_loadings(self.width_loadings)@z[self.position_dim:])
        widths = torch.exp(logwidth.clamp(-4.6, .4))
        atoms = torch.tanh((x[:, None]-c[None, :])/widths[None, :])
        return torch.cat((torch.ones_like(x[:, None]), x[:, None], atoms), dim=1)/math.sqrt(self.hidden+2)

    @property
    def prior_var(self):
        return self.readout_logvar.clamp(-6, 5).exp()

    @property
    def z_logstd(self):
        return self.latent_logstd.clamp(-2.5, 1.)

    def mask(self, mode):
        mask = torch.ones(self.latent_dim, dtype=torch.float64, device=self.base_centres.device)
        if mode == "position":
            mask[self.position_dim:] = 0
        elif mode == "width":
            mask[:self.position_dim] = 0
        elif mode == "fixed":
            mask[:] = 0
        elif mode not in ("full", "point", "no_readout_variance"):
            raise ValueError(mode)
        return mask

