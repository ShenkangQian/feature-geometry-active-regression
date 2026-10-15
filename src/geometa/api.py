"""Minimal prediction/query interface for a saved research checkpoint."""
import numpy as np
from .training import load_family
from .inference import geometric_predict
from .acquisition import select_query


class ActiveRegressor:
    def __init__(self, checkpoint_path, mode="full", seed=0):
        self.model, _, self.mean, self.scale, self.checkpoint = load_family(checkpoint_path)
        self.mode, self.seed = mode, int(seed)
        self.config = self.checkpoint["config"]
        self.x = self.y = None

    def observe(self, x, y):
        """Replace context with ORIGINAL-unit measurements; no hidden labels."""
        x, y = np.asarray(x).reshape(-1), np.asarray(y)
        if y.ndim == 1: y = y[:, None]
        if len(x) != len(y) or y.shape[1] != self.model.output_dim:
            raise ValueError("Context dimensions do not match checkpoint.")
        if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Finite nonempty context required.")
        self.x, self.y = x, (y-self.mean)/self.scale
        return self

    def posterior(self, x):
        if self.x is None: raise RuntimeError("Call observe first.")
        return geometric_predict(self.model, self.x, self.y, np.asarray(x), self.config["noise_std"]**2,
                                 self.config["inference"], self.seed, self.mode)

    def predict(self, x):
        posterior = self.posterior(x)
        mean, cov, within, between = posterior.moments()
        scales = np.tile(self.scale, len(np.asarray(x).reshape(-1)))
        return {"mean": mean.reshape(-1, self.model.output_dim)*self.scale+self.mean,
                "covariance": cov*scales[:, None]*scales[None, :],
                "within_covariance": within*scales[:, None]*scales[None, :],
                "between_covariance": between*scales[:, None]*scales[None, :]}

    def suggest(self, candidate_x, integration_x, available_indices=None, seed=None):
        candidates = np.asarray(candidate_x).reshape(-1); quadrature = np.asarray(integration_x).reshape(-1)
        if available_indices is None:
            available_indices = np.flatnonzero(~np.isclose(candidates[:, None], self.x[None, :]).any(axis=1))
        posterior = self.posterior(np.concatenate((candidates, quadrature)))
        settings = self.config["evaluation"]
        index, diagnostics = select_query(posterior, len(candidates), len(quadrature), available_indices,
                        self.seed if seed is None else seed, settings["exploration"],
                        settings["acquisition"], settings["fantasies"])
        return {"index": index, "x": float(candidates[index]), **diagnostics}

