"""Integrated latent squared-error reduction. No target labels are arguments."""
import numpy as np
from scipy.special import logsumexp


def moment_ivr(prediction, n_candidates, n_integration):
    """Gaussian MOMENT-MATCHED approximation for a non-Gaussian mixture.

    Exact for Gaussian baselines; not exact mixture Bayesian experimental design.
    Integrates all outputs with equal, normalized-domain quadrature weights.
    """
    _, cov, _, _ = prediction.moments()
    o = prediction.output_dim
    integration = np.arange(n_candidates*o, (n_candidates+n_integration)*o)
    scores = []
    for j in range(n_candidates):
        idx = np.arange(j*o, (j+1)*o)
        cross = cov[np.ix_(integration, idx)]
        obs = cov[np.ix_(idx, idx)] + prediction.noise_var*np.eye(o)
        scores.append(np.sum(cross*np.linalg.solve(obs, cross.T).T)/(n_integration*o))
    return np.maximum(scores, 0.)


def fantasy_ivr(prediction, n_candidates, n_integration, fantasies=16, seed=0):
    """Finite-mixture Bayesian conditioning + Monte Carlo fantasy expectation.

    z-particles form a discrete approximation to q(z). For each prospective
    vector observation, component weights and Gaussian readouts are conditioned
    analytically. This does not rerun VI for each fantasy. It is exact
    conditioning OF THE FINITE MIXTURE, with a finite Monte Carlo expectation.
    """
    rng = np.random.default_rng(seed); o = prediction.output_dim
    ids = np.arange(n_candidates*o, (n_candidates+n_integration)*o)
    mm, cov, _, _ = prediction.moments()
    base_risk = float(np.diag(cov)[ids].mean())
    s = len(prediction.weights)
    component_ids = rng.choice(s, fantasies, p=prediction.weights)
    normal = rng.normal(size=(fantasies, o))
    scores = []
    for j in range(n_candidates):
        qids = np.arange(j*o, (j+1)*o)
        obs_cov = prediction.covariances[:, qids[:, None], qids] + prediction.noise_var*np.eye(o)[None]
        chol = np.linalg.cholesky(obs_cov)
        # Common random numbers reduce ranking noise across candidate locations.
        simulated = prediction.means[component_ids][:, qids] + np.einsum("fij,fj->fi", chol[component_ids], normal)
        cross = prediction.covariances[:, ids[:, None], qids]
        gains = np.linalg.solve(obs_cov, cross.transpose(0, 2, 1)).transpose(0, 2, 1)
        remaining_diag = np.diagonal(prediction.covariances, axis1=1, axis2=2)[:, ids] - np.sum(gains*cross, axis=-1)
        risks = []
        for observation in simulated:
            delta = observation[None, :]-prediction.means[:, qids]
            solved = np.linalg.solve(obs_cov, delta[:, :, None])[:, :, 0]
            logp = -.5*(np.sum(delta*solved, axis=1)+2*np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)+o*np.log(2*np.pi))
            lw = np.log(prediction.weights)+logp; weights = np.exp(lw-logsumexp(lw))
            updated = prediction.means[:, ids]+np.einsum("sij,sj->si", gains, delta)
            updated_mean = weights@updated
            risk = np.sum(weights[:, None]*(remaining_diag+(updated-updated_mean)**2))/len(ids)
            risks.append(risk)
        scores.append(base_risk-np.mean(risks))
    # Finite MC can yield negative estimated gains; retain this fact in logs.
    return np.array(scores)


def select_query(prediction, n_candidates, n_integration, available, seed, exploration=.15,
                 strategy="moment", fantasies=16, random_only=False):
    rng = np.random.default_rng(seed)
    available = np.asarray(available, dtype=int)
    if len(available) == 0:
        raise ValueError("No unqueried candidates remain.")
    if random_only or rng.random() < exploration:
        return int(rng.choice(available)), {"exploration": True, "score": None, "strategy": strategy}
    if strategy == "moment":
        scores = moment_ivr(prediction, n_candidates, n_integration)
    elif strategy == "fantasy":
        scores = fantasy_ivr(prediction, n_candidates, n_integration, fantasies, seed)
    else:
        raise ValueError(strategy)
    index = int(available[np.argmax(scores[available])])
    return index, {"exploration": False, "score": float(scores[index]), "strategy": strategy,
                   "negative_mc_scores": int(np.sum(scores[available] < 0))}

