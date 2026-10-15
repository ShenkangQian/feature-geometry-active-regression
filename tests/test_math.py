import numpy as np
import torch
from geometa.blr import posterior, gaussian_kl, marginal_loglik
from geometa.inference import GaussianMixturePrediction
from geometa.acquisition import moment_ivr, fantasy_ivr


def test_blr_mean_matches_ridge_and_total_variance():
    torch.manual_seed(12)
    phi = torch.randn(11, 5, dtype=torch.float64)
    y = torch.randn(11, 1, dtype=torch.float64)
    noise_var, tau2 = .13, 1.7
    mean, covariance = posterior(phi, y, torch.zeros(5, 1, dtype=torch.float64),
                                 torch.full((5,), tau2, dtype=torch.float64), noise_var)
    reference = torch.linalg.solve(phi.T@phi+noise_var/tau2*torch.eye(5, dtype=phi.dtype), phi.T@y)
    torch.testing.assert_close(mean, reference, rtol=1e-9, atol=1e-9)
    assert float(torch.linalg.eigvalsh(covariance).min()) > 0
    means = np.array([[0., 1.], [2., 3.]])
    covs = np.array([np.eye(2)*.3, np.eye(2)*.7])
    p = GaussianMixturePrediction(means, covs, np.array([.25, .75]), 1, .1)
    mu, total, within, between = p.moments()
    np.testing.assert_allclose(total, within+between)
    np.testing.assert_allclose(np.diag(total), [1.35, 1.35])
    draws=p.sample_functions(20000,seed=89).reshape(-1,2)
    np.testing.assert_allclose(np.cov(draws.T),total,atol=.04)


def test_gaussian_kl_and_collapsed_likelihood_gradients():
    mu = torch.zeros(3, dtype=torch.float64)
    ls = torch.tensor([-.2, .3, .1], dtype=torch.float64)
    assert abs(float(gaussian_kl(mu, ls, ls))) < 1e-12
    phi = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(5, 1, dtype=torch.float64)
    fun = lambda f: marginal_loglik(f, y, torch.zeros(3, 1, dtype=torch.float64), torch.ones(3, dtype=torch.float64), .1)
    assert torch.autograd.gradcheck(fun, (phi,), eps=1e-6, atol=1e-4)


def test_fantasy_equals_gaussian_ivr_for_one_component():
    rng = np.random.default_rng(19)
    a = rng.normal(size=(7, 4)); cov = a@a.T+.2*np.eye(7)
    p = GaussianMixturePrediction(np.zeros((1, 7)), cov[None], np.ones(1), 1, .1)
    exact = moment_ivr(p, 3, 4)
    sampled = fantasy_ivr(p, 3, 4, fantasies=8, seed=4)
    np.testing.assert_allclose(exact, sampled, rtol=1e-10, atol=1e-10)


def test_multivariate_fantasy_gaussian_and_missing_head_uncertainty():
    rng = np.random.default_rng(91)
    a = rng.normal(size=(12, 7)); cov = a@a.T+.1*np.eye(12)
    p = GaussianMixturePrediction(np.zeros((1, 12)), cov[None], np.ones(1), 2, .02)
    np.testing.assert_allclose(moment_ivr(p, 3, 3), fantasy_ivr(p, 3, 3, fantasies=5), atol=1e-9)
    empty = GaussianMixturePrediction(p.means, np.zeros_like(p.covariances), p.weights, 2, .02)
    assert np.trace(empty.moments()[1]) < np.trace(p.moments()[1])


def test_fantasy_updates_latent_component_probabilities():
    # A single measurement reveals whether the whole function is near +2 or
    # -2. An implementation that updates only Gaussian readouts, leaving
    # component probabilities fixed, would miss almost all of this gain.
    means=np.array([[-2.,-2.,-2.],[2.,2.,2.]])
    covs=np.repeat((.01*np.eye(3))[None],2,axis=0)
    p=GaussianMixturePrediction(means,covs,np.array([.5,.5]),1,.01)
    gains=fantasy_ivr(p,1,2,fantasies=64,seed=91)
    assert 3.99 < gains[0] < 4.01
