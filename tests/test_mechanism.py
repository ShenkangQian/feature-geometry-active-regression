"""Tests for the 2026-09-24 mechanism additions: repaired linearized ablation,
span-residual diagnostics, oracle geometry, sweep groups and 2x2 labels."""
import numpy as np
import torch
from geometa.data import make_task, parse_sweep_group, task_covariates, evaluation_grid
from geometa.model import GeometricFeatures
from geometa.inference import geometric_predict, linearized_predict, span_residual, oracle_geometry, GaussianMixturePrediction
from geometa.experiment import resolve_method, atom_gap_ratio
from geometa.baselines import BasisGP


def _model():
    torch.manual_seed(5); torch.set_num_threads(2)
    model = GeometricFeatures(hidden=6)
    with torch.no_grad():
        model.readout_mean.zero_()
    return model


def test_linearized_prior_collapses_to_fixed_but_conditional_mean_does_not():
    model = _model()
    task = make_task(3, "test", 0, "pulse", "shift_r3")
    xc = np.array([-.9, -.5, 0., .5, .9]); yc = task.evaluate(xc)
    xp = np.linspace(-1, 1, 21)
    fixed = geometric_predict(model, xc, yc, xp, .01, {"steps": 1, "lr": .1, "samples": 1, "particles": 1}, mode="fixed")
    prior = linearized_predict(model, xc, yc, xp, .01, expansion="prior_mean")
    cond = linearized_predict(model, xc, yc, xp, .01, expansion="conditional_mean")
    # With m0 = 0 the prior-mean expansion has a zero z-block: identical mean.
    np.testing.assert_allclose(prior.moments()[0], fixed.moments()[0], atol=1e-9)
    # The conditional-mean expansion actually moves.
    assert np.abs(cond.moments()[0]-fixed.moments()[0]).max() > 1e-4
    assert np.linalg.eigvalsh(cond.moments()[1]).min() > -1e-8
    assert cond.latent_mean is not None and np.isfinite(cond.latent_mean).all()


def test_span_residual_and_oracle_geometry_are_consistent():
    model = _model()
    task = make_task(11, "test", 1, "pulse", "shift_r2")
    grid = evaluation_grid(61); truth = task.evaluate(grid)
    r0 = span_residual(model, grid, truth, np.zeros(model.latent_dim))
    z_star, r_star = oracle_geometry(model, grid, truth, restarts=3, steps=60, lr=.1, seed=1)
    assert r_star <= r0 + 1e-12
    np.testing.assert_allclose(span_residual(model, grid, truth, z_star), r_star, rtol=1e-8, atol=1e-12)
    # Prop. 2: the least-squares residual lower-bounds the error of any
    # readout on that span, including the data-conditioned oracle_z readout.
    xc = np.array([-.8, -.4, 0., .4, .8]); yc = task.evaluate(xc)
    pred = geometric_predict(model, xc, yc, grid, 1e-4, {"steps": 1, "lr": .1, "samples": 1, "particles": 1}, fixed_z=z_star)
    assert pred.marginal_scores(truth)["mse"] >= r_star - 1e-9


def test_sweep_groups_and_covariates():
    assert parse_sweep_group("shift_r1.5") == ("shift", 1.5)
    assert parse_sweep_group("width_s0.35") == ("width", .35)
    assert parse_sweep_group("id") is None and parse_sweep_group("lorentz") is None
    t = make_task(2, "test", 0, "pulse", "shift_r2")
    np.testing.assert_allclose(sorted(np.abs(t.parameters["centres"])), [.72, .72])
    np.testing.assert_allclose(t.parameters["widths"], [.1, .1])
    assert task_covariates(t)["shift_ratio"] == 2.
    w = make_task(2, "test", 0, "pulse", "width_s0.5")
    assert all(.035-1e-9 <= v <= .09+1e-9 for v in w.parameters["widths"])
    inside = make_task(2, "test", 0, "pulse", "id")
    assert task_covariates(inside)["shift_ratio"] == 0.
    assert np.isfinite(atom_gap_ratio(_model(), inside))
    # Sweep tasks are disjoint from the id/shift tasks used elsewhere.
    assert t.task_id != make_task(2, "test", 0, "pulse", "id").task_id


def test_resolve_method_and_interval_inflation():
    assert resolve_method("random") == ("full", True)
    assert resolve_method("fixed_random") == ("fixed", True)
    assert resolve_method("gp_random") == ("gp", True)
    assert resolve_method("gp_basis") == ("gp_basis", False)
    p = GaussianMixturePrediction(np.zeros((1, 4)), np.eye(4)[None]*.25, np.ones(1), 1, .01)
    lo1, hi1 = p.interval(1.); lo2, hi2 = p.interval(2.)
    assert (hi2-lo2 > hi1-lo1).all()
    s = p.marginal_scores(np.full(4, .8), inflation=2.)
    assert s["coverage95_calibrated"] >= s["coverage95"] and s["width95_calibrated"] > s["width95"]


def test_basis_gp_predicts_and_leaves_features_frozen():
    model = _model()
    gp = BasisGP(model)
    assert all(not p.requires_grad for p in model.parameters())
    x = torch.linspace(-1, 1, 9, dtype=torch.float64); y = torch.sin(3*x)[:, None]
    loss = gp.loss(x, y, .01); loss.backward()
    assert gp.basis_logvar.grad is not None and gp.gp.log_lengths.grad is not None
    pred = gp.predict(x.numpy(), y.numpy(), np.linspace(-1, 1, 5), .01)
    assert np.isfinite(pred.moments()[1]).all()
