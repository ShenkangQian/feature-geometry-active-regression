import inspect
import numpy as np
import pytest
import torch
from geometa.data import make_split, make_task, QueryOracle, verify_chain, integration_grid, evaluation_grid, fit_normalization
from geometa.model import GeometricFeatures
from geometa.inference import geometric_predict, linearized_predict
from geometa.acquisition import select_query
from geometa.training import episode, meta_loss


def test_split_disjoint_deterministic_and_training_scaling():
    splits = [make_split(1, name, 5) for name in ("train", "validation", "test")]
    ids = [{t.task_id for t in split} for split in splits]
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    for a,b in zip(splits[0], make_split(1,"train",5)):
        np.testing.assert_array_equal(a.evaluate([-.2,.3]), b.evaluate([-.2,.3]))
    a = fit_normalization(splits[0])
    # Mutating a held-out task cannot affect fitted training normalization.
    splits[2][0].parameters["offset"] = 100000
    b = fit_normalization(splits[0])
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(integration_grid(13), evaluation_grid(13))
    assert not np.isclose(evaluation_grid(97)[:,None], np.linspace(-1,1,49)[None,:], atol=1e-10, rtol=0).any()
    assert not np.isclose(evaluation_grid(97)[:,None], integration_grid(24)[None,:], atol=1e-10, rtol=0).any()


def test_query_api_and_common_noise():
    task = make_task(8,"test",0)
    pool=np.linspace(-1,1,11)
    a=QueryOracle(task,pool,.05,[0],[1],12)
    b=QueryOracle(task,pool,.05,[0],[1],12)
    v=a.query(2); b.query(5)
    np.testing.assert_array_equal(v,b.query(2))
    with pytest.raises(ValueError): a.query(2)
    # Acquisition receives only posterior predictions and index metadata.
    assert not any("label" in name or "truth" in name for name in inspect.signature(select_query).parameters)


def test_chain_modal_truth_matches_direct_solver():
    checks=verify_chain(make_task(13,"test",0,"chain","shift"))
    assert checks["modal_direct_max_abs"]<1e-9
    assert checks["max_residual"]<1e-10


def test_posterior_and_queries_reproducible_all_ablation_modes():
    torch.manual_seed(31); torch.set_num_threads(2)
    model=GeometricFeatures(hidden=5)
    task=make_task(2,"train",0)
    xc=np.array([-.9,-.3,.3,.9]); yc=task.evaluate(xc)
    xp=np.linspace(-1,1,15)
    cfg={"steps":2,"lr":.1,"samples":2,"particles":3}
    for mode in ("full","fixed","position","width","point","no_readout_variance"):
        a=geometric_predict(model,xc,yc,xp,.01,cfg,seed=17,mode=mode)
        b=geometric_predict(model,xc,yc,xp,.01,cfg,seed=17,mode=mode)
        np.testing.assert_array_equal(a.means,b.means)
        np.testing.assert_array_equal(a.covariances,b.covariances)
        assert np.isfinite(a.moments()[1]).all()
    p=linearized_predict(model,xc,yc,xp,.01)
    assert np.linalg.eigvalsh(p.moments()[1]).min()>-1e-8


def test_meta_gradients_reach_shared_features_and_priors():
    torch.manual_seed(14); torch.set_num_threads(2)
    model=GeometricFeatures(hidden=5)
    task=make_task(19,"train",0)
    ep=episode(task,np.random.default_rng(3),np.zeros(1),np.ones(1),5,4,.05)
    settings={"inner_steps":2,"inner_lr":.1,"inner_samples":3,"query_particles":4}
    loss=meta_loss(model,ep,settings,.05**2,seed=7,differentiable=True)
    loss.backward()
    for name in ("base_centres","base_logwidth","position_loadings","width_loadings","readout_mean","readout_logvar","latent_logstd"):
        grad=dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0, name
