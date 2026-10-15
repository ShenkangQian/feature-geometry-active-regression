"""Actual cross-task training through differentiable context adaptation."""
import json
import math
import time
import numpy as np
import torch
from .data import make_split, fit_normalization, derived_seed
from .model import GeometricFeatures
from .baselines import MultiscaleGP, BasisGP
from .inference import as_tensor, adapt_q, geometric_predict
from .blr import conditional_loglik
from .data import evaluation_grid


def episode(task, rng, mean, scale, contexts, queries, noise):
    locations = rng.uniform(-1, 1, contexts+queries)
    yc = (task.evaluate(locations[:contexts])-mean)/scale
    yq = (task.evaluate(locations[contexts:])-mean)/scale
    yc += rng.normal(0, noise, yc.shape)
    yq += rng.normal(0, noise, yq.shape)
    return tuple(as_tensor(x) for x in (locations[:contexts], yc, locations[contexts:], yq))


def meta_loss(model, ep, settings, noise_var, seed, differentiable):
    xc, yc, xq, yq = ep
    mu, ls, mask = adapt_q(model, xc, yc, noise_var, steps=settings["inner_steps"], lr=settings["inner_lr"],
                          samples=settings["inner_samples"], seed=seed, differentiable=differentiable)
    generator = torch.Generator().manual_seed(seed+313)
    eps = torch.randn(settings["query_particles"], model.latent_dim, generator=generator, dtype=torch.float64)
    zs = mu[None, :]+eps*ls.exp()[None, :]
    ll = torch.stack([conditional_loglik(model(xc, z), yc, model(xq, z), yq,
                     model.readout_mean, model.prior_var, noise_var) for z in zs])
    # A finite-mixture JOINT predictive NLL; not a sum of separately mixed
    # pointwise likelihoods. MC approximation and particle count are logged.
    return -(torch.logsumexp(ll, dim=0)-math.log(len(ll)))/yq.numel()


def train_family(config, family, seed, folder):
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(derived_seed(seed, family, "training"))
    data, settings = config["data"], config["training"]
    train = make_split(seed, "train", data["train_tasks"], family)
    valid = make_split(seed, "validation", data["validation_tasks"], family)
    mean, scale = fit_normalization(train)
    model = GeometricFeatures(hidden=config["model"]["hidden"], output_dim=train[0].output_dim,
                              position_dim=config["model"]["position_dim"], width_dim=config["model"]["width_dim"])
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    noise = config["noise_std"]; noise_var = noise**2
    history = []; start = time.perf_counter()
    best_score = float("inf"); best_state = None
    valid_rng = np.random.default_rng(derived_seed(seed, family, "validation-episodes"))
    validation_episodes = [episode(t, valid_rng, mean, scale, settings["contexts"], settings["queries"], noise)
                           for t in valid[:settings["validation_episodes"]]]
    for epoch in range(settings["epochs"]):
        optimizer.zero_grad()
        losses = []
        for b in range(settings["batch_tasks"]):
            task = train[int(rng.integers(len(train)))]
            context_n = int(rng.integers(3, settings["contexts"]+1))
            ep = episode(task, rng, mean, scale, context_n, settings["queries"], noise)
            loss = meta_loss(model, ep, settings, noise_var, derived_seed(seed, family, epoch, b), True)
            losses.append(loss)
        objective = torch.stack(losses).mean()
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(f"Nonfinite meta objective at epoch {epoch}")
        objective.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.))
        optimizer.step()
        record = {"epoch": epoch+1, "train_joint_nll_per_observation": float(objective.detach()),
                  "gradient_norm": grad_norm, "elapsed_seconds": time.perf_counter()-start}
        if (epoch+1)%settings["validate_every"] == 0 or epoch == settings["epochs"]-1:
            # Validation labels select checkpoints, NEVER contribute gradients.
            values = [float(meta_loss(model, ep, settings, noise_var,
                      derived_seed(seed, family, "valid", i), False).detach())
                      for i, ep in enumerate(validation_episodes)]
            score = float(np.mean(values)); record["validation_joint_nll"] = score
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(record)
        if (epoch+1)%settings["validate_every"] == 0 or epoch == settings["epochs"]-1:
            print(json.dumps({"stage": "meta_train", "family": family, "seed": seed, **record}), flush=True)
    model.load_state_dict(best_state)
    # Baseline GP is trained on the SAME training task IDs and normalization.
    gp = MultiscaleGP(); gp_optimizer = torch.optim.Adam(gp.parameters(), lr=.035)
    gp_history = []
    for epoch in range(settings["gp_epochs"]):
        gp_optimizer.zero_grad(); losses = []
        for _ in range(settings["batch_tasks"]):
            task = train[int(rng.integers(len(train)))]
            ep = episode(task, rng, mean, scale, settings["contexts"]+settings["queries"], 1, noise)
            losses.append(gp.loss(ep[0], ep[1], noise_var))
        loss = torch.stack(losses).mean(); loss.backward(); gp_optimizer.step()
        gp_history.append(float(loss.detach()))
    # Semiparametric GP: frozen meta-trained features as explicit basis plus a
    # Matern residual process, trained on the SAME training tasks. It isolates
    # whether a GP loses to geometric adaptation merely for lacking a
    # meta-learned basis. Skipped when no method in the config requests it.
    gp_basis, basis_history = None, []
    if any(m.startswith("gp_basis") for m in config.get("methods", [])):
        gp_basis = BasisGP(model); basis_optimizer = torch.optim.Adam([p for p in gp_basis.parameters() if p.requires_grad], lr=.035)
        for epoch in range(settings["gp_epochs"]):
            basis_optimizer.zero_grad(); losses = []
            for _ in range(settings["batch_tasks"]):
                task = train[int(rng.integers(len(train)))]
                ep = episode(task, rng, mean, scale, settings["contexts"]+settings["queries"], 1, noise)
                losses.append(gp_basis.loss(ep[0], ep[1], noise_var))
            loss = torch.stack(losses).mean(); loss.backward(); basis_optimizer.step()
            basis_history.append(float(loss.detach()))
    # Variance-inflation factor for the full method's intervals, chosen on
    # VALIDATION tasks only and frozen before any held-out task is seen. It is
    # a reported post-hoc correction; a factor far above one measures how
    # overconfident the approximate posterior is at this training budget.
    calibration = calibrate_inflation(model, valid, mean, scale, config, seed)
    elapsed = time.perf_counter()-start
    change = {k: float((v-initial_state[k]).norm()) for k, v in model.state_dict().items()}
    checkpoint = {"model_state": model.state_dict(), "gp_state": gp.state_dict(), "normalization_mean": mean.tolist(),
                  "normalization_scale": scale.tolist(), "family": family, "seed": seed,
                  "model_config": config["model"], "output_dim": train[0].output_dim,
                  "train_ids": [t.task_id for t in train], "validation_ids": [t.task_id for t in valid],
                  "best_validation_nll": best_score, "training_seconds": elapsed,
                  "parameter_change_norm": change, "config": config,
                  "gp_basis_state": None if gp_basis is None else {k: v for k, v in gp_basis.state_dict().items() if not k.startswith("features.")},
                  "calibration": calibration}
    torch.save(checkpoint, folder/"checkpoint.pt")
    (folder/"training.json").write_text(json.dumps({"history": history, "gp_objective": gp_history,
                        "gp_basis_objective": basis_history, "calibration": calibration,
                        "parameter_change_norm": change, "training_seconds": elapsed}, indent=2))
    (folder/"split_manifest.json").write_text(json.dumps({"train": checkpoint["train_ids"],
                              "validation": checkpoint["validation_ids"], "normalization_source": "train only"}, indent=2))
    return model, gp, mean, scale, checkpoint


def calibrate_inflation(model, valid, mean, scale, config, seed):
    """Choose a std inflation factor for `full` intervals on validation tasks.

    For each validation task and each context size in the configured list, a
    random context is drawn from the query pool, the full posterior is
    formed, and pointwise 95% coverage on the evaluation grid is measured for
    every candidate factor. The factor whose mean coverage is closest to 0.95
    is stored. Grid and context sizes are configuration; defaults are used
    when the training block omits them.
    """
    settings = config["training"]
    grid = settings.get("calibration_grid", [1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4.])
    contexts = settings.get("calibration_contexts", [4, 6, 8, 10])
    evaluation = config["evaluation"]
    pool = np.linspace(-1, 1, evaluation["pool_points"]); test = evaluation_grid(evaluation["test_points"])
    rng = np.random.default_rng(derived_seed(seed, "calibration"))
    noise_var = config["noise_std"]**2
    coverage = np.zeros(len(grid)); count = 0
    for task in valid:
        truth = ((task.evaluate(test)-mean)/scale).reshape(-1)
        for n in contexts:
            idx = rng.choice(len(pool), size=n, replace=False)
            yc = (task.evaluate(pool[idx])-mean)/scale + rng.normal(0, config["noise_std"], (n, task.output_dim))
            pred = geometric_predict(model, pool[idx], yc, test, noise_var, config["inference"],
                                     seed=derived_seed(seed, "calibration", task.task_id, n), mode="full")
            for i, factor in enumerate(grid):
                lo, hi = pred.interval(factor)
                coverage[i] += float(np.mean((truth >= lo) & (truth <= hi)))
            count += 1
    coverage /= max(count, 1)
    best = int(np.argmin(np.abs(coverage-.95)))
    return {"factor": float(grid[best]), "grid": [float(g) for g in grid],
            "validation_coverage_by_factor": coverage.tolist(), "validation_episodes": count,
            "context_sizes": [int(c) for c in contexts]}


def load_family(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = GeometricFeatures(output_dim=ckpt["output_dim"], **ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    gp = MultiscaleGP(); gp.load_state_dict(ckpt["gp_state"])
    return model, gp, np.array(ckpt["normalization_mean"]), np.array(ckpt["normalization_scale"]), ckpt


def load_basis_gp(model, ckpt):
    state = ckpt.get("gp_basis_state")
    if state is None:
        return None
    gp_basis = BasisGP(model)
    gp_basis.load_state_dict(state, strict=False)
    return gp_basis

