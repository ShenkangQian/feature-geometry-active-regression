import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .data import make_split, integration_grid, evaluation_grid, QueryOracle, derived_seed, verify_chain, task_covariates
from .inference import geometric_predict, linearized_predict, GaussianMixturePrediction, span_residual, oracle_geometry
from .acquisition import select_query
from .training import train_family, load_family, load_basis_gp


def slice_prediction(pred, point_start, point_end):
    o = pred.output_dim
    ids = np.arange(point_start*o, point_end*o)
    return GaussianMixturePrediction(pred.means[:, ids], pred.covariances[:, ids[:, None], ids], pred.weights, o, pred.noise_var)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def resolve_method(method):
    """Map a method label to (inference base, random-only acquisition flag).

    `random` keeps its historical meaning (full inference, uniform queries).
    Any other label ending in `_random` uses that base's inference with
    uniform queries, giving the 2x2 prediction-vs-acquisition design.
    """
    if method == "random":
        return "full", True
    if method.endswith("_random"):
        return method[:-len("_random")], True
    return method, False


def atom_gap_ratio(model, task):
    """Proposition 1 quantity for the learned fixed feature map: distance from
    each peak centre to the nearest tanh atom centre at z=0, divided by the
    peak width, maximised over peaks. NaN for non-pulse tasks."""
    if task.family != "pulse":
        return float("nan")
    with torch.no_grad():
        z0 = torch.zeros(model.latent_dim, dtype=torch.float64)
        centres = (model.base_centres + .25*(model.normalized_loadings(model.position_loadings)@z0[:model.position_dim])).numpy()
    return float(max(np.min(np.abs(centres-c))/w for c, w in zip(task.parameters["centres"], task.parameters["widths"])))


def predict_for(method_base, bundle, config, xc, yc, all_x, seed, task, step, oracle):
    model, noise_var = bundle["model"], config["noise_std"]**2
    inference_seed = derived_seed(seed, task.task_id, "full" if method_base == "no_readout_variance" else method_base, step)
    if method_base == "gp":
        return bundle["gp"].predict(xc, yc, all_x, noise_var)
    if method_base == "gp_basis":
        if bundle.get("gp_basis") is None:
            raise ValueError("gp_basis requested but no basis GP was trained for this checkpoint")
        return bundle["gp_basis"].predict(xc, yc, all_x, noise_var)
    if method_base == "linearized":
        return linearized_predict(model, xc, yc, all_x, noise_var, expansion="conditional_mean")
    if method_base == "linearized_prior":
        return linearized_predict(model, xc, yc, all_x, noise_var, expansion="prior_mean")
    if method_base == "oracle_z":
        return geometric_predict(model, xc, yc, all_x, noise_var, config["inference"], seed=inference_seed, fixed_z=oracle["z_star"])
    if method_base in config.get("inference_variants", {}):
        # Same full nonlinear posterior with a different inference budget
        # (fewer or more VI steps and particles), for compute matching and
        # for attributing coverage to approximation error.
        return geometric_predict(model, xc, yc, all_x, noise_var, config["inference_variants"][method_base],
                                 seed=derived_seed(seed, task.task_id, "full", step), mode="full")
    return geometric_predict(model, xc, yc, all_x, noise_var, config["inference"], seed=inference_seed, mode=method_base)


def evaluate_task(config, task, bundle, seed, method, traces_folder, oracle=None):
    settings = config["evaluation"]
    model, mean, scale = bundle["model"], bundle["mean"], bundle["scale"]
    pool = np.linspace(-1, 1, settings["pool_points"])
    quad = integration_grid(settings["integration_points"])
    test = evaluation_grid(settings["test_points"])
    all_x = np.concatenate((pool, quad, test))
    eval_start = len(pool)+len(quad)
    truth = (task.evaluate(test)-mean)/scale
    oracle = oracle or {}
    oracle_needed = method.startswith("oracle_z")
    if oracle_needed and "z_star" not in oracle:
        raise ValueError("oracle_z requires precomputed z_star")
    oracle_query = QueryOracle(task, pool, config["noise_std"], mean, scale, seed)
    initial = np.unique(np.linspace(0, len(pool)-1, settings["initial_queries"], dtype=int))
    x_c, y_c = [], []
    for j in initial:
        x_c.append(pool[j]); y_c.append(oracle_query.query(j))
    base, random_only = resolve_method(method)
    covariates = task_covariates(task)
    covariates["atom_gap_ratio"] = atom_gap_ratio(model, task)
    # Proposition 2 diagnostics that do not depend on the context: residual of
    # the true function outside the fixed span (z=0) and outside the best
    # span found by the truth-fitted oracle geometry.
    residual_z0 = span_residual(model, test, truth, np.zeros(model.latent_dim))
    residual_star = oracle.get("residual_star", float("nan"))
    calibration = bundle.get("calibration") or {}
    inflation = calibration.get("factor") if base in ("full", "position", "width", "point", "no_readout_variance") or base in config.get("inference_variants", {}) else None
    rows, trace = [], []
    fit_total = 0.; acquisition_total = 0.
    for step in range(settings["additional_queries"]+1):
        start = time.perf_counter()
        xc, yc = np.array(x_c), np.array(y_c)
        prediction = predict_for(base, bundle, config, xc, yc, all_x, seed, task, step, oracle)
        fit_elapsed = time.perf_counter()-start; fit_total += fit_elapsed
        evaluated = slice_prediction(prediction, eval_start, len(all_x))
        scores = evaluated.marginal_scores(truth, inflation=inflation)
        scores.update({"family": task.family, "group": task.group, "task_id": task.task_id,
                       "seed": seed, "method": method, "queries": len(x_c), "step": step,
                       "fit_seconds": fit_elapsed, "cumulative_fit_seconds": fit_total,
                       "cumulative_acquisition_seconds": acquisition_total, **covariates,
                       "span_residual_z0": residual_z0, "span_residual_zstar": residual_star})
        if prediction.latent_mean is not None:
            scores["span_residual_zhat"] = span_residual(model, test, truth, prediction.latent_mean)
            scores["latent_norm"] = float(np.linalg.norm(prediction.latent_mean))
        # Error in physical/original output units is reported too. Primary MSE
        # uses training-only scaling so outputs do not dominate by units.
        pred_mean, _, _, _ = evaluated.moments()
        scores["mse_original_units"] = float(np.mean(((pred_mean.reshape(truth.shape)-truth)*scale)**2))
        rows.append(scores)
        if step == settings["additional_queries"]:
            mu, cov, _, _ = evaluated.moments()
            traces_folder.mkdir(parents=True, exist_ok=True)
            safe_id = task.task_id.replace(":", "_")
            np.savez_compressed(traces_folder/f"{safe_id}_{method}.npz", x=test, truth=truth,
                 mean=mu.reshape(truth.shape), variance=np.diag(cov).reshape(truth.shape),
                 queried_x=xc, queried_y=yc, scale=scale, offset=mean)
            break
        start = time.perf_counter()
        acquisition_prediction = slice_prediction(prediction, 0, eval_start)
        available = np.setdiff1d(np.arange(len(pool)), oracle_query.queried)
        j, info = select_query(acquisition_prediction, len(pool), len(quad), available,
                       derived_seed(seed, task.task_id, "acquisition", step), settings["exploration"],
                       settings["acquisition"], settings["fantasies"], random_only=random_only)
        acquisition_total += time.perf_counter()-start
        trace.append({"step": step, "query_index": j, "query_x": float(pool[j]), **info})
        x_c.append(pool[j]); y_c.append(oracle_query.query(j))
    return rows, {"task_id": task.task_id, "method": method, "initial_indices": initial.tolist(),
                  "queries": trace, "parameters": task.parameters, "query_count": len(oracle_query.queried),
                  "oracle_z_star": None if "z_star" not in oracle else np.asarray(oracle["z_star"]).tolist()}


def summarize(rows, output):
    groups = {}
    for row in rows:
        key = (row["family"], row["group"], row["method"], row["queries"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, values in sorted(groups.items()):
        record = dict(zip(("family", "group", "method", "queries"), key)); record["n_tasks"] = len(values)
        for metric in ("mse", "rmse", "latent_nll", "coverage95", "width95", "cumulative_fit_seconds", "cumulative_acquisition_seconds"):
            arr = np.array([v[metric] for v in values])
            record[metric+"_mean"] = float(arr.mean())
            record[metric+"_std"] = float(arr.std(ddof=1)) if len(arr)>1 else 0.
        summaries.append(record)
    write_csv(output/"summary.csv", summaries)
    task_groups = {}
    for row in rows:
        key = (row["family"], row["group"], row["method"], row["task_id"])
        task_groups.setdefault(key, []).append(row)
    auc = []
    for key, values in sorted(task_groups.items()):
        values.sort(key=lambda r: r["queries"])
        xs = np.array([v["queries"] for v in values]); ys = np.array([v["mse"] for v in values])
        value = float(np.trapezoid(ys, xs)/max(xs[-1]-xs[0], 1))
        last = values[-1]
        extra = {k: last.get(k, float("nan")) for k in ("shift_ratio", "min_width", "atom_gap_ratio",
                 "span_residual_z0", "span_residual_zstar", "span_residual_zhat", "coverage95",
                 "coverage95_calibrated", "within_variance", "between_variance", "cumulative_fit_seconds")}
        auc.append(dict(zip(("family", "group", "method", "task_id"), key), normalized_mse_auc=value,
                        final_mse=ys[-1], final_queries=int(xs[-1]), **extra))
    write_csv(output/"task_auc.csv", auc)
    # Paired bootstrap because methods evaluate identical task IDs. Tasks from
    # different model seeds are independent draws; this is a small local CI,
    # not a substitute for a full multi-training-seed paper experiment.
    paired = []
    for family, group in sorted(set((r["family"], r["group"]) for r in auc)):
        full = {r["task_id"]: r["normalized_mse_auc"] for r in auc if r["family"]==family and r["group"]==group and r["method"]=="full"}
        for method in sorted(set(r["method"] for r in auc)-{"full"}):
            other = {r["task_id"]: r["normalized_mse_auc"] for r in auc if r["family"]==family and r["group"]==group and r["method"]==method}
            common = sorted(set(full)&set(other))
            if not common: continue
            diff = np.array([full[k]-other[k] for k in common]); rng = np.random.default_rng(127)
            boot = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
            paired.append({"family": family, "group": group, "baseline": method, "n_tasks": len(diff),
                 "full_minus_baseline_auc": float(diff.mean()), "ci_low": float(np.quantile(boot,.025)),
                 "ci_high": float(np.quantile(boot,.975)), "interpretation": "negative favors full"})
    write_csv(output/"paired_comparisons.csv", paired)
    return summaries


def plot_results(rows, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = summarize(rows, folder)
    pairs = sorted(set((r["family"], r["group"]) for r in summary))
    fig, axs = plt.subplots(1, len(pairs), figsize=(5*len(pairs), 4), squeeze=False)
    for ax, (family, group) in zip(axs[0], pairs):
        methods = sorted(set(r["method"] for r in summary if r["family"]==family and r["group"]==group))
        for method in methods:
            vals = [r for r in summary if r["family"]==family and r["group"]==group and r["method"]==method]
            vals.sort(key=lambda r:r["queries"])
            ax.plot([v["queries"] for v in vals], [v["mse_mean"] for v in vals], marker=".", label=method)
        ax.set(title=f"{family} / {group}", xlabel="Queries", ylabel="Latent MSE (training-normalized)")
        ax.set_yscale("log"); ax.grid(alpha=.2)
    axs[0][-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder/"learning_curves.png", dpi=160); plt.close(fig)


def run_experiment(config, output, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"metrics.csv").exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite prior evidence: {output}; use a new output directory.")
    torch.set_num_threads(config.get("threads", 2)); torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    (output/"config.json").write_text(json.dumps(config, indent=2))
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    (output/"source_manifest.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                          for p in source_files}, indent=2))
    (output/"environment.json").write_text(json.dumps({"python":sys.version, "torch":torch.__version__,
             "numpy":np.__version__, "platform":platform.platform(), "device":"cpu", "threads":torch.get_num_threads()}, indent=2))
    rows, traces, test_manifest, oracle_checks = [], [], [], []
    start = time.perf_counter()
    for seed in config["seeds"]:
        for family in config["families"]:
            ckdir = output/"checkpoints"/f"{family}_seed{seed}"
            if resume and (ckdir/"checkpoint.pt").exists():
                model, gp, mean, scale, ckpt = load_family(ckdir/"checkpoint.pt")
            else:
                model, gp, mean, scale, ckpt = train_family(config, family, seed, ckdir)
            bundle = {"model": model, "gp": gp, "gp_basis": load_basis_gp(model, ckpt), "mean": mean,
                      "scale": scale, "calibration": ckpt.get("calibration")}
            oracle_settings = config.get("oracle", {"restarts": 4, "steps": 150, "lr": .1})
            test_grid = evaluation_grid(config["evaluation"]["test_points"])
            group_names = config["data"]["groups"][family]
            for group in group_names:
                tasks = make_split(seed, "test", config["data"]["test_tasks"], family, group)
                seen = set(ckpt["train_ids"]+ckpt["validation_ids"])
                assert not seen & {t.task_id for t in tasks}
                test_manifest.extend(t.task_id for t in tasks)
                for task in tasks:
                    if family == "chain":
                        oracle_checks.append({"task_id":task.task_id, **verify_chain(task)})
                    # Truth-fitted oracle geometry: read by the evaluator only,
                    # never by inference or acquisition of any ranked method.
                    truth = (task.evaluate(test_grid)-mean)/scale
                    z_star, residual_star = oracle_geometry(model, test_grid, truth, restarts=oracle_settings["restarts"],
                                                            steps=oracle_settings["steps"], lr=oracle_settings["lr"],
                                                            seed=derived_seed(seed, task.task_id, "oracle"))
                    oracle = {"z_star": z_star, "residual_star": residual_star}
                    for method in config["methods"]:
                        metrics, trace = evaluate_task(config, task, bundle, seed, method, output/"predictions", oracle)
                        rows.extend(metrics); traces.append(trace)
                    write_csv(output/"metrics.csv", rows)
                    (output/"query_logs.json").write_text(json.dumps(traces, indent=2))
                    print(json.dumps({"stage":"evaluate", "task":task.task_id, "records":len(rows),
                          "elapsed_seconds":round(time.perf_counter()-start, 2)}), flush=True)
    (output/"test_manifest.json").write_text(json.dumps(test_manifest, indent=2))
    (output/"oracle_checks.json").write_text(json.dumps(oracle_checks, indent=2))
    plot_results(rows, output)
    (output/"completion.json").write_text(json.dumps({"status":"complete", "metric_rows":len(rows),
            "test_tasks":len(test_manifest), "elapsed_seconds":time.perf_counter()-start,
            "methods":config["methods"], "external_baselines_not_reproduced":["ALPaCA original", "LUMA", "ADKF-IFT", "Neural Processes", "AAA", "vector fitting"]}, indent=2))
    return rows
