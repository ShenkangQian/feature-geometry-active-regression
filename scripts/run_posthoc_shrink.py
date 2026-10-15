"""Post-hoc exploratory check (2026-09-24, NOT pre-registered).

The sweep showed that the full posterior moves the feature geometry to a
state whose span residual is higher than at z = 0 in most groups. This script
asks whether shrinking the latent prior at test time (a narrower prior around
z = 0, meta-training unchanged) removes that loss. It reuses the sweep's saved
checkpoints, evaluates the translation groups for two seeds, and writes a
separate output directory. No source file of the package is modified; the
prior is shifted on the loaded model object for the duration of one method.
"""
import argparse
import contextlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import yaml
from geometa.data import make_split, evaluation_grid, derived_seed
from geometa.experiment import evaluate_task, summarize, write_csv
from geometa.inference import oracle_geometry
from geometa.training import load_family, load_basis_gp


@contextlib.contextmanager
def shrunk_prior(model, scale):
    original = model.latent_logstd.data.clone()
    model.latent_logstd.data += math.log(scale)
    try:
        yield
    finally:
        model.latent_logstd.data.copy_(original)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="outputs/sweep-20260924")
    ap.add_argument("--output", default="outputs/posthoc-shrink-20260924")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.3, 0.15])
    args = ap.parse_args()
    sweep, out = Path(args.sweep), Path(args.output)
    if (out/"metrics.csv").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    config = yaml.safe_load((Path("configs")/"sweep.yaml").read_text())
    torch.set_num_threads(config.get("threads", 2)); torch.set_default_dtype(torch.float64); torch.use_deterministic_algorithms(True)
    groups = [g for g in config["data"]["groups"]["pulse"] if g.startswith("shift_r")]
    test_grid = evaluation_grid(config["evaluation"]["test_points"])
    rows = []
    for seed in args.seeds:
        model, gp, mean, scale_, ckpt = load_family(sweep/"checkpoints"/f"pulse_seed{seed}"/"checkpoint.pt")
        bundle = {"model": model, "gp": gp, "gp_basis": load_basis_gp(model, ckpt), "mean": mean, "scale": scale_, "calibration": ckpt.get("calibration")}
        print(json.dumps({"seed": seed, "latent_logstd": model.z_logstd.detach().tolist()}), flush=True)
        for group in groups:
            for task in make_split(seed, "test", config["data"]["test_tasks"], "pulse", group):
                truth = (task.evaluate(test_grid)-mean)/scale_
                z_star, res_star = oracle_geometry(model, test_grid, truth, restarts=2, steps=100, lr=.1, seed=derived_seed(seed, task.task_id, "oracle"))
                oracle = {"z_star": z_star, "residual_star": res_star}
                for method in ("fixed", "full", "linearized"):
                    m, _ = evaluate_task(config, task, bundle, seed, method, out/"predictions", oracle); rows.extend(m)
                for s in args.scales:
                    with shrunk_prior(model, s):
                        m, _ = evaluate_task(config, task, bundle, seed, "full", out/"predictions", oracle)
                    for r in m: r["method"] = f"full_shrink{s:g}"
                    rows.extend(m)
            write_csv(out/"metrics.csv", rows)
            print(json.dumps({"seed": seed, "group": group, "rows": len(rows)}), flush=True)
    summarize(rows, out)
    (out/"completion.json").write_text(json.dumps({"status": "complete", "metric_rows": len(rows), "post_hoc": True,
        "note": "Exploratory test-time prior shrinkage on the sweep checkpoints; not pre-registered.", "scales": args.scales, "seeds": args.seeds}, indent=2))


if __name__ == "__main__":
    main()
