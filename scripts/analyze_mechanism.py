"""Mechanism analysis for a completed sweep run: theory quantities against
measured errors, the pre-registered scorecard, and the figures the paper cites.

Usage: python scripts/analyze_mechanism.py outputs/sweep-20260924
Writes <run>/mechanism/{figures, tables, mechanism_summary.json, scorecard.md}.
Every number is read from the run's CSV files; nothing is typed in by hand.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"fixed": "#64748b", "full": "#bd5426", "gp": "#186d83", "gp_basis": "#3fa7bf", "oracle_z": "#222222",
          "linearized": "#7a3e9d", "linearized_prior": "#c9a7dc", "position": "#d98c3f", "width": "#e2c044",
          "point": "#8c6a4f", "full_cheap": "#e59a7a", "full_rich": "#7a2f10", "random": "#999999"}


class _Rho:
    def __init__(self, statistic, pvalue): self.statistic, self.pvalue = statistic, pvalue


def spearmanr(x, y):
    """Spearman rank correlation with average ranks for ties (numpy only; the
    scipy.stats import stalls on the development machine). p-value is a
    large-sample t approximation and is reported for orientation only."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    def rank(v):
        order = np.argsort(v, kind="mergesort"); ranks = np.empty(len(v)); sv = v[order]
        i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and sv[j+1] == sv[i]: j += 1
            ranks[order[i:j+1]] = (i+j)/2+1; i = j+1
        return ranks
    if len(x) < 3: return _Rho(float("nan"), float("nan"))
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x); t = rho*np.sqrt(max(n-2, 1)/max(1-rho*rho, 1e-12))
    p = float(2*(1-0.5*(1+math.erf(abs(t)/math.sqrt(2)))))
    return _Rho(rho, p)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def group_key(group):
    if group.startswith("shift_r"):
        return "shift", float(group[len("shift_r"):])
    if group.startswith("width_s"):
        return "width", float(group[len("width_s"):])
    return "other", float("nan")


def paired(auc, group, method, baseline="fixed", resamples=2000, seed=127):
    a = {r["task_id"]: fnum(r["normalized_mse_auc"]) for r in auc if r["group"] == group and r["method"] == method}
    b = {r["task_id"]: fnum(r["normalized_mse_auc"]) for r in auc if r["group"] == group and r["method"] == baseline}
    common = sorted(set(a) & set(b))
    if not common:
        return None
    diff = np.array([a[k]-b[k] for k in common])
    rng = np.random.default_rng(seed)
    boot = rng.choice(diff, size=(resamples, len(diff)), replace=True).mean(axis=1)
    return {"n": len(diff), "mean": float(diff.mean()), "lo": float(np.quantile(boot, .025)), "hi": float(np.quantile(boot, .975))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); args = ap.parse_args()
    run = Path(args.run)
    out = run/"mechanism"; (out/"figures").mkdir(parents=True, exist_ok=True); (out/"tables").mkdir(exist_ok=True)
    auc = read(run/"task_auc.csv"); metrics = read(run/"metrics.csv"); summary = read(run/"summary.csv")
    methods = sorted(set(r["method"] for r in auc))
    groups = sorted(set(r["group"] for r in auc), key=lambda g: (group_key(g)[0], group_key(g)[1]))
    shift_groups = [g for g in groups if group_key(g)[0] == "shift"]
    width_groups = sorted([g for g in groups if group_key(g)[0] == "width"], key=lambda g: -group_key(g)[1])
    r_of = {g: group_key(g)[1] for g in shift_groups}; s_of = {g: group_key(g)[1] for g in width_groups}
    S = {"run": str(run), "n_tasks_per_group": {}, "group_mean_auc": {}, "gain_vs_fixed": {}, "gain_gp_vs_full": {}}

    # Group means and paired gains.
    for g in groups:
        S["group_mean_auc"][g] = {}
        for m in methods:
            vals = [fnum(r["normalized_mse_auc"]) for r in auc if r["group"] == g and r["method"] == m]
            S["group_mean_auc"][g][m] = float(np.mean(vals)) if vals else float("nan")
            S["n_tasks_per_group"][g] = len(vals)
        S["gain_vs_fixed"][g] = {m: paired(auc, g, m) for m in methods if m != "fixed"}
        S["gain_gp_vs_full"][g] = {"gp": paired(auc, g, "gp", "full"), "gp_basis": paired(auc, g, "gp_basis", "full")}

    # Figure 1: gain vs r.
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for m in ("full", "linearized", "linearized_prior", "position", "width", "point", "oracle_z"):
        if m not in methods: continue
        xs = [r_of[g] for g in shift_groups]; ys = [S["gain_vs_fixed"][g][m]["mean"] for g in shift_groups]
        lo = [S["gain_vs_fixed"][g][m]["lo"] for g in shift_groups]; hi = [S["gain_vs_fixed"][g][m]["hi"] for g in shift_groups]
        ax.plot(xs, ys, "o-", color=COLORS.get(m, None), label=m, markersize=4)
        ax.fill_between(xs, lo, hi, color=COLORS.get(m, None), alpha=.12)
    ax.axhline(0, color="k", lw=.8); ax.set(xlabel="r = translation beyond training range / peak width", ylabel="AUC minus fixed AUC (negative favors method)")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/"figures/gain_vs_r.png", dpi=180); plt.close(fig)

    # Figure 2: Proposition 2 diagnostics.
    fixed_rows = [r for r in auc if r["method"] == "fixed" and group_key(r["group"])[0] in ("shift", "width")]
    full_rows = {r["task_id"]: r for r in auc if r["method"] == "full"}
    x_res = np.array([fnum(r["span_residual_z0"]) for r in fixed_rows]); y_mse = np.array([fnum(r["normalized_mse_auc"]) for r in fixed_rows])
    ok = np.isfinite(x_res) & np.isfinite(y_mse)
    rho1 = spearmanr(x_res[ok], y_mse[ok])
    reduction, gain = [], []
    for r in fixed_rows:
        f = full_rows.get(r["task_id"])
        if f is None: continue
        red = fnum(r["span_residual_z0"])-fnum(f["span_residual_zhat"])
        gn = fnum(f["normalized_mse_auc"])-fnum(r["normalized_mse_auc"])
        if np.isfinite(red) and np.isfinite(gn): reduction.append(red); gain.append(gn)
    reduction, gain = np.array(reduction), np.array(gain)
    rho2 = spearmanr(reduction, gain)
    S["prop2"] = {"n_tasks": int(ok.sum()), "spearman_residual_vs_fixed_auc": float(rho1.statistic), "p_residual": float(rho1.pvalue),
                  "n_pairs": int(len(gain)), "spearman_reduction_vs_gain": float(rho2.statistic), "p_reduction": float(rho2.pvalue)}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axs[0].scatter(x_res[ok], y_mse[ok], s=10, alpha=.6, color=COLORS["fixed"]); axs[0].set(xscale="log", yscale="log",
        xlabel="Span residual of truth at z = 0 (Prop. 2 floor for fixed)", ylabel="fixed MSE-AUC", title=f"Spearman rho = {rho1.statistic:.2f}, n = {ok.sum()}")
    axs[1].scatter(reduction, gain, s=10, alpha=.6, color=COLORS["full"]); axs[1].axhline(0, color="k", lw=.8); axs[1].axvline(0, color="k", lw=.8)
    axs[1].set(xlabel="Residual reduction found by posterior: res(z=0) - res(z_hat)", ylabel="full AUC minus fixed AUC", title=f"Spearman rho = {rho2.statistic:.2f}, n = {len(gain)}")
    for ax in axs: ax.grid(alpha=.2)
    fig.savefig(out/"figures/residual_vs_error.png", dpi=180); plt.close(fig)

    # Figure 3: error decomposition of full at the final budget, by r.
    dec = {}
    for g in shift_groups:
        rows = [r for r in auc if r["group"] == g and r["method"] == "full"]
        floor = np.mean([fnum(r["span_residual_zstar"]) for r in rows])
        found = np.mean([fnum(r["span_residual_zhat"])-fnum(r["span_residual_zstar"]) for r in rows])
        rest = np.mean([fnum(r["final_mse"])-fnum(r["span_residual_zhat"]) for r in rows])
        dec[g] = {"representation_floor": float(floor), "geometry_not_found": float(found), "readout_and_posterior": float(rest)}
    S["decomposition_full_final"] = dec
    fig, ax = plt.subplots(figsize=(6.4, 4), constrained_layout=True)
    xs = np.arange(len(shift_groups)); bottom = np.zeros(len(xs))
    for key, color, label in (("representation_floor", "#222222", "residual at z* (best geometry)"),
                              ("geometry_not_found", "#bd5426", "residual(z_hat) - residual(z*)"),
                              ("readout_and_posterior", "#64748b", "final MSE - residual(z_hat)")):
        vals = np.array([max(dec[g][key], 0) for g in shift_groups]); ax.bar(xs, vals, bottom=bottom, color=color, label=label); bottom += vals
    ax.set_xticks(xs); ax.set_xticklabels([f"r={r_of[g]:g}" for g in shift_groups]); ax.set(ylabel="mean squared error at final budget (full)")
    ax.legend(fontsize=8); ax.grid(alpha=.2, axis="y")
    fig.savefig(out/"figures/error_decomposition.png", dpi=180); plt.close(fig)

    # Figure 4: GP position by width scale.
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for m in ("fixed", "full", "gp", "gp_basis", "oracle_z"):
        if m not in methods: continue
        ax.plot([s_of[g] for g in width_groups], [S["group_mean_auc"][g][m] for g in width_groups], "o-", color=COLORS.get(m), label=m, markersize=4)
    ax.set(xlabel="width scale s (test width / training width)", ylabel="mean MSE-AUC", yscale="log"); ax.invert_xaxis(); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/"figures/gp_vs_full_by_width.png", dpi=180); plt.close(fig)

    # Figure 5: variance share and coverage vs r.
    share, cov_raw, cov_cal, cov_by_variant = {}, {}, {}, defaultdict(dict)
    for g in shift_groups:
        rows = [r for r in auc if r["group"] == g and r["method"] == "full"]
        w = np.array([fnum(r["within_variance"]) for r in rows]); b = np.array([fnum(r["between_variance"]) for r in rows])
        share[g] = float(np.mean(b/np.maximum(w+b, 1e-12)))
        cov_raw[g] = float(np.mean([fnum(r["coverage95"]) for r in rows])); cov_cal[g] = float(np.mean([fnum(r["coverage95_calibrated"]) for r in rows]))
        for m in ("full", "full_cheap", "full_rich", "fixed", "gp", "gp_basis", "linearized", "oracle_z"):
            vals = [fnum(r["coverage95"]) for r in auc if r["group"] == g and r["method"] == m]
            if vals: cov_by_variant[m][g] = float(np.mean(vals))
    S["variance_share_full"] = share; S["coverage_full"] = cov_raw; S["coverage_full_calibrated"] = cov_cal; S["coverage_by_method"] = dict(cov_by_variant)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axs[0].plot([r_of[g] for g in shift_groups], [share[g] for g in shift_groups], "o-", color=COLORS["full"]); axs[0].set(xlabel="r", ylabel="between / (within + between) variance, full", ylim=(0, 1))
    for m in ("full", "full_cheap", "full_rich", "gp", "gp_basis", "linearized"):
        if m in cov_by_variant: axs[1].plot([r_of[g] for g in shift_groups], [cov_by_variant[m][g] for g in shift_groups], "o-", color=COLORS.get(m), label=m, markersize=4)
    axs[1].plot([r_of[g] for g in shift_groups], [cov_cal[g] for g in shift_groups], "s--", color=COLORS["full"], label="full, validation-inflated", markersize=4)
    axs[1].axhline(.95, color="k", lw=.8); axs[1].set(xlabel="r", ylabel="pointwise 95% coverage at final budget", ylim=(0, 1.02)); axs[1].legend(fontsize=7)
    for ax in axs: ax.grid(alpha=.2)
    fig.savefig(out/"figures/variance_share_and_coverage.png", dpi=180); plt.close(fig)

    # Figure 6: 2x2 acquisition gains.
    acq = {}
    for base, rnd in (("full", "random"), ("fixed", "fixed_random"), ("gp", "gp_random")):
        if rnd not in methods or base not in methods: continue
        acq[base] = {g: paired(auc, g, rnd, base) for g in shift_groups}
    S["acquisition_gain_random_minus_active"] = {b: {g: v[g] for g in shift_groups} for b, v in acq.items()}
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for base, v in acq.items():
        ax.errorbar([r_of[g] for g in shift_groups], [v[g]["mean"] for g in shift_groups],
                    yerr=[[v[g]["mean"]-v[g]["lo"] for g in shift_groups], [v[g]["hi"]-v[g]["mean"] for g in shift_groups]],
                    fmt="o-", color=COLORS.get(base), label=f"{base}: random minus active", markersize=4, capsize=2)
    ax.axhline(0, color="k", lw=.8); ax.set(xlabel="r", ylabel="AUC(random queries) - AUC(active queries)"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/"figures/acquisition_2x2.png", dpi=180); plt.close(fig)

    # Post-hoc (not pre-registered) group-level view of Proposition 2: how far
    # the representation floor sits below the realised error, and whether the
    # groups with a higher floor are the groups where adaptation pays.
    post = {}
    for g in groups:
        fx = [r for r in auc if r["group"] == g and r["method"] == "fixed"]
        fl = [r for r in auc if r["group"] == g and r["method"] == "full"]
        post[g] = {"mean_residual_z0": float(np.mean([fnum(r["span_residual_z0"]) for r in fx])),
                   "mean_residual_zstar": float(np.mean([fnum(r["span_residual_zstar"]) for r in fx])),
                   "mean_residual_zhat_full": float(np.mean([fnum(r["span_residual_zhat"]) for r in fl])),
                   "mean_final_mse_fixed": float(np.mean([fnum(r["final_mse"]) for r in fx])),
                   "mean_final_mse_full": float(np.mean([fnum(r["final_mse"]) for r in fl])),
                   "gain_full_minus_fixed": S["gain_vs_fixed"][g]["full"]["mean"],
                   "gain_linearized_minus_fixed": S["gain_vs_fixed"][g]["linearized"]["mean"] if "linearized" in methods else float("nan")}
        post[g]["floor_over_realised_fixed"] = post[g]["mean_residual_z0"]/max(post[g]["mean_final_mse_fixed"], 1e-12)
    xs = np.array([post[g]["mean_residual_z0"] for g in groups]); ys = np.array([post[g]["gain_full_minus_fixed"] for g in groups])
    yl = np.array([post[g]["gain_linearized_minus_fixed"] for g in groups])
    rho_g = spearmanr(xs, ys); rho_gl = spearmanr(xs, yl)
    S["posthoc_group_level"] = {"per_group": post, "n_groups": len(groups),
                                "spearman_groupmean_residual_vs_full_gain": float(rho_g.statistic),
                                "spearman_groupmean_residual_vs_linearized_gain": float(rho_gl.statistic),
                                "note": "post hoc; group means over 60 tasks; not part of the pre-registered scorecard"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for g in groups:
        axs[0].scatter(post[g]["mean_residual_z0"], post[g]["gain_full_minus_fixed"], color=COLORS["full"], s=28)
        axs[0].scatter(post[g]["mean_residual_z0"], post[g]["gain_linearized_minus_fixed"], color=COLORS["linearized"], s=28, marker="s")
        axs[0].annotate(g.replace("shift_", "").replace("width_", ""), (post[g]["mean_residual_z0"], post[g]["gain_full_minus_fixed"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    axs[0].axhline(0, color="k", lw=.8); axs[0].set(xscale="log", xlabel="group mean span residual at z = 0 (fixed floor)", ylabel="group mean AUC gain vs fixed",
        title=f"circles full (rho={rho_g.statistic:.2f}), squares linearized (rho={rho_gl.statistic:.2f})")
    axs[1].bar(np.arange(len(groups))-.2, [post[g]["mean_residual_z0"] for g in groups], width=.4, color="#222222", label="floor: residual at z=0")
    axs[1].bar(np.arange(len(groups))+.2, [post[g]["mean_final_mse_fixed"] for g in groups], width=.4, color=COLORS["fixed"], label="realised: fixed final MSE")
    axs[1].set_xticks(np.arange(len(groups))); axs[1].set_xticklabels([g.replace("shift_", "").replace("width_", "") for g in groups], rotation=60, fontsize=7)
    axs[1].set(yscale="log", ylabel="mean squared error"); axs[1].legend(fontsize=7)
    for ax in axs: ax.grid(alpha=.2)
    fig.savefig(out/"figures/posthoc_floor_vs_realised.png", dpi=180); plt.close(fig)

    # Compute table and calibration factors.
    fit = {m: float(np.mean([fnum(r["cumulative_fit_seconds"]) for r in auc if r["method"] == m])) for m in methods}
    S["mean_cumulative_fit_seconds"] = fit
    cal = {}
    for tj in sorted(run.glob("checkpoints/*/training.json")):
        c = json.loads(tj.read_text()).get("calibration", {})
        cal[tj.parent.name] = {"factor": c.get("factor"), "validation_coverage_by_factor": c.get("validation_coverage_by_factor")}
    S["calibration"] = cal

    # Tables.
    with open(out/"tables/group_mean_auc.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "n_tasks"]+methods)
        for g in groups: w.writerow([g, S["n_tasks_per_group"][g]]+[f"{S['group_mean_auc'][g][m]:.4f}" for m in methods])
    with open(out/"tables/gain_vs_fixed.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "method", "n", "mean_gain", "ci_low", "ci_high"])
        for g in groups:
            for m, v in S["gain_vs_fixed"][g].items():
                if v: w.writerow([g, m, v["n"], f"{v['mean']:.5f}", f"{v['lo']:.5f}", f"{v['hi']:.5f}"])

    # Scorecard, following the rules written in PREDICTIONS.md.
    def gm(g, m): return S["group_mean_auc"][g][m]
    def gv(g, m): return S["gain_vs_fixed"][g][m]
    verdict = {}
    r3 = [g for g in shift_groups if r_of[g] == 3.]; r4 = [g for g in shift_groups if r_of[g] == 4.]; r0 = [g for g in shift_groups if r_of[g] == 0.]
    if r3 and r4 and r0:
        big_neg = all(gv(g, "full")["hi"] < 0 for g in r3+r4); big_nonneg = all(gv(g, "full")["mean"] >= 0 for g in r3+r4)
        r0_ok = gv(r0[0], "full")["lo"] <= 0 <= gv(r0[0], "full")["hi"] or gv(r0[0], "full")["mean"] > 0
        verdict["H1"] = "supported" if (big_neg and r0_ok) else ("not supported" if big_nonneg else "inconclusive")
    rho = S["prop2"]["spearman_residual_vs_fixed_auc"]
    verdict["H2"] = "supported" if (rho > .5 and S["prop2"]["n_tasks"] >= 100) else ("not supported" if rho < .2 else "inconclusive")
    rho = S["prop2"]["spearman_reduction_vs_gain"]
    verdict["H3"] = "supported" if rho < -.4 else ("not supported" if rho > -.1 else "inconclusive")
    if "linearized" in methods and "linearized_prior" in methods:
        def frac(g):
            full_g, lin_g = gv(g, "full")["mean"], gv(g, "linearized")["mean"]
            return float("nan") if abs(full_g) < 1e-12 else lin_g/full_g
        small = [frac(g) for g in shift_groups if r_of[g] <= 1]; large = [frac(g) for g in shift_groups if r_of[g] >= 3]
        prior_gap = max(abs(gv(g, "linearized_prior")["mean"]) for g in shift_groups)
        h4_small = all(f >= .5 for f in small if np.isfinite(f)); h4_large = all(f < .5 for f in large if np.isfinite(f))
        closer_to_full_r4 = any(abs(gm(g, "linearized")-gm(g, "full")) < abs(gm(g, "linearized")-gm(g, "fixed")) for g in r4)
        S["H4_detail"] = {"lin_over_full_gain_small_r": small, "lin_over_full_gain_large_r": large, "max_abs_gain_linearized_prior": prior_gap}
        verdict["H4"] = "supported" if (h4_small and h4_large and prior_gap <= 1e-3) else ("not supported" if (closer_to_full_r4 or prior_gap > 1e-3) else "inconclusive")
    if "oracle_z" in methods:
        envelope = all(gm(g, "oracle_z") <= gm(g, "full") for g in groups)
        ratios = {g: (gm(g, "full")-gm(g, "oracle_z"))/max(gm(g, "fixed")-gm(g, "oracle_z"), 1e-12) for g in shift_groups}
        S["H5_recovery_ratio"] = ratios
        ratio_ok = all(ratios[g] < .5 for g in shift_groups if r_of[g] >= 2)
        verdict["H5"] = "supported" if (envelope and ratio_ok) else ("partial" if envelope else "not supported")
    if width_groups and "gp" in methods:
        g1 = [g for g in width_groups if s_of[g] == 1.]; g25 = [g for g in width_groups if s_of[g] == .25]
        if g1 and g25:
            a, b = S["gain_gp_vs_full"][g1[0]]["gp"], S["gain_gp_vs_full"][g25[0]]["gp"]
            flips = np.sign(a["mean"]) != np.sign(b["mean"]) and not (b["lo"] <= 0 <= b["hi"])
            gp_wins_narrow = b["hi"] < 0
            verdict["H6"] = "supported" if flips else ("not supported" if gp_wins_narrow else "inconclusive")
            S["H6_detail"] = {"gp_minus_full_s1": a, "gp_minus_full_s0.25": b}
    if "gp_basis" in methods:
        S["H7_detail"] = {g: {"gp": gm(g, "gp"), "gp_basis": gm(g, "gp_basis"), "full": gm(g, "full")} for g in groups}
        verdict["H7"] = "reported (see H7_detail)"
    if r0 and r4:
        share_up = share[r4[0]] > share[r0[0]]; cov_low = all(cov_raw[g] < .9 for g in shift_groups)
        factors = [c["factor"] for c in cal.values() if c["factor"] is not None]
        S["H8_detail"] = {"share_r0": share[r0[0]], "share_r4": share[r4[0]], "min_factor": min(factors) if factors else None}
        verdict["H8"] = "supported" if (share_up and cov_low and factors and min(factors) > 1.5) else ("not supported" if not (share_up or cov_low) else "partial")
    if "full_rich" in cov_by_variant:
        d = float(np.mean([cov_by_variant["full_rich"][g]-cov_by_variant["full"][g] for g in shift_groups]))
        S["H9_detail"] = {"mean_coverage_gain_full_rich_minus_full": d}
        verdict["H9"] = "supported" if d < .05 else ("not supported" if d >= .10 else "inconclusive")
    if "full" in acq and "fixed" in acq and r0 and r4:
        gf = np.mean([acq["full"][g]["mean"] for g in shift_groups]); gx = np.mean([acq["fixed"][g]["mean"] for g in shift_groups])
        grows = acq["full"][r4[0]]["mean"] > acq["full"][r0[0]]["mean"]
        overlap = all(acq["full"][g]["lo"] <= acq["fixed"][g]["hi"] and acq["fixed"][g]["lo"] <= acq["full"][g]["hi"] for g in shift_groups)
        S["H10_detail"] = {"mean_gain_full": float(gf), "mean_gain_fixed": float(gx), "grows_with_r": bool(grows), "all_intervals_overlap": bool(overlap)}
        verdict["H10"] = "supported" if (gf > gx and grows) else ("inconclusive" if overlap else "not supported")
    S["scorecard"] = verdict
    (out/"mechanism_summary.json").write_text(json.dumps(S, indent=2, default=float))
    lines = ["# Scorecard filled from the recorded run", "", f"Run: `{run}`. Rules: see PREDICTIONS.md. Generated by scripts/analyze_mechanism.py.", "",
             "| ID | Verdict | Key numbers |", "|---|---|---|"]
    detail = {
        "H1": lambda: "; ".join(f"r={r_of[g]:g}: {gv(g,'full')['mean']:+.4f} [{gv(g,'full')['lo']:+.4f}, {gv(g,'full')['hi']:+.4f}]" for g in shift_groups),
        "H2": lambda: f"rho={S['prop2']['spearman_residual_vs_fixed_auc']:.3f}, n={S['prop2']['n_tasks']}",
        "H3": lambda: f"rho={S['prop2']['spearman_reduction_vs_gain']:.3f}, n={S['prop2']['n_pairs']}",
        "H4": lambda: f"lin/full gain ratio small r {['%.2f'%v for v in S['H4_detail']['lin_over_full_gain_small_r']]}, large r {['%.2f'%v for v in S['H4_detail']['lin_over_full_gain_large_r']]}, max |linearized_prior gain| {S['H4_detail']['max_abs_gain_linearized_prior']:.2e}",
        "H5": lambda: "; ".join(f"r={r_of[g]:g}: recovery {S['H5_recovery_ratio'][g]:.2f}" for g in shift_groups),
        "H6": lambda: f"gp-full at s=1: {S['H6_detail']['gp_minus_full_s1']['mean']:+.4f}; at s=0.25: {S['H6_detail']['gp_minus_full_s0.25']['mean']:+.4f} [{S['H6_detail']['gp_minus_full_s0.25']['lo']:+.4f}, {S['H6_detail']['gp_minus_full_s0.25']['hi']:+.4f}]",
        "H7": lambda: "; ".join(f"{g}: gp {v['gp']:.3f}, gp_basis {v['gp_basis']:.3f}, full {v['full']:.3f}" for g, v in S["H7_detail"].items()),
        "H8": lambda: f"share r=0 {S['H8_detail']['share_r0']:.2f}, r=4 {S['H8_detail']['share_r4']:.2f}; coverage " + ", ".join(f"{cov_raw[g]:.2f}" for g in shift_groups) + f"; min inflation factor {S['H8_detail']['min_factor']}",
        "H9": lambda: f"coverage(full_rich) - coverage(full) = {S['H9_detail']['mean_coverage_gain_full_rich_minus_full']:+.3f}",
        "H10": lambda: f"mean gain full {S['H10_detail']['mean_gain_full']:+.4f}, fixed {S['H10_detail']['mean_gain_fixed']:+.4f}, grows with r: {S['H10_detail']['grows_with_r']}",
    }
    for k in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10"):
        if k in verdict:
            try: text = detail[k]()
            except Exception as exc: text = f"(detail unavailable: {exc})"
            lines.append(f"| {k} | {verdict[k]} | {text} |")
    (out/"scorecard.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
