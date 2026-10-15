"""Deterministic task splits, analytical pulses, and damped mechanical chains.

Only QueryOracle.query discloses acquisition-pool labels. Integration locations
are unlabeled. Evaluation labels are held by the experiment driver, not supplied
to inference or acquisition functions. This is an API boundary, not a security
sandbox against a caller deliberately inspecting Python attributes.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.linalg import eigh


def derived_seed(seed: int, *parts: object) -> int:
    payload = ":".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


@dataclass
class FunctionTask:
    task_id: str
    family: str
    group: str
    parameters: dict

    @property
    def output_dim(self):
        return 2 if self.family == "chain" else 1

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        p = self.parameters
        if self.family == "chain":
            omega = 0.15 + (x + 1.0) * 1.625
            # eigh eigenvectors are M-orthonormal. Modal decomposition here is
            # an independently checkable exact oracle for this small chain;
            # the learner sees only the queried complex responses.
            eig = np.asarray(p["eigenvalues"])
            residues = np.asarray(p["residues"])
            damping = p["damping"]
            h = (residues[None, :] / (eig[None, :] - omega[:, None] ** 2
                   + 1j * damping * omega[:, None])).sum(axis=1)
            return np.column_stack((h.real, h.imag))
        y = p["offset"] + p["slope"] * x + p["sin_amp"] * np.sin(2.2*x + p["phase"])
        for c, width, amp in zip(p["centres"], p["widths"], p["amplitudes"]):
            u = (x-c) / width
            if self.group == "lorentz":
                y = y + amp / (1 + u*u)
            else:
                y = y + amp * np.exp(-0.5*u*u)
        return y[:, None]


def make_task(seed, split, index, family="pulse", group="id"):
    rng = np.random.default_rng(derived_seed(seed, family, split, group, index))
    task_id = f"{family}:{split}:{group}:{index}:seed{seed}"
    if family == "chain":
        n = 8
        masses = rng.uniform(0.8, 1.2, n)
        springs = rng.uniform(0.7, 1.3, n+1)
        if group == "shift":
            masses[-2:] *= rng.uniform(1.6, 2.3)
        k = np.diag(springs[:-1]+springs[1:]) - np.diag(springs[1:-1], 1) - np.diag(springs[1:-1], -1)
        m = np.diag(masses)
        vals, vecs = eigh(k, m)
        # Force and measurement are collocated at the endpoint. This avoids
        # deliberately hiding modes through an unobservable sensor pairing.
        b = np.zeros(n); b[-1] = 1.
        residues = (b @ vecs)**2
        damping = rng.uniform(.045, .085) if group == "id" else rng.uniform(.015, .035)
        p = {"mass": masses.tolist(), "springs": springs.tolist(), "damping": float(damping),
             "eigenvalues": vals.tolist(), "residues": residues.tolist()}
    else:
        count = 2
        p = {"offset": float(rng.normal(0, .15)), "slope": float(rng.normal(0, .15)),
             "sin_amp": float(rng.uniform(.05, .25)), "phase": float(rng.uniform(-1, 1))}
        sweep = parse_sweep_group(group)
        if sweep is None:
            extent = .52 if group == "id" else .85
            widths = rng.uniform(.07, .18, count) if group == "id" else rng.uniform(.025, .075, count)
            centres = rng.uniform(-extent, extent, count)
        elif sweep[0] == "shift":
            # Single-variable translation stress test. Training centres cover
            # [-.52, .52]. Both peaks sit at +-(.52 + r*width), one on each side,
            # with a fixed width of 0.10 that lies inside the training range.
            # r = 0 is the in-distribution edge; larger r moves the peaks
            # outside the training support by r peak widths. Nothing else
            # differs from the id group.
            ratio = sweep[1]
            widths = np.full(count, .10)
            centres = np.array([-(.52+ratio*.10), .52+ratio*.10])
            p["shift_ratio"] = float(ratio)
        else:
            # Single-variable width stress test. Centres as in id; widths are
            # the id widths multiplied by the scale. Scale 1.0 reproduces the
            # id distribution up to the random stream.
            scale = sweep[1]
            widths = rng.uniform(.07, .18, count)*scale
            centres = rng.uniform(-.52, .52, count)
            p["width_scale"] = float(scale)
        p.update({"centres": centres.tolist(), "widths": widths.tolist(),
                  "amplitudes": rng.uniform(.45, 1.3, count).tolist()})
    return FunctionTask(task_id, family, group, p)


def parse_sweep_group(group):
    """Groups named shift_r<value> or width_s<value> are single-variable sweeps."""
    if group.startswith("shift_r"):
        return "shift", float(group[len("shift_r"):])
    if group.startswith("width_s"):
        return "width", float(group[len("width_s"):])
    return None


def task_covariates(task):
    """Task-level stress variables recorded next to every metric row.

    shift_ratio: configured translation-to-width ratio for sweep groups; for
    other pulse groups the distance of each peak centre beyond the training
    centre range [-.52, .52], divided by that peak's width, maximised over
    peaks (zero when every peak lies inside the range).
    min_width: narrowest pulse width. Chain tasks report the minimum modal
    damping ratio proxy instead (damping / sqrt(min eigenvalue)).
    """
    p = task.parameters
    if task.family != "pulse":
        return {"shift_ratio": float("nan"), "min_width": float("nan"),
                "damping": float(p["damping"]), "min_eigenvalue": float(min(p["eigenvalues"]))}
    if "shift_ratio" in p:
        ratio = float(p["shift_ratio"])
    else:
        ratio = max(max(abs(c)-.52, 0.)/w for c, w in zip(p["centres"], p["widths"]))
    return {"shift_ratio": float(ratio), "min_width": float(min(p["widths"]))}


def make_split(seed, split, count, family="pulse", group="id"):
    return [make_task(seed, split, i, family, group) for i in range(count)]


def integration_grid(n):
    """Unlabeled midpoint quadrature, independent of evaluation grid."""
    return -1 + (np.arange(n)+.5)*2/n


def evaluation_grid(n):
    # Fixed equal-width stratified quadrature with an irrational offset. Unlike
    # nested linspace grids, these points do not coincide with queried pool
    # locations or the midpoint integration grid in our configured experiments.
    offset = (3-np.sqrt(5))/2
    return -1 + (np.arange(n)+offset)*2/n


class QueryOracle:
    def __init__(self, task, pool_x, noise_std, mean, scale, seed):
        self._task = task
        self.pool_x = np.asarray(pool_x)
        self.noise_std = float(noise_std)
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self._seed = seed
        self.queried = []

    def query(self, index):
        index = int(index)
        if index in self.queried:
            raise ValueError("A query cannot be repeated within this experiment.")
        if not 0 <= index < len(self.pool_x):
            raise IndexError(index)
        self.queried.append(index)
        # Per-location noise is common across methods, independent of their
        # query order; otherwise a method can get a luckier observation stream.
        rng = np.random.default_rng(derived_seed(self._seed, self._task.task_id, index))
        y = (self._task.evaluate(self.pool_x[index:index+1])-self.mean)/self.scale
        return y[0] + rng.normal(0, self.noise_std, self._task.output_dim)


def fit_normalization(tasks):
    """Training tasks ONLY. Scaling is never fitted on held-out task labels."""
    values = np.concatenate([t.evaluate(evaluation_grid(129)) for t in tasks])
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-4)


def verify_chain(task):
    """Compare modal oracle against direct complex solves and report residuals."""
    p = task.parameters
    masses, springs = np.array(p["mass"]), np.array(p["springs"])
    n = len(masses)
    k = np.diag(springs[:-1]+springs[1:])-np.diag(springs[1:-1], 1)-np.diag(springs[1:-1], -1)
    m = np.diag(masses); c = p["damping"]*m
    b = np.zeros(n); b[-1] = 1
    xs = np.linspace(-.97, .93, 23); direct = []
    residual = 0.
    for x in xs:
        omega = .15+(x+1)*1.625
        d = k-omega**2*m+1j*omega*c
        q = np.linalg.solve(d, b)
        residual = max(residual, float(np.linalg.norm(d@q-b)))
        direct.append(b@q)
    truth = task.evaluate(xs)
    modal = truth[:, 0]+1j*truth[:, 1]
    return {"modal_direct_max_abs": float(np.max(np.abs(modal-direct))), "max_residual": residual,
            "min_mass": float(masses.min()), "min_eigenvalue": float(min(p["eigenvalues"]))}
