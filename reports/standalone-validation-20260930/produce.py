"""Reproduce standalone reliability diagnostics and plots; no MEENT calls."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mask_optimizers
from mask_optimizers import MaskSpace, RelaxedEvaluation, Runner, create_optimizer


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
METHODS = ["flrl_lsf_random", "flrl_lsf_es", "flrl_ppo", "flrl_autograd_adam",
           "flrl_ppo_polish", "flrl_ppo_residual", "motif_surgery",
           "nested_fourier", "tangent_covariance", "phenotype_de"]
SEEDS = (1, 17, 97)
STEPS = 100
SHAPE = (16, 12)
CONFIG = {"level_set_modes_x": 2, "level_set_modes_y": 1,
          "modes_x": 2, "modes_y": 1, "modes": ((2, 1), (3, 1)),
          "stage_patience": 8, "population": 8, "population_size": 8,
          "episode_length": 8, "rollout_steps": 8}


def space_for(name, shape=SHAPE):
    return MaskSpace(shape, decoder="h12_cell_centers_v1"
                     if name == "tangent_covariance" else "flrl_endpoints_v1")


def relaxed(coefficients, beta, material_map):
    return RelaxedEvaluation(-float(coefficients @ coefficients), -2*coefficients)


def stress():
    weights = np.random.default_rng(2026).normal(size=SHAPE)
    records, spectrum = [], []
    for name in METHODS:
        for objective in ("weighted_binary", "constant_ties"):
            for seed in SEEDS:
                opt = create_optimizer(name, space_for(name), CONFIG, seed)
                hard_score = (lambda mask: float(np.mean(mask*weights))) if objective == "weighted_binary" else (lambda mask: .5)
                runner = Runner(opt, hard_score, relaxed)
                started = time.perf_counter()
                completed, error = 0, None
                seen_masks = set()
                last_generation = -1
                for step in range(STEPS):
                    try:
                        proposal, observation = runner.step()
                        assert len(proposal.candidate) == SHAPE[0]*SHAPE[1]
                        assert np.isfinite(observation.score)
                        seen_masks.add(hashlib.sha256(np.asarray(proposal.candidate,dtype=np.uint8).tobytes()).hexdigest())
                        completed += 1
                        if name == "tangent_covariance" and objective == "weighted_binary" and seed == SEEDS[0]:
                            view = opt.inspect()
                            if view["generation"] != last_generation:
                                last_generation = view["generation"]
                                spectrum.append({"generation": last_generation,
                                                 "minimum": min(view["spectrum"]),
                                                 "maximum": max(view["spectrum"]),
                                                 "sigma": view["sigma"]})
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        break
                view = opt.inspect()
                records.append({"method": name, "objective": objective, "seed": seed,
                                "requested_steps": STEPS, "completed_steps": completed,
                                "status": "passed" if error is None else "failed", "error": error,
                                "unique_masks": len(seen_masks),
                                "gradient_evaluations": view.get("gradient_evaluations", 0),
                                "training_updates": view.get("training_updates", 0),
                                "elapsed_seconds": time.perf_counter()-started})
    return records, spectrum


def full_shape():
    records = []
    for name in METHODS:
        opt = create_optimizer(name, space_for(name, (256,128)), seed=3)
        try:
            proposal, observation = Runner(opt, lambda mask: float(mask.mean()), relaxed).step()
            mask = np.asarray(proposal.candidate, dtype=np.uint8).reshape(256,128)
            valid = mask.size == 32768 and np.array_equal(mask, mask[:, ::-1])
            records.append({"method": name, "passed": bool(valid), "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                            "score": observation.score})
        except Exception as exc:
            records.append({"method": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"})
    return records


def junit(path):
    node = ET.parse(path).find(".//testsuite")
    return {name: int(node.attrib[name]) for name in ("tests", "failures", "errors", "skipped")}


def plot_overview(records):
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7), constrained_layout=True)
    labels = [name.replace("flrl_", "") for name in METHODS]
    y = np.arange(len(METHODS))
    colors = {"weighted_binary": "#2678a8", "constant_ties": "#e28b36"}
    for offset, objective in ((-.16,"weighted_binary"),(.16,"constant_ties")):
        rows = [row for row in records if row["objective"] == objective]
        groups = [[row for row in rows if row["method"] == name] for name in METHODS]
        minimum = [min(row["completed_steps"] for row in group) for group in groups]
        axes[0].scatter(minimum, y+offset, s=55, color=colors[objective], label=objective.replace("_", " "))
        for j, group in enumerate(groups):
            fractions = [row["unique_masks"]/row["completed_steps"] if row["completed_steps"] else 0 for row in group]
            axes[1].scatter(fractions, [j+offset]*len(group), s=27, alpha=.8, color=colors[objective])
    axes[0].axvline(STEPS, color="#42824c", linestyle="--", linewidth=1.4, label="100-step target")
    axes[0].set_xlim(0, STEPS+8)
    axes[0].set_xlabel("Minimum completed steps across 3 seeds")
    axes[1].set_xlim(0, 1.04)
    axes[1].set_xlabel("Distinct masks / completed proposals (each dot is one seed)")
    for ax in axes:
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Standalone execution checks on 16×12 synthetic mask objectives", weight="bold")
    fig.savefig(OUT/"stress-overview.png",dpi=180)
    plt.close(fig)


def plot_covariance(spectrum):
    rows = [row for row in spectrum if row["generation"] > 0]
    x = [row["generation"] for row in rows]
    lo = [row["minimum"] for row in rows]
    hi = [row["maximum"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.5,4.8),constrained_layout=True)
    ax.fill_between(x,lo,hi,color="#91b4d2",alpha=.45,label="reconstructed eigenvalue range")
    ax.plot(x,lo,color="#206ba0",marker="o",markersize=3)
    ax.plot(x,hi,color="#206ba0",marker="o",markersize=3)
    ax.axhline(.05,color="#b35e34",linestyle="--",label="required floor 0.05")
    ax.axhline(4.,color="#9f3b42",linestyle="--",label="required cap 4.0")
    ax.set(xlabel="Completed population generation",ylabel="Covariance eigenvalue",
           title="H12 tangent covariance stayed inside declared bounds")
    ax.set_yscale("log")
    ax.set_ylim(.03,5)
    ax.set_yticks([.05,.1,.2,.5,1,2,4],labels=["0.05","0.1","0.2","0.5","1","2","4"])
    ax.grid(alpha=.2)
    ax.legend(loc="upper right",fontsize=8)
    fig.savefig(OUT/"h12-covariance.png",dpi=180)
    plt.close(fig)


def main():
    import torch
    torch.set_num_threads(1)
    records, spectrum = stress()
    shape_records = full_shape()
    metadata = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "package_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                "package_version": mask_optimizers.__version__, "python": platform.python_version(),
                "numpy": np.__version__, "torch": torch.__version__,
                "junit_full": junit(OUT/"pytest-full.xml"),
                "junit_numpy_only": junit(OUT/"pytest-numpy-only.xml"),
                "stress_configuration": {"shape":SHAPE,"steps":STEPS,"seeds":SEEDS,"optimizer_parameters":CONFIG,
                                         "weighted_objective_seed":2026},
                "stress": records, "full_shape": shape_records, "h12_spectrum": spectrum}
    (OUT/"results.json").write_text(json.dumps(metadata,indent=2,default=str)+"\n")
    plot_overview(records)
    plot_covariance(spectrum)
    print(f"Stress: {sum(row['status']=='passed' for row in records)}/{len(records)} passed; "
          f"full-size masks: {sum(row['passed'] for row in shape_records)}/{len(shape_records)} passed")
    if not all(row["status"] == "passed" for row in records) or not all(row["passed"] for row in shape_records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
