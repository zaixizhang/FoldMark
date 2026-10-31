"""Calibrate on held-out negatives, aggregate within runs, then plot run means and SDs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from common import METHODS, OPS, IDENTITIES, output_file

GROUP = ["run_id", "method", "operation", "identity"]
TITLES = ["Conservative substitutions", "ProteinMPNN partial redesign", "pLM-guided optimization"]
LABELS = ["Chen et al.", "SynthIDBio-sequence", "SynthIDBio-structure", "FoldMark"]
COLORS = ["#eeeeee", "#89cdff", "#bdbdbd", "#5aa6ce"]


def evaluate(mapping, scores, n_test, n_runs, fpr):
    raw = pd.read_csv(mapping, usecols=GROUP + ["row_id", "sample_id", "seed", "split", "label"],
                      dtype=str, keep_default_na=False)
    values = pd.concat([pd.read_csv(p, dtype={"row_id": str}) for p in scores], ignore_index=True)
    if set(values.row_id) != set(raw.row_id):
        raise ValueError("Score files and manifest must cover exactly the same row IDs")
    df = raw.merge(values[["row_id", "score", "scTM"]], on="row_id", validate="one_to_one")
    if df.empty or not np.isfinite(df.score).all():
        raise ValueError("Every input must have a finite score")
    df["identity"] = df.identity.astype(int)
    if set(df.split) != {"test", "calibration"} or set(df.label) != {"0", "1"}:
        raise ValueError("Supply test positives/negatives and a separate calibration split")
    if set(df[df.split == "test"].sample_id) & set(df[df.split == "calibration"].sample_id):
        raise ValueError("Protein IDs overlap between test and calibration sets")
    if df.duplicated(GROUP + ["sample_id", "label"]).any():
        raise ValueError("More than one score per protein/condition/class")
    if (df.groupby("run_id").seed.nunique() != 1).any() or df.seed.nunique() != n_runs:
        raise ValueError("Require one distinct seed per run")
    axes = (set(METHODS), set(OPS), set(IDENTITIES))
    if any(set(df[c]) != domain for c, domain in zip(GROUP[1:], axes)):
        raise ValueError("Incomplete or unrecognized experiment design")
    groups = df.groupby(GROUP, sort=True)
    if df.run_id.nunique() != n_runs or groups.ngroups != n_runs*len(METHODS)*len(OPS)*len(IDENTITIES):
        raise ValueError("Missing runs or conditions")
    test_ids = set(df.loc[df.split == "test", "sample_id"])
    if len(test_ids) != n_test:
        raise ValueError(f"Expected {n_test} unique test proteins, got {len(test_ids)}")
    cal_ids = set(df.loc[(df.split == "calibration") & (df.label == "0"), "sample_id"])
    stats = []
    decisions = []
    for key, g in groups:
        cal = g[(g.split == "calibration") & (g.label == "0")]
        test = g[g.split == "test"].copy()
        pos = test[test.label == "1"]
        neg = test[test.label == "0"]
        if set(pos.sample_id) != test_ids or set(neg.sample_id) != test_ids:
            raise ValueError(f"Incomplete test pairs: {key}")
        if set(cal.sample_id) != cal_ids or len(cal) * fpr < 1:
            raise ValueError(f"Missing calibration pairs or insufficient FPR resolution: {key}")
        # Strict '>' on the (k+1)-th highest negative permits at most k positives, including ties.
        k = int(np.floor(fpr * len(cal)))
        threshold = np.sort(cal.score.to_numpy())[-1-k]
        test["threshold"] = threshold
        test["detected"] = test.score > threshold
        decisions.append(test)
        tm = np.nan
        if key[-1] == 90:
            if not pos.scTM.between(0, 1).all():
                raise ValueError(f"Missing/invalid scTM for panel d: {key}")
            tm = pos.scTM.mean()
        stats.append(dict(zip(GROUP, key), threshold=threshold, n_cal=len(cal),
                          calibration_fpr=(cal.score > threshold).mean(), n_pos=len(pos),
                          n_neg=len(neg), tpr=(pos.score > threshold).mean(),
                          test_fpr=(neg.score > threshold).mean(), scTM=tm))
    return pd.DataFrame(stats), pd.concat(decisions, ignore_index=True)


def draw(runs, output, fpr):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), layout="constrained")
    width = 0.19
    for i, ax in enumerate(axes.flat):
        detection = i < 3
        metric = "tpr" if detection else "scTM"
        categories = IDENTITIES if detection else OPS
        xs = np.arange(len(categories))
        for j, method in enumerate(METHODS):
            arm = runs[runs.method == method]
            if detection:
                arm = arm[arm.operation == OPS[i]]
            else:
                arm = arm[arm.identity == 90]
            axis = "identity" if detection else "operation"
            values = arm.groupby(axis)[metric].agg(["mean", "std"]).loc[list(categories)]
            ax.bar(xs + (j-1.5)*width, values["mean"], width, yerr=values["std"],
                   color=COLORS[j], label=LABELS[j], error_kw={"lw": 1, "capsize": 0})
        ax.set_xticks(xs, categories if detection else ["Conservative\nsubstitutions", "ProteinMPNN\nredesign", "pLM-guided\noptimization"])
        ax.set_title(TITLES[i] if detection else "Structural fidelity after clean AF3 refolding", fontsize=11)
        ax.set_ylabel(f"TPR at FPR = {100*fpr:g}%" if detection else "scTM")
        ax.set_xlabel("Final sequence identity (%)" if detection else "Editing / optimization at 90% identity")
        if detection:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0.6, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.12, 1.05, "abcd"[i], transform=ax.transAxes, fontweight="bold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mapping", type=Path)
    p.add_argument("--scores", nargs="+", type=Path, required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--fpr", type=float, default=0.001)
    args = p.parse_args()
    if not 0 < args.fpr < 1 or args.n_runs < 2 or args.n_test < 1:
        p.error("Require 0 < FPR < 1, at least two runs, and a nonempty test set")
    runs, decisions = evaluate(args.mapping, args.scores, args.n_test, args.n_runs, args.fpr)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in [("per_run.csv", runs), ("test_decisions.csv", decisions)]:
        with output_file(args.out / name) as handle:
            table.to_csv(handle, index=False)
    draw(runs, args.out / "extended_data_fig5.pdf", args.fpr)
