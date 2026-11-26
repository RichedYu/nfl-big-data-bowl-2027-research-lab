"""Generate private plots and descriptive evidence for a completed probe."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from movement_probe import atomic_json, digest, TASKS

METRICS = ("turn_deg", "entry", "exit", "braking", "recovery_s", "peak_brake",
           "mean_brake", "peak_normal", "p75_normal", "peak_curvature", "p75_curvature")


def median(values):
    values = [x for x in values if x is not None]
    return float(np.median(values)) if values else None


def summarize(drill, trials):
    valid = [t for t in trials if t["base"] is not None]
    players = defaultdict(list)
    for trial in valid:
        players[trial["player"]].append(trial)
    paired = [ts[:2] for ts in players.values() if len(ts) >= 2]
    robustness, variance, redundancy = {}, {}, {}
    for key in METRICS:
        scale = median(abs(t["base"][key]) for t in valid if t["base"][key] is not None)
        conditions = {"no_smoothing": ("smooth_0",), "five_frames": ("smooth_2",),
                      "5Hz": ("5Hz_phase_0", "5Hz_phase_1"),
                      "noise_0.03": tuple(f"noise_0.03_{i}" for i in range(5)),
                      "noise_0.10": tuple(f"noise_0.1_{i}" for i in range(5))}
        robustness[key] = {}
        for condition, labels in conditions.items():
            errors, differences, shifts = [], [], []
            eligible = 0
            for t in valid:
                val = t["base"][key]
                if val is None:
                    continue
                eligible += len(labels)
                per_trial = []
                for label in labels:
                    variant = t["variants"][label]
                    if variant is not None and variant[key] is not None:
                        per_trial.append(abs(variant[key] - val))
                        shifts.append(abs(variant["cut_s"] - t["base"]["cut_s"]))
                        differences.append(abs(variant[key] - val))
                if per_trial:
                    errors.append(median(per_trial))
            error = median(errors)
            robustness[key][condition] = dict(pairs=len(differences), eligible=eligible,
                median_abs_change=error,
                p90_abs_change=float(np.quantile(differences, .9)) if differences else None,
                relative_to_typical_level=error / scale if error is not None and scale and scale > 1e-8 else None,
                median_cut_shift_s=median(shifts))
        pair = np.array([[a["base"][key], b["base"][key]] for a, b in paired
                         if a["base"][key] is not None and b["base"][key] is not None])
        if len(pair) >= 5:
            within = float(np.mean(np.diff(pair, axis=1)**2) / 2)
            between = float(np.var(pair.mean(axis=1), ddof=1) - within / 2)
            variance[key] = dict(pairs=len(pair), within_variance=within,
                                 between_variance_estimate=between,
                                 median_abs_repeat_difference=float(np.median(np.abs(np.diff(pair, axis=1)))))
    per_player = [(ts[0]["baseline"], {k: median(t["base"][k] for t in ts) for k in METRICS})
                  for ts in players.values()]
    for key in ("entry", "turn_deg", "mean_brake", "exit"):
        redundancy[key] = {}
        for j, baseline in enumerate(("forty", "short_shuttle", "three_cone")):
            values = np.array([(b[j], x[key]) for b, x in per_player if b[j] is not None and x[key] is not None])
            rho = None
            if len(values) >= 10:
                def ranks(x):
                    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
                    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]
                a, b = map(ranks, values.T)
                if a.std() and b.std():
                    rho = float(np.corrcoef(a,b)[0,1])
            redundancy[key][baseline] = dict(n=len(values), spearman=rho)
    split = median(x["entry"] for _, x in per_player)
    strata = {}
    if split is not None:
        for label, predicate in (("below_median", lambda v: v < split), ("at_or_above_median", lambda v: v >= split)):
            group = [x for _, x in per_player if predicate(x["entry"])]
            strata[label] = {"n": len(group)} | {k: median(x[k] for x in group) for k in ("entry", "braking", "exit", "turn_deg")}
    return dict(drill=drill, trials=len(trials), players=len({t["player"] for t in trials}),
                valid_trials=len(valid), valid_players=len(players), players_with_two_valid=len(paired),
                valid_with_forty_and_role_label=sum(ts[0]["baseline"][0] is not None and
                    ts[0]["opportunity_counts"][int(TASKS[drill]=="DB")] > 0 for ts in players.values()),
                median={k: median(t["base"][k] for t in valid) for k in METRICS},
                recovery_detected=sum(t["base"]["recovery_s"] is not None for t in valid),
                robustness=robustness, variance=variance, redundancy=redundancy,
                entry_split=split, speed_strata=strata), (paired[0] if paired else valid[:1])


def plot(drill, examples, path):
    fig, grid = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    axes = grid.ravel()
    fig.suptitle(drill.replace("_", " ").title() + " · 2023 development sample")
    for j, record in enumerate(examples, 1):
        time, xy, speed, tan, normal, curvature = record["curves"]
        cut = record["base"]["cut_s"]
        time = np.asarray(time) - cut
        xy = np.asarray(xy) - np.asarray(xy)[0]
        color = f"C{j-1}"
        axes[0].plot(*xy.T, color=color, label=f"Attempt {j}")
        axes[0].scatter(*xy[np.argmin(abs(time))], color=color, s=30)
        for axis, values in zip(axes[1:], (speed, tan, normal, curvature)):
            axis.plot(time, np.array(values, dtype=float), color=color)
        f = record["base"]
        recovery = f'{f["recovery_s"]:.2f} s' if f["recovery_s"] is not None else "unobserved"
        axes[5].text(0,.9-.42*(j-1), f"Attempt {j}\nTurn {f['turn_deg']:.1f}°\nEntry {f['entry']:.2f} → exit {f['exit']:.2f} yd/s\nSpeed loss {f['braking']:.2f} yd/s · recovery {recovery}",
                     transform=axes[5].transAxes, va="top", color=color)
    axes[0].set(xlabel="Δx (yd)",ylabel="Δy (yd)",aspect="equal",title="Path · dot marks turn center")
    axes[0].legend()
    for axis, title, unit in zip(axes[1:], ("Speed","Tangential acceleration","Signed normal acceleration","Signed curvature"),
                                 ("yd/s","yd/s²","yd/s²","1/yd")):
        axis.axvline(0, color=".55", linestyle="--", linewidth=1)
        axis.set(title=title, ylabel=unit, xlabel="Seconds from turn center")
    axes[5].axis("off")
    temp = path.with_suffix(".partial")
    fig.savefig(temp, format="png", dpi=155)
    temp.replace(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    cp = json.loads((run / "checkpoint.json").read_text())
    if digest(run / "trials.json") != cp["phases"]["measurements"]:
        raise ValueError("Trial artifact differs from checkpoint")
    trials = json.loads((run / "trials.json").read_text())
    stats = {}
    for i, drill in enumerate(TASKS, 1):
        stats[drill], examples = summarize(drill, [t for t in trials if t["drill"] == drill])
        if examples:
            plot(drill, examples, run / f"trajectory_{i}.png")
    diagnostics = {k: median(t["diagnostics"][k] for t in trials) for k in trials[0]["diagnostics"]}
    diagnostics["negative_a_frames"] = sum(t["diagnostics"]["negative_a"] for t in trials)
    db_tasks = [drill for drill, group in TASKS.items() if group == "DB"]
    angles = defaultdict(lambda: defaultdict(list))
    for trial in trials:
        if trial["drill"] in db_tasks and trial["base"]:
            angles[trial["player"]][trial["drill"]].append(trial["base"]["turn_deg"])
    differences = [median(group[db_tasks[1]]) - median(group[db_tasks[0]])
                   for group in angles.values() if len(group) == 2]
    paired_angle = dict(players=len(differences), median_90_minus_45_deg=median(differences),
                        count_90_larger=sum(x > 0 for x in differences))
    atomic_json(run / "statistics.json", {"tasks": stats, "diagnostics": diagnostics,
                                          "db_paired_angle": paired_angle})
    cp["summary"] = dict(source_sha256=digest(__file__), matplotlib=matplotlib.__version__,
                         artifacts={p.name:digest(p) for p in sorted(run.glob("trajectory_*.png"))}
                                   | {"statistics.json":digest(run / "statistics.json")})
    atomic_json(run / "checkpoint.json", cp)


if __name__ == "__main__":
    main()
