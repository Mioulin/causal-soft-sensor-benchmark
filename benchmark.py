"""Causal soft-sensor benchmark on original simulated six-stage dynamics."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HORIZONS = [1, 3, 6, 12]
WINDOW = 18
STAGES = 6


def simulate(seed, length=240, ood=False):
    rng = np.random.default_rng(seed)
    u = np.zeros(length)
    u[0] = 0.5
    for t in range(1, length):
        u[t] = np.clip(
            0.94 * u[t - 1] + 0.06 * rng.uniform(0.15, 0.85) + rng.normal(0, 0.04),
            0.05,
            0.95,
        )
    x = np.zeros((length, STAGES))
    x[0] = rng.uniform(0.2, 0.5, STAGES)
    for t in range(1, length):
        upstream = np.r_[u[t - 1], x[t - 1, :-1]]
        rate = 0.15 if not ood else 0.24
        x[t] = np.clip(
            x[t - 1]
            + rate * (upstream - x[t - 1])
            - 0.018 * x[t - 1] ** 2
            + rng.normal(0, 0.006, STAGES),
            0,
            1,
        )
    mask = np.eye(STAGES)[(np.arange(length) + seed) % STAGES]
    obs = (x + rng.normal(0, 0.012, x.shape)) * mask
    # Causal auxiliary sensor: noisy outlet proxy available at each timestamp.
    proxy = x[:, -1] + rng.normal(0, 0.025, length)
    return {
        "u": u,
        "x": x,
        "mask": mask,
        "obs": obs,
        "proxy": proxy,
        "seed": seed,
        "ood": ood,
    }


def physics_step(state, u):
    upstream = np.r_[u, state[:-1]]
    return np.clip(state + 0.12 * (upstream - state) - 0.01 * state**2, 0, 1)


def replay(run):
    """All features at t use only samples <=t, including current observed point."""
    n = len(run["u"])
    last = np.full(STAGES, 0.35)
    state = last.copy()
    features = []
    physics = []
    persistence = []
    for t in range(n):
        if t:
            state = physics_step(state, run["u"][t - 1])
        m = run["mask"][t].astype(bool)
        last[m] = run["obs"][t, m]
        state[m] = 0.7 * run["obs"][t, m] + 0.3 * state[m]
        features.append(np.r_[run["u"][t], run["proxy"][t], last, run["mask"][t], state])
        forecast = state.copy()
        out = []
        for h in range(1, max(HORIZONS) + 1):
            forecast = physics_step(forecast, run["u"][t])  # freeze last known control
            if h in HORIZONS:
                out.append(forecast.copy())
        physics.append(out)
        persistence.append(np.tile(last, (len(HORIZONS), 1)))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(physics),
        np.asarray(persistence),
    )


def windows(runs):
    xx = []
    yy = []
    mm = []
    latent = []
    pp = []
    pers = []
    ids = []
    times = []
    for run in runs:
        features, p, persistence = replay(run)
        for t in range(WINDOW - 1, len(features) - max(HORIZONS)):
            future = t + np.asarray(HORIZONS)
            xx.append(features[t - WINDOW + 1 : t + 1])
            yy.append(run["obs"][future])
            mm.append(run["mask"][future])
            latent.append(run["x"][future])
            pp.append(p[t])
            pers.append(persistence[t])
            ids.append(run["seed"])
            times.append(t)
    return {
        k: np.asarray(v)
        for k, v in {
            "features": xx,
            "targets": yy,
            "mask": mm,
            "latent": latent,
            "physics": pp,
            "persistence": pers,
            "run": ids,
            "time": times,
        }.items()
    }


class SensorGRU(nn.Module):
    def __init__(self, features=20):
        super().__init__()
        self.gru = nn.GRU(features, 32, batch_first=True)
        self.head = nn.Linear(32, len(HORIZONS) * STAGES)

    def forward(self, x):
        y, _ = self.gru(x)
        return self.head(y[:, -1]).reshape(-1, len(HORIZONS), STAGES)


def fuse_weights(physics, neural, target, mask, correlated=True):
    # Minimise E[(w e_neural+(1-w)e_physics)^2] on observed validation points.
    ep = physics - target
    en = neural - target
    axis = (0, 2)
    count = mask.sum(axis=axis)
    a = (ep**2 * mask).sum(axis=axis) / count
    b = (en**2 * mask).sum(axis=axis) / count
    c = (ep * en * mask).sum(axis=axis) / count if correlated else np.zeros_like(a)
    return np.clip((a - c) / np.maximum(a + b - 2 * c, 1e-12), 0, 1)


def metrics(pred, data, radius=None):
    rows = []
    for j, h in enumerate(HORIZONS):
        sel = data["mask"][:, j].astype(bool)
        error = pred[:, j] - data["targets"][:, j]
        r = {
            "horizon": h,
            "observed_rmse": float(np.sqrt(np.mean(error[sel] ** 2))),
            "observed_mae": float(np.mean(np.abs(error[sel]))),
            "latent_rmse_simulation_only": float(
                np.sqrt(np.mean((pred[:, j] - data["latent"][:, j]) ** 2))
            ),
        }
        if radius is not None:
            r["interval_coverage"] = float(np.mean(np.abs(error[sel]) <= radius[j]))
            r["interval_width"] = float(2 * radius[j])
        rows.append(r)
    return rows


def bootstrap_delta(a, b, data, h):
    # Difference of pooled RMSE, cluster resampling whole runs, retaining each time series.
    ids = np.unique(data["run"])
    sa = []
    sb = []
    counts = []
    for rid in ids:
        select = (data["run"] == rid)[:, None] & data["mask"][:, h].astype(bool)
        sa.append(np.sum((a[:, h] - data["targets"][:, h])[select] ** 2))
        sb.append(np.sum((b[:, h] - data["targets"][:, h])[select] ** 2))
        counts.append(select.sum())
    rng = np.random.default_rng(992)
    idx = rng.integers(len(ids), size=(3000, len(ids)))
    delta = np.sqrt(np.asarray(sa)[idx].sum(1) / np.asarray(counts)[idx].sum(1)) - np.sqrt(
        np.asarray(sb)[idx].sum(1) / np.asarray(counts)[idx].sum(1)
    )
    return np.quantile(delta, [0.025, 0.975]).tolist()


def run(output, epochs=18):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    groups = {
        "train": list(range(100, 124)),
        "validation": list(range(200, 208)),
        "test": list(range(300, 308)),
        "ood": list(range(400, 408)),
    }
    data = {
        name: windows([simulate(seed, ood=name == "ood") for seed in seeds])
        for name, seeds in groups.items()
    }
    mean = data["train"]["features"].mean((0, 1))
    std = data["train"]["features"].std((0, 1)).clip(0.001)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "dataset": "six-stage-sim-v1",
                "license": "CC0-1.0",
                "groups": groups,
                "length": 240,
                "window": WINDOW,
                "horizons": HORIZONS,
                "epochs": epochs,
                "training_seeds": [42, 43, 44],
                "device": "cpu",
                "torch": torch.__version__,
                "numpy": np.__version__,
                "target": "future noisy rotated observations",
                "features": "only past/current observations, control, proxy, causal filtered state",
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            indent=2,
        )
    )
    xx = {
        k: torch.tensor((v["features"] - mean) / std, dtype=torch.float32) for k, v in data.items()
    }
    target = torch.tensor(data["train"]["targets"], dtype=torch.float32)
    mask = torch.tensor(data["train"]["mask"], dtype=torch.float32)
    table = []
    history = []
    paired = []
    for seed in [42, 43, 44]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = SensorGRU()
        opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
        start = time.perf_counter()
        best = float("inf")
        for epoch in range(epochs):
            model.train()
            losses = []
            for batch in np.array_split(rng.permutation(len(target)), max(1, len(target) // 128)):
                opt.zero_grad()
                pred = model(xx["train"][batch])
                loss = (((pred - target[batch]) ** 2) * mask[batch]).sum() / mask[batch].sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                vp = model(xx["validation"]).numpy()
            vloss = float(
                np.sum((vp - data["validation"]["targets"]) ** 2 * data["validation"]["mask"])
                / data["validation"]["mask"].sum()
            )
            history.append(
                {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)),
                    "validation_loss": vloss,
                }
            )
            if vloss < best:
                best = vloss
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "mean": mean.tolist(),
                        "std": std.tolist(),
                        "epoch": epoch + 1,
                    },
                    output / f"gru_{seed}.pt",
                )
            if epoch % 6 == 0:
                print("seed", seed, "epoch", epoch + 1, "val", vloss, flush=True)
        model.load_state_dict(
            torch.load(output / f"gru_{seed}.pt", weights_only=True)["state_dict"]
        )
        model.eval()
        with torch.inference_mode():
            neural = {k: model(xx[k]).numpy() for k in ["validation", "test", "ood"]}
        val = data["validation"]
        weights = {
            name: fuse_weights(
                val["physics"],
                neural["validation"],
                val["targets"],
                val["mask"],
                correlated=name == "correlated_fusion",
            )
            for name in ["independent_fusion", "correlated_fusion"]
        }
        preds = {}
        for split in ["validation", "test", "ood"]:
            d = data[split]
            preds[split] = {
                "persistence": d["persistence"],
                "physics": d["physics"],
                "gru": neural[split],
            }
            for name, w in weights.items():
                preds[split][name] = (
                    d["physics"] + (neural[split] - d["physics"]) * w[None, :, None]
                )
        for name, vpred in preds["validation"].items():
            radii = np.array(
                [
                    np.quantile(
                        np.abs(vpred[:, h] - val["targets"][:, h])[val["mask"][:, h].astype(bool)],
                        0.95,
                        method="higher",
                    )
                    for h in range(4)
                ]
            )
            for split in ["test", "ood"]:
                pp = preds[split][name]
                for m in metrics(pp, data[split], radii):
                    table.append({"seed": seed, "split": split, "model": name, **m})
        for split in ["test", "ood"]:
            np.savez_compressed(
                output / f"predictions_{seed}_{split}.npz",
                **preds[split],
                target=data[split]["targets"],
                mask=data[split]["mask"],
                run=data[split]["run"],
                time=data[split]["time"],
                latent=data[split]["latent"],
            )
            for h in range(4):
                paired.append(
                    {
                        "seed": seed,
                        "split": split,
                        "horizon": HORIZONS[h],
                        "comparison": "correlated_fusion minus physics RMSE",
                        "ci95": bootstrap_delta(
                            preds[split]["correlated_fusion"],
                            preds[split]["physics"],
                            data[split],
                            h,
                        ),
                    }
                )
        (output / f"training_{seed}.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETED",
                    "elapsed_seconds": time.perf_counter() - start,
                    "weights": {k: v.tolist() for k, v in weights.items()},
                    "best_validation_mse": best,
                },
                indent=2,
            )
        )
    for name, value in [("metrics", table), ("history", history), ("paired", paired)]:
        (output / f"{name}.json").write_text(json.dumps(value, indent=2))
    analyse(output)


def analyse(output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = json.loads((output / "metrics.json").read_text())
    models = [
        "persistence",
        "physics",
        "gru",
        "independent_fusion",
        "correlated_fusion",
    ]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    lines = [
        "# Causal soft-sensor benchmark — measured results",
        "",
        "Original synthetic data only. No legacy industrial observations are redistributed. This is a controlled numerical experiment, not validation on a carbon-capture plant. Six coupled stages, rotating noisy observation, causal auxiliary proxy. 24 training runs, 8 validation, 8 independent test, 8 kinetic-shift test runs. Three GRU optimisation seeds; each split retains the same process realisations.",
        "",
        "| Split | Method | h=1 RMSE | h=3 | h=6 | h=12 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ax, split in zip(axs, ["test", "ood"]):
        for model in models:
            vals = [
                np.mean(
                    [
                        r["observed_rmse"]
                        for r in table
                        if r["split"] == split and r["model"] == model and r["horizon"] == h
                    ]
                )
                for h in HORIZONS
            ]
            ax.plot(HORIZONS, vals, marker="o", label=model)
            lines.append(
                "| " + split + " | " + model + " | " + " | ".join(f"{v:.4f}" for v in vals) + " |"
            )
        ax.set(title=split, xlabel="Forecast horizon (samples)", ylabel="Observed RMSE")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    history = json.loads((output / "history.json").read_text())
    for seed in [42, 43, 44]:
        rows = [r for r in history if r["seed"] == seed]
        axs[0].plot(
            [r["epoch"] for r in rows], [r["validation_loss"] for r in rows], label=str(seed)
        )
    axs[0].set(xlabel="Epoch", ylabel="Validation observed MSE")
    axs[0].legend(title="Seed")
    for split in ["test", "ood"]:
        vals = [
            np.mean(
                [
                    r["interval_coverage"]
                    for r in table
                    if r["split"] == split
                    and r["model"] == "correlated_fusion"
                    and r["horizon"] == h
                ]
            )
            for h in HORIZONS
        ]
        axs[1].plot(HORIZONS, vals, marker="o", label=split)
    axs[1].axhline(0.95, color="black", linestyle="--", label="Nominal .95")
    axs[1].set(xlabel="Horizon", ylabel="Empirical coverage", ylim=(0, 1))
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(output / "diagnostics.png", dpi=160)
    plt.close(fig)
    lines += [
        "",
        "## Coverage and negative comparisons",
        "| Split | Horizon | Correlated fusion coverage | Mean RMSE change versus independent fusion |",
        "|---|---:|---:|---:|",
    ]
    for split in ["test", "ood"]:
        for h in HORIZONS:
            rs = [
                r
                for r in table
                if r["split"] == split and r["horizon"] == h and r["model"] == "correlated_fusion"
            ]
            comp = [
                r
                for r in table
                if r["split"] == split and r["horizon"] == h and r["model"] == "independent_fusion"
            ]
            lines.append(
                f"| {split} | {h} | {np.mean([r['interval_coverage'] for r in rs]):.3f} | {np.mean([r['observed_rmse'] for r in rs]) - np.mean([r['observed_rmse'] for r in comp]):+.6f} |"
            )
    lines += [
        "",
        "## Methods and validity",
        "GRU is trained on future observed points with a masked MSE, never on latent states. Latent-state error is simulation-only secondary analysis. Normalisation uses training runs only. Checkpoints are selected by validation observed MSE. Forecasts assume constant current control; no future measurements, labels or controls enter features. Prefix-invariance is unit-tested.",
        "Fusion estimates second moments of forecast errors on validation data. With physics error ep and neural error en, neural weight w=(E[ep²]-E[ep en])/(E[ep²]+E[en²]-2E[ep en]), clipped to [0,1]. The comparator sets the cross moment to zero. This avoids an unjustified independence assumption but may overfit eight validation runs.",
        "Intervals use the validation 95th percentile of absolute observed residuals per horizon. They are empirical residual intervals, not a distribution-free guarantee under dependent windows or drift. Overlap and temporal autocorrelation invalidate naive row bootstrap; intervals for method differences resample whole test runs. Eight clusters give imprecise intervals; three optimisation seeds do not add independent test trajectories.",
        "## Legacy audit and limitations",
        "The earlier my_v2 pipeline indexes a kinetic prior at t+h. Whether that is leakage depends on when that prior was available; this study does not assume a proven violation. Full-run residual interpolation is suitable as an offline target construction only if never used as an online feature. Saved metric versions differ from the earlier README. None of those old metrics is treated as a result here.",
        "Simulated dynamics favour a related mechanistic model; mismatch and OOD kinetics only partially address that bias. No real-plant transfer, transformer baseline, future-control oracle or industrial deployment is claimed. These extensions are NOT RUN. Units and sample times are dimensionless. Negative comparisons remain in metrics.json.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results")
    p.add_argument("--epochs", type=int, default=18)
    a = p.parse_args()
    run(a.output, a.epochs)
