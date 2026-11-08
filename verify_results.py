"""Recompute metrics and reload every checkpoint against saved predictions."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from benchmark import SensorGRU, metrics, simulate, windows


def verify(root=Path("results")):
    torch.set_num_threads(2)
    table = json.loads((root / "metrics.json").read_text())
    checked = 0
    manifest = {
        "status": "COMPLETED",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "checkpoints": {},
        "files": {},
    }
    for seed in [42, 43, 44]:
        checkpoint = torch.load(root / f"gru_{seed}.pt", weights_only=True)
        model = SensorGRU()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        manifest["checkpoints"][str(seed)] = hashlib.sha256(
            (root / f"gru_{seed}.pt").read_bytes()
        ).hexdigest()
        for split, seeds in [("test", range(300, 308)), ("ood", range(400, 408))]:
            d = windows([simulate(i, ood=split == "ood") for i in seeds])
            saved = np.load(root / f"predictions_{seed}_{split}.npz", allow_pickle=False)
            mean = np.asarray(checkpoint["mean"], dtype=np.float32)
            std = np.asarray(checkpoint["std"], dtype=np.float32)
            with torch.inference_mode():
                pred = model(
                    torch.tensor((d["features"] - mean) / std, dtype=torch.float32)
                ).numpy()
            np.testing.assert_allclose(pred, saved["gru"], atol=1e-7, rtol=1e-6)
            for name in [
                "persistence",
                "physics",
                "gru",
                "independent_fusion",
                "correlated_fusion",
            ]:
                for actual in metrics(saved[name], d):
                    expected = next(
                        r
                        for r in table
                        if r["seed"] == seed
                        and r["split"] == split
                        and r["model"] == name
                        and r["horizon"] == actual["horizon"]
                    )
                    for key, value in actual.items():
                        np.testing.assert_allclose(value, expected[key], atol=1e-10)
                    checked += 1
    for p in sorted(root.glob("*")):
        if p.is_file() and p.name != "verification.json":
            manifest["files"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest["verified_metric_rows"] = checked
    (root / "verification.json").write_text(json.dumps(manifest, indent=2))
    print(f"Checkpoint reloads and {checked} metric rows verified.")


if __name__ == "__main__":
    verify()
