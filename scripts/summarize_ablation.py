"""汇总 rank 8/16/32 的训练与固定集自动指标。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    specs = {
        "rank8": (ROOT / "results/run-r8-clean", ROOT / "results/evaluation-r8.json"),
        "rank16": (ROOT / "results/run-r16-clean", ROOT / "results/evaluation-r16.json"),
        "rank32": (ROOT / "results/run-r32-clean", ROOT / "results/evaluation.json"),
    }
    summary = {}
    for name, (run_dir, evaluation_path) in specs.items():
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        artifacts = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        best = artifacts["best_checkpoint"]
        summary[name] = {
            "rank": int(name.removeprefix("rank")),
            "train_loss": history[best["epoch"] - 1]["train_loss"],
            "validation_loss": best["val_loss"],
            "sampler_size_bytes": best["sampler_size_bytes"],
            **evaluation["summary"]["tuned"],
        }
    output = ROOT / "results/ablation_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
