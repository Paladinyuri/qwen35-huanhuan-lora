"""根据训练生成的 history.json 绘制 train/validation loss 曲线。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="训练目录中的 history.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = json.loads(args.history.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"训练历史为空: {args.history}")
    output = args.output or args.history.with_name("loss_curve.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train_loss"] for row in rows], marker="o", label="Train")
    plt.plot(epochs, [row["val_loss"] for row in rows], marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Token-level cross-entropy")
    plt.xticks(epochs)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    print(output.resolve())


if __name__ == "__main__":
    main()
