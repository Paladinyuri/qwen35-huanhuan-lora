"""将 Base、rank 8/16/32 的回答随机标为 A-D，生成消融实验盲评表。"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r8", type=Path, default=Path("results/evaluation-r8.json"))
    parser.add_argument("--r16", type=Path, default=Path("results/evaluation-r16.json"))
    parser.add_argument("--r32", type=Path, default=Path("results/evaluation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/ablation_blind_eval.csv"))
    parser.add_argument("--key", type=Path, default=Path("results/ablation_blind_key.json"))
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()

    payloads = {name: json.loads(path.read_text(encoding="utf-8"))
                for name, path in (("rank8", args.r8), ("rank16", args.r16), ("rank32", args.r32))}
    by_model = {name: {row["id"]: row for row in payload["rows"]}
                for name, payload in payloads.items()}
    ids = [row["id"] for row in payloads["rank32"]["rows"]]
    rng = random.Random(args.seed)
    labels = ("A", "B", "C", "D")
    output_rows, key = [], {}
    for item_id in ids:
        source = by_model["rank32"][item_id]
        answers = {
            "base": source["base"]["response"],
            "rank8": by_model["rank8"][item_id]["tuned"]["response"],
            "rank16": by_model["rank16"][item_id]["tuned"]["response"],
            "rank32": source["tuned"]["response"],
        }
        models = list(answers)
        rng.shuffle(models)
        mapping = dict(zip(labels, models))
        key[item_id] = mapping
        row = {"id": item_id, "category": source["category"], "prompt": source["prompt"]}
        for label in labels:
            row[f"answer_{label}"] = answers[mapping[label]]
            for metric in ("role", "relevance", "naturalness", "factuality"):
                row[f"{label}_{metric}_1_5"] = ""
        row["best_A_B_C_D_or_tie"] = ""
        row["notes"] = ""
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    args.key.write_text(json.dumps({"seed": args.seed, "key": key}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
