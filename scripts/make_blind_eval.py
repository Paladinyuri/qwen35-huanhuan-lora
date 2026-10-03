"""将 Base/LoRA 回答随机标为 A/B，生成可填写的人工盲评表。"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/blind_eval_form.csv"))
    parser.add_argument("--key", type=Path, default=Path("results/blind_eval_key.json"))
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()

    payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    key = {}
    rows = []
    for item in payload["rows"]:
        a_model, b_model = ("base", "tuned") if rng.random() < 0.5 else ("tuned", "base")
        key[item["id"]] = {"A": a_model, "B": b_model}
        rows.append({
            "id": item["id"], "category": item["category"], "prompt": item["prompt"],
            "answer_A": item[a_model]["response"], "answer_B": item[b_model]["response"],
            "A_role_1_5": "", "B_role_1_5": "",
            "A_relevance_1_5": "", "B_relevance_1_5": "",
            "A_naturalness_1_5": "", "B_naturalness_1_5": "",
            "A_factuality_1_5": "", "B_factuality_1_5": "",
            "preferred_A_B_tie": "", "notes": "",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.key.write_text(json.dumps({"seed": args.seed, "key": key}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())
    print(f"评分映射保存在 {args.key.resolve()}，盲评完成前不要交给评审者。")


if __name__ == "__main__":
    main()
