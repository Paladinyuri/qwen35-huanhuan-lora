"""清洗 Chat-嬛嬛数据，并按用户提示分组做稳定切分。"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_records(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("数据集顶层必须是 JSON 数组")
    records = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction", "")).strip()
        input_text = str(item.get("input", "")).strip()
        output = str(item.get("output", "")).strip()
        if instruction and output:
            records.append({"instruction": instruction, "input": input_text, "output": output})
    return records


def prompt_key(record: dict[str, str]) -> str:
    return record["instruction"] if not record["input"] else f'{record["instruction"]}\n{record["input"]}'


def split_name(key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "valid"
    return "test"


def prepare(source: Path, output_dir: Path, seed: int = 42) -> dict:
    loaded = load_records(source)
    seen = set()
    unique = []
    for row in loaded:
        signature = (row["instruction"], row["input"], row["output"])
        if signature not in seen:
            seen.add(signature)
            unique.append(row)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in unique:
        groups[prompt_key(row)].append(row)

    splits = {name: [] for name in ("train", "valid", "test")}
    for key in sorted(groups):
        splits[split_name(key, seed)].extend(groups[key])

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (output_dir / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    prompt_sets = {name: {prompt_key(row) for row in rows} for name, rows in splits.items()}
    stats = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": seed,
        "raw_records": len(loaded),
        "invalid_records": 0,
        "exact_duplicates_removed": len(loaded) - len(unique),
        "unique_records": len(unique),
        "unique_user_prompts": len(groups),
        "prompts_with_multiple_outputs": sum(len(rows) > 1 for rows in groups.values()),
        "split_records": {name: len(rows) for name, rows in splits.items()},
        "split_user_prompts": {name: len(prompt_sets[name]) for name in splits},
        "prompt_overlap": {
            "train_valid": len(prompt_sets["train"] & prompt_sets["valid"]),
            "train_test": len(prompt_sets["train"] & prompt_sets["test"]),
            "valid_test": len(prompt_sets["valid"] & prompt_sets["test"]),
        },
        "output_length_chars": {
            "min": min(map(len, (r["output"] for r in unique))),
            "mean": sum(map(len, (r["output"] for r in unique))) / len(unique),
            "max": max(map(len, (r["output"] for r in unique))),
        },
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "huanhuan.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output_dir, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

