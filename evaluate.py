"""在相同 system prompt 和解码参数下比较基座模型与 LoRA 模型。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import json
import re
import sys
import statistics
from pathlib import Path

import pytrio as trio

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = "现在你要扮演皇帝身边的女人——甄嬛。请保持角色口吻，并诚实承认未知信息。"
STYLE_MARKERS = ("臣妾", "皇上", "本宫", "嫔妾", "妾身", "宫中", "娘娘", "罢了")


def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text.lower())


def ngrams(text: str, n: int) -> list[str]:
    normalized = normalize_text(text)
    return [normalized[i:i + n] for i in range(max(0, len(normalized) - n + 1))]


def repeated_ngram_rate(text: str, n: int = 3) -> float:
    grams = ngrams(text, n)
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def closest_training_output(response: str, training_outputs: list[str]) -> dict | None:
    """返回与生成文本最相近的训练答案及两种字符级相似度。"""
    if not response or not training_outputs:
        return None
    normalized = normalize_text(response)
    response_grams = set(ngrams(response, 3))
    best = None
    best_score = -1.0
    for candidate in training_outputs:
        candidate_normalized = normalize_text(candidate)
        sequence_ratio = SequenceMatcher(None, normalized, candidate_normalized).ratio()
        candidate_grams = set(ngrams(candidate, 3))
        union = response_grams | candidate_grams
        ngram_jaccard = len(response_grams & candidate_grams) / len(union) if union else 0.0
        score = max(sequence_ratio, ngram_jaccard)
        if score > best_score:
            best_score = score
            best = {"text": candidate, "sequence_ratio": sequence_ratio,
                    "char_3gram_jaccard": ngram_jaccard}
    return best


def render_prompt(tokenizer, text: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def sample(client, tokenizer, text: str) -> dict:
    params = trio.SamplingParams(max_tokens=160, temperature=0.0, seed=42,
                                 stop=[tokenizer.eos_token] if tokenizer.eos_token else ["<|im_end|>"])
    result = client.sample(prompt=trio.ModelInput.from_ints(render_prompt(tokenizer, text)),
                           sampling_params=params, num_samples=1).result()
    sequence = result.sequences[0]
    response = sequence.text.strip()
    return {"response": response, "stop_reason": str(sequence.stop_reason),
            "output_tokens": getattr(result, "output_tokens", None),
            "style_markers": [word for word in STYLE_MARKERS if word in response],
            "repeated_3gram_rate": repeated_ngram_rate(response, 3)}


def summarize(rows: list[dict], model: str, training_outputs: list[str]) -> dict:
    responses = [row[model]["response"] for row in rows]
    repeated = [row[model]["repeated_3gram_rate"] for row in rows]
    similarities = [row[model].get("training_similarity") for row in rows]
    similarities = [item for item in similarities if item]
    return {
        "responses": len(rows),
        "style_marker_rate": sum(bool(row[model]["style_markers"]) for row in rows) / len(rows),
        "empty_response_rate": sum(not response for response in responses) / len(rows),
        "mean_response_characters": statistics.mean(map(len, responses)),
        "length_stop_rate": sum("length" in row[model]["stop_reason"].lower() for row in rows) / len(rows),
        "mean_repeated_3gram_rate": statistics.mean(repeated),
        "high_repetition_rate": sum(rate >= 0.35 for rate in repeated) / len(rows),
        "exact_training_output_matches": sum(response in training_outputs for response in responses) if training_outputs else None,
        "near_training_output_matches": sum(item["sequence_ratio"] >= 0.85 for item in similarities) if similarities else None,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="PyTrio 保存的 trio://... sampler 权重路径")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/eval_prompts.json")
    parser.add_argument("--training-data", type=Path, default=ROOT / "huanhuan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/evaluation.json")
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    service = trio.ServiceClient()
    base = service.create_sampling_client(base_model=args.base_model)
    tuned = service.create_sampling_client(base_model=args.base_model, model_path=args.model_path)
    tokenizer = base.get_tokenizer()
    rows = []
    training_outputs = []
    if args.training_data.exists():
        training_rows = json.loads(args.training_data.read_text(encoding="utf-8"))
        training_outputs = sorted({str(row.get("output", "")).strip() for row in training_rows
                                   if str(row.get("output", "")).strip()})
    for item in prompts:
        row = {"id": item["id"], "category": item["category"], "prompt": item["prompt"],
               "base": sample(base, tokenizer, item["prompt"]),
               "tuned": sample(tuned, tokenizer, item["prompt"])}
        if training_outputs:
            for model in ("base", "tuned"):
                row[model]["training_similarity"] = closest_training_output(
                    row[model]["response"], training_outputs
                )
        rows.append(row)
        print(f'[{item["id"]}] base={row["base"]["response"][:30]!r} tuned={row["tuned"]["response"][:30]!r}')

    summary = {model: summarize(rows, model, training_outputs) for model in ("base", "tuned")}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    category_summary = {
        category: {model: summarize(category_rows, model, training_outputs)
                   for model in ("base", "tuned")}
        for category, category_rows in grouped.items()
    }
    payload = {"base_model": args.base_model, "model_path": args.model_path,
               "system_prompt": SYSTEM_PROMPT, "decoding": {"temperature": 0.0, "seed": 42, "max_tokens": 160},
               "summary": summary, "category_summary": category_summary, "rows": rows,
               "limitations": "风格词命中率只是描述性代理指标；角色一致性、事实性和回答质量仍需人工盲评。"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# 基座模型与 LoRA 模型固定集对比", "", "自动指标仅作辅助，请逐条人工复核。", "",
          "## 分类汇总", "",
          "| 类别 | 模型 | 平均字数 | 风格词命中 | 高重复 | 近似复述 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for category, values in category_summary.items():
        for model in ("base", "tuned"):
            item = values[model]
            md.append(f'| {category} | {model} | {item["mean_response_characters"]:.1f} | '
                      f'{item["style_marker_rate"]:.1%} | {item["high_repetition_rate"]:.1%} | '
                      f'{item["near_training_output_matches"] if item["near_training_output_matches"] is not None else "-"} |')
    md += [""]
    for row in rows:
        similarity = row["tuned"].get("training_similarity")
        similarity_line = (f'**最相似训练答案：** {similarity["text"]} '
                           f'(SequenceMatcher={similarity["sequence_ratio"]:.3f}, '
                           f'3-gram Jaccard={similarity["char_3gram_jaccard"]:.3f})') if similarity else ""
        md += [f'## {row["id"]} · {row["category"]}', "", f'**用户：** {row["prompt"]}', "",
               f'**基座：** {row["base"]["response"]}', "", f'**LoRA：** {row["tuned"]["response"]}', "",
               similarity_line, ""]
    args.output.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

