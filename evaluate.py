"""在相同 system prompt 和解码参数下比较基座模型与 LoRA 模型。"""
from __future__ import annotations

import argparse
import json
import sys
import statistics
from pathlib import Path

import pytrio as trio

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = "现在你要扮演皇帝身边的女人——甄嬛。请保持角色口吻，并诚实承认未知信息。"
STYLE_MARKERS = ("臣妾", "皇上", "本宫", "嫔妾", "妾身", "宫中", "娘娘", "罢了")


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
            "style_markers": [word for word in STYLE_MARKERS if word in response]}


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
    training_outputs = set()
    if args.training_data.exists():
        training_rows = json.loads(args.training_data.read_text(encoding="utf-8"))
        training_outputs = {str(row.get("output", "")).strip() for row in training_rows}
    for item in prompts:
        row = {"id": item["id"], "category": item["category"], "prompt": item["prompt"],
               "base": sample(base, tokenizer, item["prompt"]),
               "tuned": sample(tuned, tokenizer, item["prompt"])}
        rows.append(row)
        print(f'[{item["id"]}] base={row["base"]["response"][:30]!r} tuned={row["tuned"]["response"][:30]!r}')

    summary = {}
    for model in ("base", "tuned"):
        responses = [row[model]["response"] for row in rows]
        summary[model] = {
            "responses": len(rows),
            "style_marker_rate": sum(bool(row[model]["style_markers"]) for row in rows) / len(rows),
            "empty_response_rate": sum(not row[model]["response"] for row in rows) / len(rows),
            "mean_response_characters": statistics.mean(map(len, responses)),
            "exact_training_output_matches": sum(response in training_outputs for response in responses) if training_outputs else None,
        }
    payload = {"base_model": args.base_model, "model_path": args.model_path,
               "system_prompt": SYSTEM_PROMPT, "decoding": {"temperature": 0.0, "seed": 42, "max_tokens": 160},
               "summary": summary, "rows": rows,
               "limitations": "风格词命中率只是描述性代理指标；角色一致性、事实性和回答质量仍需人工盲评。"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# 基座模型与 LoRA 模型固定集对比", "", "自动指标仅作辅助，请逐条人工复核。", ""]
    for row in rows:
        md += [f'## {row["id"]} · {row["category"]}', "", f'**用户：** {row["prompt"]}', "",
               f'**基座：** {row["base"]["response"]}', "", f'**LoRA：** {row["tuned"]["response"]}', ""]
    args.output.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

