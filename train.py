"""使用 PyTrio 对 Qwen3.5-4B 做 Chat-嬛嬛 LoRA SFT。"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = "现在你要扮演皇帝身边的女人——甄嬛。请保持角色口吻，并诚实承认未知信息。"


def load_examples(path: Path) -> list[dict[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()
        output = str(row.get("output", "")).strip()
        if instruction and output:
            user = instruction if not input_text else f"{instruction}\n{input_text}"
            examples.append({"user": user, "assistant": output})
    if not examples:
        raise ValueError(f"没有有效训练样本: {path}")
    return examples


def build_datum(example, tokenizer, max_length: int):
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": example["user"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(example["assistant"], add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer_ids.append(tokenizer.eos_token_id)

    # 优先保留完整答案及 EOS；提示过长时从左侧截断提示。
    answer_ids = answer_ids[: max_length - 1]
    if not answer_ids:
        raise ValueError("answer 在 tokenization 后为空")
    prompt_ids = prompt_ids[-(max_length - len(answer_ids)) :]
    tokens = prompt_ids + answer_ids
    weights = [0.0] * len(prompt_ids) + [1.0] * len(answer_ids)
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": np.asarray(tokens[1:], dtype=np.int32),
            "weights": np.asarray(weights[1:], dtype=np.float32),
        },
    )


def batch_loss(result, batch) -> float:
    loss_sum, token_count = loss_totals(result, batch)
    return loss_sum / token_count


def loss_totals(result, batch) -> tuple[float, float]:
    """返回加权负对数似然总和与有效 token 数。"""
    logprobs = np.concatenate([np.asarray(x["logprobs"]) for x in result.loss_fn_outputs])
    weights = np.concatenate([np.asarray(x.loss_fn_inputs["weights"]) for x in batch])
    token_count = float(weights.sum())
    if token_count <= 0:
        raise ValueError("batch 中没有有效的 assistant token")
    return float(-np.dot(logprobs, weights)), token_count


def evaluate_loss(trainer, datums, batch_size: int) -> float:
    """在验证集上做前向计算，按有效 token 汇总交叉熵。"""
    total_loss = 0.0
    total_tokens = 0.0
    for start in tqdm(range(0, len(datums), batch_size), desc="Validation", unit="batch", leave=False):
        batch = datums[start:start + batch_size]
        result = trainer.forward(batch, "cross_entropy").result()
        loss_sum, token_count = loss_totals(result, batch)
        total_loss += loss_sum
        total_tokens += token_count
    if total_tokens <= 0:
        raise ValueError("验证集中没有有效 token")
    return total_loss / total_tokens


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/train.json")
    parser.add_argument("--valid-data", type=Path, default=ROOT / "data/valid.json")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/latest")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 2:
        parser.error("epochs、batch-size 必须为正数，max-length 至少为 2")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"运行目录已有日志，请换一个 --run-dir: {metrics_path}")
    started = time.time()
    examples = load_examples(args.data)
    valid_examples = load_examples(args.valid_data)
    service = trio.ServiceClient()
    trainer = service.create_lora_training_client(
        base_model=args.base_model, rank=args.rank, seed=args.seed,
        train_mlp=True, train_attn=True, train_unembed=True,
    )
    tokenizer = trainer.get_tokenizer()
    datums = [build_datum(row, tokenizer, args.max_length) for row in examples]
    valid_datums = [build_datum(row, tokenizer, args.max_length) for row in valid_examples]
    steps_per_epoch = (len(datums) + args.batch_size - 1) // args.batch_size
    config = {**vars(args), "data": str(args.data), "valid_data": str(args.valid_data),
              "run_dir": str(args.run_dir), "num_examples": len(datums),
              "num_valid_examples": len(valid_datums), "steps_per_epoch": steps_per_epoch,
              "system_prompt": SYSTEM_PROMPT, "train_mlp": True,
              "train_attn": True, "train_unembed": True}
    (args.run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    run = swanlab.init(project="chat-huanhuan-lora", experiment_name=f"qwen3.5-4b-r{args.rank}", config=config)
    rng = random.Random(args.seed)
    step = 0
    best_val_loss = math.inf
    best_checkpoint = None
    history = []
    name = f"chat-huanhuan-qwen3.5-4b-r{args.rank}-{int(started)}"
    with tqdm(total=args.epochs * steps_per_epoch, desc="LoRA SFT", unit="batch") as bar:
        for epoch in range(args.epochs):
            epoch_loss_sum = 0.0
            epoch_tokens = 0.0
            order = list(range(len(datums)))
            rng.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                batch = [datums[i] for i in order[start:start + args.batch_size]]
                result = trainer.forward_backward(batch, "cross_entropy").result()
                trainer.optim_step(trio.AdamParams(learning_rate=args.learning_rate)).result()
                loss_sum, token_count = loss_totals(result, batch)
                loss = loss_sum / token_count
                epoch_loss_sum += loss_sum
                epoch_tokens += token_count
                record = {"type": "train_step", "step": step, "epoch": epoch + 1,
                          "loss": loss, "assistant_tokens": int(token_count),
                          "elapsed_seconds": time.time() - started}
                append_jsonl(metrics_path, record)
                swanlab.log({"train/loss": loss, "epoch": epoch + 1}, step=step)
                bar.update(1)
                bar.set_postfix(loss=f"{loss:.4f}", epoch=f"{epoch + 1}/{args.epochs}")
                step += 1

            train_loss = epoch_loss_sum / epoch_tokens
            val_loss = evaluate_loss(trainer, valid_datums, args.batch_size)
            if not math.isfinite(val_loss):
                raise ValueError(f"验证 loss 非有限值: {val_loss}")
            epoch_record = {"type": "epoch", "epoch": epoch + 1,
                            "train_loss": train_loss, "val_loss": val_loss,
                            "elapsed_seconds": time.time() - started}
            append_jsonl(metrics_path, epoch_record)
            history.append(epoch_record)
            swanlab.log({"train/epoch_loss": train_loss, "valid/loss": val_loss}, step=step)
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_name = f"{name}-best-epoch-{epoch + 1}"
                sampler = trainer.save_weights_for_sampler(name=checkpoint_name).result()
                state = trainer.save_state(name=f"{checkpoint_name}-train", overwrite=True).result()
                best_checkpoint = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "sampler_model_path": sampler.path,
                    "sampler_size_bytes": getattr(sampler, "size", None),
                    "train_state_path": state.path,
                }
                (args.run_dir / "best_checkpoint.json").write_text(
                    json.dumps(best_checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            (args.run_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    sampler_weights = trainer.save_weights_for_sampler(name=f"{name}-final").result()
    train_state = trainer.save_state(name=f"{name}-final-train", overwrite=True).result()
    artifact = {"base_model": args.base_model, "sampler_model_path": sampler_weights.path,
                "sampler_size_bytes": getattr(sampler_weights, "size", None),
                "train_state_path": train_state.path, "completed_steps": step,
                "elapsed_seconds": time.time() - started,
                "best_checkpoint": best_checkpoint}
    (args.run_dir / "artifacts.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    run.finish()
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

