# Qwen3.5-4B 角色对话 LoRA 微调

这个项目用 Chat-嬛嬛数据对 Qwen3.5-4B 做 LoRA 微调，训练和推理通过 PyTrio 完成。最开始我只跑通了训练，后来又补上数据去重、训练记录和基座模型/微调模型的对照评估。

原始代码保留在 `qwen 微调.py`，整理后的训练和评估入口分别是 `train.py`、`evaluate.py`。

## 数据处理

原始 `huanhuan.json` 有 3,729 条数据，其中 112 条完全重复，还有一些问题相同、回答不同的记录。为了避免同一个问题同时出现在训练集和测试集里，`scripts/prepare_data.py` 会先去重，再按用户问题分组切分数据。

```powershell
python scripts/prepare_data.py
```

使用 seed 42 得到的划分如下：

| 集合 | 数量 |
| --- | ---: |
| 训练集 | 2,887 |
| 验证集 | 371 |
| 测试集 | 359 |

三个集合之间没有重复的用户问题。数据文件不放在仓库中，可以从 [Chat-嬛嬛](https://github.com/KMnO4-zx/huanhuan-chat) 获取。

## 环境配置

需要 Python 3.10+ 和 PyTrio 账号。模型在 PyTrio 服务上运行，本地不用下载完整的 Qwen3.5-4B 权重。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
trio login
```

## 训练

```powershell
python train.py --data data/train.json --valid-data data/valid.json --epochs 2 --batch-size 16 --rank 32 --learning-rate 1e-4 --run-dir results/run-r32-clean
```

这次实验使用 LoRA rank 32、最大长度 1,024、训练 2 个 epoch。loss 只计算 assistant 的回答部分，不计算 system 和 user 消息。每个 epoch 结束后，脚本会用 `valid.json` 做一次不累积梯度的前向计算，并在验证 loss 降低时保存新的最佳 checkpoint。

训练目录包含：

- `metrics.jsonl`：每一步训练 loss 和每个 epoch 的训练/验证 loss；
- `history.json`：用于画曲线的 epoch 汇总；
- `best_checkpoint.json`：验证 loss 最低的 checkpoint；
- `artifacts.json`：最终 checkpoint，以及最佳 checkpoint 的引用。

同一个 `--run-dir` 不会被静默覆盖。训练完成后可以生成 loss 曲线：

```powershell
python scripts/plot_history.py results/run-r32-clean/history.json
```

## 评估

把训练产物中的 `sampler_model_path` 传给评估脚本：

```powershell
python evaluate.py --model-path "trio://你的权重路径" --output results/evaluation.json
```

评估集固定为 15 个问题，涵盖角色身份、宫廷对话、普通任务和提示覆盖等情况。基座模型与 LoRA 模型使用相同的 system prompt 和解码参数，逐条结果见 [`results/EVALUATION_REPORT.md`](results/EVALUATION_REPORT.md)。

在目前保存的这次实验中：

- 基座模型平均回复 199.4 字，LoRA 模型平均回复 29.5 字；
- LoRA 的回答更接近简短的剧本台词；
- 15 条 LoRA 回答中有 1 条与干净训练集答案完全相同；
- 简单风格词命中率由 86.7% 降至 66.7%；
- 第 2 轮训练 loss 继续下降，但验证 loss 上升，因此最终选择 epoch 1。

这些结果说明微调明显改变了模型的回复方式，但还不能说明整体质量一定更好。回复变短的同时也出现了训练语料记忆，后续还需要扩大测试集并做人工盲评。

## 权重

本次干净重训的最佳权重路径、续训状态和最终 epoch 权重都记录在 `results/checkpoint.json`。最佳权重由验证 loss 选择，可直接将其中的 `trio://...` 路径传给 `evaluate.py`。如需本地备份，可以在 PyTrio 控制台取得该归档的 checkpoint ID，再交给 `scripts/download_adapter.py`；下载文件会放在 `weights/`，不会提交到 Git。

## 项目结构

```text
configs/                 固定评估问题
results/                 本次实验结果和说明
scripts/prepare_data.py  数据去重与划分
scripts/download_adapter.py
scripts/plot_history.py  绘制训练/验证 loss 曲线
train.py                 训练入口
evaluate.py              基座/LoRA 对照评估
```

## 运行检查

```powershell
python -m unittest discover -s tests -v
python -m py_compile train.py evaluate.py scripts/prepare_data.py scripts/download_adapter.py
```

目前的主要不足是原始训练没有保存完整 loss 曲线，固定评估也只有 15 个问题。这一版仓库主要用于记录一次完整的 LoRA 实验流程和已经观察到的结果。
