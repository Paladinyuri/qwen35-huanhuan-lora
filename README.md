# Qwen3.5-4B 角色对话 LoRA 微调

这是一个学习型 LoRA 项目：使用 PyTrio 在 Qwen3.5-4B 上训练 Chat-嬛嬛角色对话数据，并用完全相同的 system prompt 和解码参数比较基座模型与 LoRA 模型。

目前已完成一次原始脚本训练，但当时没有保留本地 loss 日志、独立测试集结果和结构化对照输出。因此仓库不把“训练完成”写成已经验证有效；新增流程用于补齐可复现证据。原始训练脚本保留为 `qwen 微调.py`，后续实验以 `train.py` 和 `evaluate.py` 为准。

## 数据

`huanhuan.json` 含 3,729 条 `instruction/input/output` 记录，来源于公开的 Chat-嬛嬛数据。原始数据全部 `input` 为空。初步审计发现 112 条完全重复记录、238 个重复用户提示；直接随机按记录切分会造成相同提示跨集合泄漏。

`scripts/prepare_data.py` 先删除完全重复记录，再按“用户提示”分组，通过稳定哈希切分为训练/验证/测试集。这样同一个提示的多个答案只会出现在同一集合。

```powershell
python scripts/prepare_data.py
```

生成的 `data/stats.json` 会记录原始文件 SHA-256、去重数量、集合大小和提示重叠检查。`data/` 不提交 Git，以避免重复分发数据；数据来源及许可应以原项目为准：<https://github.com/KMnO4-zx/huanhuan-chat>。

## 环境

需要 Python 3.10+、PyTrio 账号及网络连接。训练和推理由 PyTrio 服务执行，本机无需保存 Qwen3.5-4B 权重。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
trio login
```

## 训练

```powershell
python train.py --epochs 2 --batch-size 16 --rank 32 --learning-rate 1e-4 --run-dir results/run-r32
```

默认配置：Qwen/Qwen3.5-4B、LoRA rank 32、最大长度 1,024、随机种子 42。PyTrio 的 attention、MLP 和 unembedding LoRA 开关均显式设为开启。训练只对 assistant 回复及 EOS 计算 loss；system 和 user token 的权重为 0。提示过长时优先保留答案与 EOS。

训练会保存：

- `config.json`：实际参数与 system prompt；
- `train.jsonl`：每一步 loss 和耗时；
- `artifacts.json`：sampler 权重路径和续训 checkpoint 路径；
- SwanLab 在线曲线。

注意：当前训练脚本只记录训练 loss。验证集用于后续独立评估；不能根据测试集反复调参。

## 固定评估

从 `artifacts.json` 复制 `sampler_model_path`：

```powershell
python evaluate.py --model-path "trio://你的权重路径" --output results/evaluation.json
```

评估集含角色身份、宫廷对话、未见措辞、域外任务和抗提示覆盖等 15 个固定问题。基座与 LoRA 模型使用同一个 system prompt、temperature=0、seed=42、max_tokens=160。输出同时保存为 JSON 和 Markdown，便于逐条比较。

脚本只自动报告风格词命中率和空回复率。风格词命中不等于回答更好；最终需要人工盲评以下四项：角色一致性、语言自然度、问题相关性、事实/诚实性。每项可用 1–5 分，由不知道模型身份的人随机顺序评分。

## 权重备份

当前 sampler checkpoint 信息记录在 `results/checkpoint.json`。直接评估使用其中的 `trio://...` 路径，无需先下载。若要把 PEFT adapter 备份到本地：

```powershell
python scripts/download_adapter.py ckpt_dgwydrq0t4zg
```

归档会写入 `weights/`，该目录不会提交 Git。PyTrio 官方导出格式是 `.zip`；原先示例中的 `.tar` 只是文件名后缀，并不会改变真实归档格式。

## 项目结论应怎样写

在新的固定评估真正运行前，只能写：

> 基于 PyTrio 完成 Qwen3.5-4B 的 LoRA SFT 流程，实现聊天模板构造、assistant-only loss、本地训练日志与权重保存，并建立基座/LoRA 固定集对照评估。

本次保存的 sampler 权重已经完成固定评估，结果见 [`results/EVALUATION_REPORT.md`](results/EVALUATION_REPORT.md)。最新完整运行中，LoRA 让回答从平均 205.3 字缩短至 18.5 字，并更接近简短剧本台词；但简单风格词命中率从 86.7% 降至 53.3%，15 条 LoRA 回答中有 3 条与训练答案完全一致。它说明模型行为确实发生变化，也提示存在记忆倾向，不能直接表述为整体质量提升。

后续如果重跑新数据划分，应补充真实训练 loss、权重大小和人工盲评。不能只展示几个挑选过的好例子，也不能把训练 loss 下降直接表述为角色能力提升。

## 已知限制

- 数据来自电视剧台词整理，版权与再分发边界应遵循上游项目说明；
- 训练语料小且重复较多，容易记忆常见台词；
- 固定集只有 15 个提示，适合作为项目检查，不足以支持普遍结论；
- PyTrio 封装了具体 LoRA 模块实现，本项目不能据此声称手动实现了 PEFT/LoRA；
- 原始训练没有保留本地 loss 日志，因此无法补画可信的训练曲线；现有 sampler 权重只能用于推理评估，不能恢复原训练过程。

## 检查

```powershell
python -m unittest discover -s tests -v
python -m py_compile train.py evaluate.py scripts/prepare_data.py scripts/download_adapter.py
```

