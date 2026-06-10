# OpenMythos-hong

OpenMythos-hong 是一个面向本地与集群实验的 MoE 模型工程初始化仓库。本仓库保留三档同构模型入口：tiny 调试模型、本地 CUDA 小 MoE、集群 MoE。三档模型使用同一套任务格式、配置方式、训练入口和服务接口，仅模型规模与运行资源不同。

> 当前仓库是“初始化版”。不包含用户个人训练产物、不包含生成数据集、不包含评测报告、不包含 checkpoint 权重文件。公开预训练模型权重需要在目标机器上离线下载到 `models/` 目录。

## 三个模型槽位

| 槽位 | 默认模型 | 目标设备 | 用途 |
|---|---|---|---|
| tiny | 项目内 tiny true-MoE | CPU / 小 GPU | 调试训练流程、验证服务、验证量化与缓存 |
| 本地 CUDA 小 MoE | `allenai/OLMoE-1B-7B-0125` | RTX 3050 Ti 4GB 级别 | 本地 4-bit/8-bit QLoRA 微调与本地服务 |
| 集群 MoE | `Qwen/Qwen1.5-MoE-A2.7B` | 多卡 / 集群 | 更大规模 MoE 微调与服务 |

## 统一任务风格

三档模型默认面向“代码运行输出预测”任务：用户输入源码，并询问“看看这个代码运行起来之后输出什么内容”，模型只输出程序 stdout，不解释。仓库只保留任务接口与训练/服务代码，不包含生成脚本或评测脚本。

## 目录说明

```text
open_mythos/                 核心模型与公开 MoE runtime
open_mythos/public_moe/      公开预训练 MoE 加载、tiny true-MoE、服务缓存、聊天格式
configs/                     三档模型配置
training/                    公开 MoE LoRA/QLoRA 训练入口
scripts/                     下载、服务、低显存预检、启动脚本
docs/                        中文详细说明文档
data/                        数据目录占位，不提交数据集
models/                      公开模型权重目录占位，不提交权重
checkpoints/                 checkpoint 目录占位，不提交训练产物
```

## 安装

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

本地 4-bit/8-bit 量化训练通常还需要：

```bash
pip install transformers accelerate peft bitsandbytes safetensors
```

## 下载公开模型

本地 CUDA 小 MoE：

```bash
python scripts/download_public_pretrained_moe.py \
  --model-id allenai/OLMoE-1B-7B-0125 \
  --local-dir models/OLMoE-1B-7B-0125
```

集群 MoE：

```bash
python scripts/download_public_pretrained_moe.py \
  --model-id Qwen/Qwen1.5-MoE-A2.7B \
  --local-dir models/Qwen1.5-MoE-A2.7B
```

下载完成后，生产/训练环境可以断网运行，配置中使用本地路径加载。

## 本地 3050Ti 4GB 预检

```bash
python scripts/check_3050ti_4g_ready.py \
  --config configs/family_small_moe_3050ti_4g.yaml \
  --require-cuda \
  --require-weights
```

## 启动本地 CUDA 小 MoE 训练

```bash
bash scripts/launch_small_moe_3050ti_4g_train.sh
```

该脚本默认使用低显存 QLoRA 配置。真实训练前需要准备自己的训练 JSONL，并在配置中填写路径。

## 启动服务

```bash
bash scripts/serve_small_moe_3050ti_4g.sh
```

服务提供 `/health`、`/generate`、`/v1/completions`、`/v1/chat/completions`。服务端支持 response cache 与跨请求前缀/KV cache，以减少重复上下文的重新计算。

## 更多中文说明

请阅读：

- `docs/项目总览.md`
- `docs/三模型架构与配置.md`
- `docs/训练与本地服务.md`
- `docs/低显存运行说明.md`
- `docs/缓存机制说明.md`
