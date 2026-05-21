# PAP-FZS3D for FOR-Instance

基于 PAP-FZS3D 的少样本三维点云语义分割，适配 [FOR-Instance](https://arxiv.org/abs/2309.13979) 无人机 LiDAR 单木点云数据集。

## 概述

本项目将 PAP-FZS3D（Prototypical Alignment and Prototype-aware Feature Generation for Zero-shot 3D Semantic Segmentation）方法适配到 FOR-Instance 基准数据集上，实现森林点云的**少样本语义分割**。

模型以 DGCNN 作为点云编码器，结合原型网络（Prototypical Network）进行 episodic few-shot 学习，并支持注意力机制、Transformer 语义增强、原型自重建和对齐等模块来提升分割精度。

> [!NOTE]
> 当前适配的是 FOR-Instance 的**语义少样本分割**任务，目标类别为 `terrain`（地面）、`low_vegetation`（低矮植被）、`stem`（树干）、`live_branches`（活枝）和 `woody_branches`（枯枝）。该任务不同于官方的单木实例分割评估。

## 目录结构

```
.
├── main.py                  # 统一入口，支持 pretain/prototrain/protoeval/mptitrain 等阶段
├── models/                  # 模型定义
│   ├── dgcnn.py             # DGCNN 点云编码器
│   ├── protonet.py          # 原型网络（基线）
│   ├── protonet_QGPA.py     # PAP-FZS3D 完整模型（含 Transformer、对齐、自重建）
│   ├── proto_learner.py     # 原型网络训练封装
│   ├── proto_learner_FZ.py  # 零样本学习训练封装
│   ├── attention.py         # 注意力模块
│   ├── gmmn.py              # 生成器模块
│   └── mpti.py / mpti_learner.py  # MPTI 相关模型
├── dataloaders/             # 数据加载
│   ├── loader.py            # 通用 DataLoader 与 episode 构造
│   ├── forinstance.py       # FOR-Instance 数据集加载器
│   ├── s3dis.py             # S3DIS 数据集加载器
│   └── scannet.py           # ScanNet 数据集加载器
├── preprocess/              # 数据预处理脚本
│   ├── collect_forinstance_data.py  # 从 .las 提取并重映射标签
│   ├── forinstance2blocks.py        # 将场景切分为固定大小的 block
│   └── ...
├── runs/                    # 训练/评估流程
│   ├── pre_train.py         # 预训练点云编码器
│   ├── proto_train.py       # PAP 少样本训练
│   ├── eval.py              # 少样本评估
│   └── fine_tune.py         # 微调
├── scripts/                 # 一键运行脚本
│   ├── preprocess_forinstance.sh   # 数据预处理
│   ├── pretrain_forinstance.sh     # 预训练编码器
│   ├── train_PAP_forinstance.sh    # 训练 PAP 少样本模型
│   └── eval_PAP_forinstance.sh     # 评估模型
├── utils/                   # 工具函数
├── docs/                    # 文档
└── datasets/                # 预处理后的数据与类别文件
    └── FORInstance/meta/forinstance_classnames.txt
```

## 环境要求

- Python 3.11+
- PyTorch 2.x（CUDA 支持）
- 依赖包：`laspy`, `pandas`, `tqdm`, `h5py`, `tensorboard`

安装 PyTorch：

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

安装其他依赖：

```bash
pip install laspy pandas tqdm h5py tensorboard
```

## 快速开始

### 1. 准备原始数据

将 FOR-Instance 数据集放置于 `data/FORinstance_dataset/` 目录下，结构如下：

```
data/FORinstance_dataset/
├── CULS/
├── NIBIO/
├── RMIT/
├── SCION/
├── TUWIEN/
└── data_split_metadata.csv
```

> 数据集详情见 [数据集说明](docs/01_数据集说明.md)。

### 2. 数据预处理

将 `.las` 原始点云转换为模型可读取的 `.npy` block 格式：

```bash
bash scripts/preprocess_forinstance.sh
```

该脚本执行两步：
- **标签映射**：从原始 `classification` 字段重映射为 5 个连续类别（0-4）
- **滑窗切块**：按 10m 窗口、5m 步长切分为 block，每块最少 1024 个点

输出位于 `datasets/FORInstance/blocks_dev_bs10.0_s5.0/`。

### 3. 预训练编码器

在 FOR-Instance 的基础类上预训练 DGCNN 点云特征提取器：

```bash
bash scripts/pretrain_forinstance.sh
```

关键配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| `pc_attribs` | `xyzIXYZ` | 输入特征：坐标 + 强度 + 归一化坐标 |
| `pc_npts` | 2048 | 每块采样点数 |
| `n_iters` | 50 | 训练 epoch 数 |
| `batch_size` | 16 | 批次大小 |

模型权重保存至 `logs/log_forinstance/log_pretrain_forinstance_S0/checkpoint.tar`。

### 4. 训练 PAP 少样本模型

```bash
bash scripts/train_PAP_forinstance.sh
```

关键配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_way` | 2 | 每 episode 2 个类别 |
| `k_shot` | 1 | 每类 1 个支持样本 |
| `n_iters` | 25000 | 训练迭代次数 |
| `use_transformer` | ✓ | 启用 Transformer 语义增强 |
| `use_supervise_prototype` | ✓ | 启用原型自重建 |
| `use_attention` | ✓ | 启用注意力学习器 |
| `use_align` | ✓ | 启用特征对齐 |
| `use_high_dgcnn` | ✓ | 使用高阶 DGCNN |

### 5. 评估模型

```bash
bash scripts/eval_PAP_forinstance.sh
```

评估日志输出至 `logs/log_forinstance_PAP/log_proto_forinstance_S0_N2_K1_Att1/log_protoeval.txt`。

## 折叠划分（Cross-Validation Folds）

FOR-Instance 适配使用 2-fold 交叉验证：

| Fold | 测试类 | 训练类 |
|------|--------|--------|
| **Fold 0** | terrain (0), stem (2) | low_vegetation (1), live_branches (3), woody_branches (4) |
| **Fold 1** | live_branches (3), woody_branches (4) | terrain (0), low_vegetation (1), stem (2) |

Fold 0 测试几何差异较大的类别（地面 vs 树干），Fold 1 测试结构相似的类别（活枝 vs 枯枝），后者评估难度更高。

## 模型架构

核心模型 PAP-FZS3D（`ProtoNetAlignQGPASR`）由以下模块组成：

- **DGCNN 编码器**：基于 EdgeConv 的点云特征提取，支持高阶 DGCNN
- **原型网络**：计算 support set 的类别原型，通过余弦/欧氏距离对 query 点进行分类
- **注意力学习器**（`AttLearner`）：学习 query 与 support 特征之间的交叉注意力
- **Transformer 语义增强**：利用类别语义嵌入（word2vec）增强原型表示
- **原型自重建**（Supervised Prototype）：通过重建 loss 约束原型质量
- **特征对齐**（Alignment）：对齐 support 和 query 的特征分布
- **线性投影**：将编码器输出映射到任务相关的特征空间

## 数据格式

预处理后的 FOR-Instance block 为 float32 的 `.npy` 文件，形状为 `[N, 5]`：

| 列 | 含义 |
|----|------|
| 0-2 | x, y, z（block 内局部坐标，单位：米） |
| 3 | intensity（归一化到 [0, 1]） |
| 4 | label（0-4 整数标签） |

## 日志与可视化

训练日志和 checkpoint 保存在 `logs/` 目录下：

```
logs/
├── log_forinstance/                     # 预训练
│   └── log_pretrain_forinstance_S0/
│       ├── checkpoint.tar               # 模型权重
│       ├── events.out.tfevents.*        # TensorBoard 日志
│       └── log_pretrain.txt             # 文本日志
└── log_forinstance_PAP/                 # PAP 少样本训练
    └── log_proto_forinstance_S0_N2_K1_Att1/
        ├── checkpoint.tar
        ├── log_prototrain.txt
        └── log_protoeval.txt
```

启动 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir=logs
```

## 引用

- FOR-Instance 数据集：Puliti S, Pearse G, Surovy P, et al. *FOR-instance: a UAV laser scanning benchmark dataset for semantic and instance segmentation of individual trees*. arXiv, 2023.
- PAP-FZS3D：原型对齐与零样本三维语义分割方法