# 第2课：环境搭建与依赖安装

## 1. 本节学习目标

- 搭建可运行 PAP-FZS3D 的 Python 环境
- 安装 PyTorch 及其生态系统
- 安装项目所需的第三方依赖
- 验证环境可用性

---

## 2. 依赖关系全景

PAP-FZS3D 基于 PyTorch 构建，需要以下依赖：

```
核心框架:     PyTorch (>= 1.0)
数据处理:     NumPy, h5py, pickle
3D 操作:      transforms3d, faiss, torch_cluster
可视化:       TensorBoard
语义嵌入:     Gensim (GloVe)
```

> **注意**：项目没有提供 `requirements.txt`，以下基于源码 import 分析得出。

---

## 3. 环境搭建步骤

### 3.1 创建虚拟环境 (推荐 Conda)

```bash
# 创建 Python 3.8 环境
conda create -n pap-fzs3d python=3.8 -y
conda activate pap-fzs3d
```

### 3.2 安装 PyTorch

根据 CUDA 版本选择：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (仅用于测试)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3.3 安装核心数据处理依赖

```bash
pip install numpy h5py scipy
```

### 3.4 安装 3D 相关依赖

```bash
# 3D 变换与数据增强
pip install transforms3d

# FPS 采样 (用于 MPTI 和部分预处理)
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Faiss 近邻搜索 (MPTI 使用)
pip install faiss-cpu   # 或 faiss-gpu
```

### 3.5 安装其他依赖

```bash
# TensorBoard 日志
pip install tensorboard

# GloVe 词嵌入 (零样本使用)
pip install gensim

# YAML 配置读取
pip install pyyaml
```

### 3.6 安装 GloVe 词向量 (零样本需要)

```python
# 在 Python 中运行：
import gensim.downloader as api
api.load("glove-wiki-gigaword-300")  # 约 1GB, 仅首次需下载
```

---

## 4. 验证安装

运行以下验证脚本：

```python
# verify_env.py
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import numpy as np
print(f"NumPy: {np.__version__}")

import h5py
print(f"h5py: {h5py.__version__}")

try:
    import transforms3d
    print("transforms3d: OK")
except ImportError:
    print("transforms3d: NOT FOUND (preprocessing only)")

try:
    import faiss
    print(f"faiss: OK (version {faiss.__version__})")
except ImportError:
    print("faiss: NOT FOUND (MPTI only)")

try:
    from torch_cluster import fps
    print("torch_cluster: OK")
except ImportError:
    print("torch_cluster: NOT FOUND (preprocessing only)")

try:
    import gensim
    print(f"gensim: {gensim.__version__}")
except ImportError:
    print("gensim: NOT FOUND (zero-shot only)")

try:
    from torch.utils.tensorboard import SummaryWriter
    print("tensorboard: OK")
except ImportError:
    print("tensorboard: NOT FOUND (logging only)")

print("\n✅ Minimal environment OK (PyTorch + NumPy + h5py)")
print("   Core modules can run. Optional deps may be needed for specific features.")
```

---

## 5. 数据集准备

项目需要 S3DIS 或 ScanNet 数据集：

```bash
# 预期目录结构
datasets/
├── S3DIS/
│   ├── data/          # 原始 S3DIS 数据
│   │   ├── Area_1/
│   │   ├── Area_2/
│   │   └── ...
│   └── meta/
│       └── class_names.txt
└── ScanNet/
    ├── data/          # 原始 ScanNet 数据
    └── meta/
        └── class_names.txt
```

预处理后会在 datasets 目录下生成 `.npy` 文件和 `.pkl` 索引。

---

## 6. 常见问题

### Q1: `ModuleNotFoundError: No module named 'torch'`

A: PyTorch 未正确安装。检查是否激活了正确的 conda 环境，或重新安装。

### Q2: `CUDA out of memory`

A: GPURAM 不足。尝试减小 batch size，或设置 `--pc_npts 1024`（默认 2048）。

### Q3: `h5py` 报错 "Unable to open file"

A: 数据集尚未预处理。需要先运行 `preprocess/` 下的脚本。

### Q4: `transforms3d` 中的 `axangle2aff` 找不到

A: 这是版本问题。新版本中此函数位置变了，尝试 `pip install transforms3d==0.3.1`。

### Q5: MPTI 相关的 faiss 导入错误

A: MPTI 是可选功能。如果不需要，可以忽略此错误（默认使用 ProtoNet，不依赖 faiss）。

---

## 7. 快速验证：运行 main.py

```bash
# 确认环境搭建成功
python main.py --help

# 期望输出类似:
# usage: main.py [-h] --phase {pretrain,prototrain,protoeval,...}
#                [--dataset {S3DIS,ScanNet}] [--cvfold FOLD]
#                ...
```

---

## 8. 本节总结

| 步骤 | 关键命令 |
|---|---|
| 创建环境 | `conda create -n pap-fzs3d python=3.8` |
| 安装 PyTorch | `pip install torch torchvision torchaudio` |
| 安装依赖 | `pip install numpy h5py transforms3d` |
| 验证环境 | 运行 verify_env.py |
| 确认可用 | `python main.py --help` |

---

## 9. 课后练习

1. 成功运行 `verify_env.py`，确保所有核心依赖已安装
2. 了解你的 GPU 显存大小，评估能否运行默认配置（约需 4-8 GB）
3. 尝试安装一个缺失的可选依赖（如 `faiss` 或 `gensim`）
4. 如果有 S3DIS 数据，尝试运行 `preprocess/collect_s3dis_data.py`（需要先修改其中的数据路径）
