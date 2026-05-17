> [!note]
> 本项目是基于原 [PAP-FZS3D](https://github.com/heshuting555/PAP-FZS3D) 实现的。

---

# 1. 项目概述

本项目实现了面向三维点云语义分割的小样本学习（Few-shot Learning）与零样本学习（Zero-shot Learning）方法。核心思想基于**原型网络（Prototypical Network）**，有以下关键模块：

1. **原 PAP-FZS3D 模块**：
	- **原型自适应与投影（PAP）**：通过查询引导的原型对齐（Query-Guided Prototype Alignment, QGPA）Transformer，动态调整支持集原型以适配查询点云的特征分布。
	- **原型自重建正则化（Self-Reconstruction）**：利用支持集自身对原型进行重建监督，增强原型的判别能力。
	- **零样本原型生成（PAP-FZ）**：基于 GMMN（Gaussian Mixture Model Network）生成器，从类别词向量（GloVe）中合成未见类别的视觉原型，实现对未见过类别的零样本分割。

2. **本项目中新增模块**：
	- **高度分层原型（Height-Stratified Prototype）**（针对 FOR-Instance 森林数据集）：按垂直方向将点云分层构建多高度原型，捕捉森林场景中不同树种在垂直结构上的差异。
	- **类别平衡损失（Class-Balanced Loss）**（针对 FOR-Instance 数据集）：在 episode 内对类别进行加权，缓解森林数据中严重的类别不平衡问题。

该代码库在原有论文实现基础上进行了扩展，新增了对 **FOR-Instance 森林激光雷达点云数据集** 的支持，包括数据预处理管线、高度分层原型匹配、类别平衡损失以及全 LiDAR 属性（xyz + intensity + 归一化坐标）支持。

---

# 2. 分支说明

当前项目有以下几个分支：

```bash
  exp/balanced-loss                     实现类别平衡损失机制
  exp/full-lidar-attribs                启用完整 LiDAR 属性支持
  exp/hproto-ablation                   HProto λ 消融实验
  for-instance/adapted                  仅在原始的 PAP-FZS3D 项目上对 FOR-Instance 数据集进行了适配
  for-instance/innovation               加入了创新点：“高度残差分层原型”，默认 λ=0.2
  main                                  原 PAP-FZS3D 项目，未做任何修改
```

请切换对应分支查看具体内容。

---

# 3. 环境准备

本项目基于 [PAP-FZS3D](https://github.com/heshuting555/PAP-FZS3D) 构建，而 PAP-FZS3D 又基于 [attMPTI](https://github.com/Na-Z/attMPTI) 构建，环境配置和数据准备请参考其说明。主要依赖：

- GPU Nvidia 5050
- Python 3.11.5
- PyTorch 2.11.0+cu128
- CUDA 12.8
- NumPy, h5py, scipy
- laspy（处理 FOR-Instance 的 .las 文件）
- pandas（FOR-Instance 元数据读取）
- gensim（下载 GloVe 词向量）
- tensorboard（训练可视化）

对于 FOR-Instance 数据集，额外安装：

```bash
pip install laspy pandas tqdm
```

---

# 4. 致谢

本项目代码基于以下开源工作进行构建，在此表示感谢：

- [DGCNN (PyTorch)](https://github.com/WangYueFt/dgcnn/tree/master/pytorch) — 动态图卷积神经网络
- [PAP-FZS3D](https://github.com/heshuting555/PAP-FZS3D) — 注意力多原型直推推理（小样本3D点云分割基线）
- [FOR-Instance](https://doi.org/10.1016/j.isprsjprs.2025.01.015) — 森林激光雷达点云实例分割数据集

---

# 5. 许可证

本项目基于 MIT License 开源。详见 [LICENSE](LICENSE) 文件。
