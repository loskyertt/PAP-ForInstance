# PAP-FZS3D 学习教程

> **Prototype Adaption and Projection for Few- and Zero-shot 3D Point Cloud Semantic Segmentation**
>
> 面向少样本与零样本3D点云语义分割的原型自适应与投影方法

---

## 教程概述

本教程系统地将 PAP-FZS3D 项目转化为一门循序渐进的学习课程。无论你是刚刚接触 3D 点云深度学习的新手，还是已有一定基础的研究者，都可以通过本教程逐步掌握该项目的核心概念、架构设计与工程实现。

PAP-FZS3D 是发表在 **IEEE Transactions on Image Processing (2023)** 上的一个学术研究项目，实现了当前领先的少样本（Few-Shot）和零样本（Zero-Shot）3D 点云语义分割方法。

---

## 你将要学到什么

- **3D 点云深度学习基础**：点云表示、特征提取、DGCNN 架构
- **少样本学习（Few-Shot Learning）**：度量学习、原型网络、Episode 训练
- **注意力机制**：自注意力、交叉注意力、QGPA 原型自适应模块
- **零样本学习（Zero-Shot Learning）**：GMMN 生成器、词嵌入、分布匹配
- **完整的训练与评估流程**：预处理、预训练、元训练、测试评估
- **工程实践**：代码阅读、实验复现、模块扩展

---

## 教程结构

```
Tutorial/
├── README.md              # 本文件：教程总览
├── Learning_Path.md       # 学习路线图：适合人群、前置知识、学习阶段
├── Course_Outline.md      # 课程大纲：30节课的内容概要
├── Lesson_01_*.md         # 第1课
├── Lesson_02_*.md         # 第2课
├── ...
└── Lesson_30_*.md         # 第30课
```

---

## 快速开始

1. **先阅读** → `Learning_Path.md`：了解是否适合你
2. **再浏览** → `Course_Outline.md`：了解课程全貌
3. **开始学习** → 从 `Lesson_01` 开始，按顺序推进

---

## 项目信息

- **论文**：[arXiv 2305.14335](https://arxiv.org/pdf/2305.14335.pdf)
- **许可**：MIT License
- **基于**：DGCNN (PyTorch) + attMPTI
