# 课程大纲

> 30 节循序渐进课程，覆盖 PAP-FZS3D 项目的全部核心内容

---

## 第一阶段：项目概览与环境搭建

### [第1课：项目总览 — PAP-FZS3D 是什么](./Lesson_01_Project_Overview.md)
- **目标**：了解项目背景、论文贡献、系统架构
- **内容**：3D 点云语义分割问题定义、Few-Shot / Zero-Shot 概念、PAP-FZS3D 整体框架
- **实践**：浏览项目目录，运行 `python main.py --help`

### [第2课：环境搭建与依赖安装](./Lesson_02_Environment_Setup.md)
- **目标**：在本地搭建可运行项目环境
- **内容**：Python 环境、PyTorch 安装、h5py/faiss/transforms3d 等依赖、CUDA 配置
- **实践**：安装依赖，验证 `import torch; print(torch.cuda.is_available())`

### [第3课：项目代码架构总览](./Lesson_03_Codebase_Architecture.md)
- **目标**：建立对项目代码结构的整体认知
- **内容**：目录职责划分（models/dataloaders/runs/utils/preprocess）、main.py 入口、模块间调用关系
- **实践**：绘制项目模块依赖图

### [第4课：main.py 入口与命令行参数系统](./Lesson_04_Entry_Point.md)
- **目标**：理解主程序的 phase 路由和参数系统
- **内容**：argparse 参数定义、phase 路由机制（pretrain/prototrain/protoeval 等）、参数传递链路
- **实践**：追踪不同 phase 下的代码执行路径

---

## 第二阶段：数据流水线

### [第5课：点云数据表示与预处理](./Lesson_05_Point_Cloud_Data.md)
- **目标**：理解点云数据的表示方式和预处理流程
- **内容**：XYZ+RGB 格式、点采样（2048点）、归一化、数据增强（旋转/缩放/平移）、room2blocks 分块策略
- **实践**：运行 preprocess/ 下的脚本，可视化采样后的点云

### [第6课：S3DIS 与 ScanNet 数据集](./Lesson_06_Datasets.md)
- **目标**：理解两个数据集的类划分与 2-fold 交叉验证
- **内容**：S3DIS（13类，Area1-6）、ScanNet（21类）、base/novel 类划分、fold_0/fold_1 拆分策略
- **实践**：检查 `dataloaders/s3dis.py` 和 `dataloaders/scannet.py` 中的类映射

### [第7课：Episode 采样机制](./Lesson_07_Episode_Sampling.md)
- **目标**：理解 Few-Shot 学习的 Episode 构建过程
- **内容**：N-way K-shot 概念、MyDataset 类的 Episode 构建逻辑、support/query 划分、数据增强区别
- **实践**：跟踪 `dataloaders/loader.py:MyDataset.__getitem__` 的执行

### [第8课：HDF5 数据缓存与 Test Dataset](./Lesson_08_HDF5_Caching.md)
- **目标**：理解测试数据集的高效缓存策略
- **内容**：HDF5 格式优势、MyTestDataset 的预生成与去重机制、class2scans 映射、Pickle 索引
- **实践**：分析 `dataloaders/loader.py` 中 `MyTestDataset` 的完整实现

---

## 第三阶段：DGCNN 骨干网络

### [第9课：点云特征学习基础 — 从 PointNet 到 EdgeConv](./Lesson_09_Point_Cloud_Learning.md)
- **目标**：理解点云特征提取的核心挑战与演变
- **内容**：点云的无序性与置换不变性、PointNet 的对称函数思想、k-NN 局部图构建
- **实践**：手写简单的 k-NN 搜索实现

### [第10课：EdgeConv — 动态边卷积详解](./Lesson_10_EdgeConv.md)
- **目标**：深入理解 EdgeConv 算子的数学原理和实现
- **内容**：边特征定义（中心-邻域差值 + 中心）、动态图更新、Conv2d 处理
- **实践**：阅读 `models/dgcnn.py` 中的 `EdgeConv` 类实现

### [第11课：DGCNN 完整架构](./Lesson_11_DGCNN_Architecture.md)
- **目标**：掌握 DGCNN 分类/分割骨干的完整结构
- **内容**：多层 EdgeConv 堆叠、特征拼接、MLP 映射、DGCNN_cls / DGCNN_partseg / DGCNN_semseg 的区别
- **实践**：在 `dgcnn_new.py` 中打印每层输出的张量形状

### [第12课：DGCNNSeg 预训练流程](./Lesson_12_Pretraining.md)
- **目标**：理解 backbone 的预训练过程和意义
- **内容**：预训练数据集（base classes）、DGCNNSeg 分类头、交叉熵损失、checkpoint 保存与加载
- **实践**：运行 `pretrain_segmentor.sh`，观察 TensorBoard 曲线

---

## 第四阶段：原型网络与少样本学习

### [第13课：度量学习与原型网络原理](./Lesson_13_Metric_Learning.md)
- **目标**：理解少样本学习中度量学习的核心思想
- **内容**：嵌入空间概念、距离度量、原型（Prototype）的定义、原型网络 vs 匹配网络
- **实践**：用伪代码实现简单的原型计算

### [第14课：ProtoNet 模型的代码实现](./Lesson_14_ProtoNet_Implementation.md)
- **目标**：逐行阅读 ProtoNet 的 PyTorch 实现
- **内容**：forward() 的完整流程（编码 → 原型计算 → 相似度 → 损失）、masked avg pooling 细节
- **实践**：跟踪一次完整的 forward 调用

### [第15课：ProtoLearner — 训练与测试循环](./Lesson_15_ProtoLearner.md)
- **目标**：理解 ProtoLearner 如何组织训练和测试
- **内容**：train() 和 test_few_shot() 的完整流程、交叉熵损失、准确率计算
- **实践**：在 `models/proto_learner.py` 中添加日志打印

### [第16课：参数标志与模型变体](./Lesson_16_Model_Variants.md)
- **目标**：理清各个 `--use_xxx` 开关与模型变体的对应关系
- **内容**：ProtoNet vs ProtoNetAlignQGPASR vs ProtoNetAlignFZ、BaseLearner/attention/transformer/align 的组合逻辑
- **实践**：绘制 flag 组合的决策树

---

## 第五阶段：高级模块 — 注意力与自适应

### [第17课：自注意力机制 (Self-Attention)](./Lesson_17_Self_Attention.md)
- **目标**：理解点云自注意力的原理与实现
- **内容**：Scaled Dot-Product Attention、Multi-Head Attention、在点云中的应用、SelfAttention 类源码解读
- **实践**：在 `models/attention.py` 中调试 attention weights

### [第18课：交叉注意力与 QGPA 模块](./Lesson_18_QGPA.md)
- **目标**：理解 Query-Guided Prototype Alignment（QGPA）的设计思想
- **内容**：原型自适应动机、Query 特征引导、交叉注意力机制、残差连接 + LayerNorm
- **实践**：跟踪 QGPA forward 中的 Q, K, V 张量

### [第19课：特征投影与 BaseLearner](./Lesson_19_BaseLearner.md)
- **目标**：理解特征预处理和投影层的作用
- **内容**：BaseLearner 的 Conv1d 层、线性投影 (--use_linear_proj)、多级特征拼接
- **实践**：对比有无 linear_proj 时的特征维度变化

### [第20课：Self-Reconstruction 损失](./Lesson_20_Self_Reconstruction.md)
- **目标**：理解原型自重建损失的动机与实现
- **内容**：--use_supervise_prototype 的工作原理、Encoder-Decoder 重建、残差计算
- **实践**：分析 `protonet_QGPA.py` 中的 reconstruction_loss 计算

### [第21课：原型对齐损失 (Alignment Loss)](./Lesson_21_Alignment_Loss.md)
- **目标**：理解原型对齐损失的原理
- **内容**：--use_align 的含义、对齐什么和什么、对齐损失的计算方式、对训练的影响
- **实践**：在训练中关闭 --use_align，对比损失曲线

---

## 第六阶段：零样本学习扩展

### [第22课：词嵌入与语义映射](./Lesson_22_Word_Embeddings.md)
- **目标**：理解如何用词向量连接视觉和语义空间
- **内容**：GloVe 词嵌入原理、读取与映射、`get_embedding.py` 脚本解析、语义嵌入作为 Zero-Shot 桥梁
- **实践**：运行 `python dataloaders/get_embedding.py --dataset S3DIS`

### [第23课：GMMN 生成器](./Lesson_23_GMMN_Generator.md)
- **目标**：理解生成式矩匹配网络的原理
- **内容**：生成架构（噪声 + 词嵌入 → 伪原型）、GMMNLoss 定义、MMD (Maximum Mean Discrepancy)
- **实践**：阅读 `models/gmmn.py` 中的 GMMNLoss 多带宽高斯核实现

### [第24课：ProtoLearnerFZ — 联合少样本与零样本](./Lesson_24_ProtoLearnerFZ.md)
- **目标**：理解如何在训练中同时处理 seen 和 unseen 类
- **内容**：GMMN 生成器集成、生成损失权重 (--gmm_weight)、前向传播中的生成分支
- **实践**：运行 `train_PAPFZ.sh`，对比与 `train_PAP.sh` 的区别

---

## 第七阶段：训练流程与实验评估

### [第25课：Episode 训练策略与超参数](./Lesson_25_Training_Strategy.md)
- **目标**：掌握元训练的超参数调优技巧
- **内容**：学习率调度、Episode 数量与验证频率、模型保存策略、early stopping

### [第26课：评估指标 — IoU 计算与解释](./Lesson_26_Evaluation_Metrics.md)
- **目标**：深入理解语义分割的评估标准
- **内容**：Per-class IoU、Mean IoU (mIoU)、混淆矩阵、背景类的处理
- **实践**：阅读 `runs/eval.py` 中的 IoU 计算代码

### [第27课：Shell 实验脚本解析](./Lesson_27_Experiment_Scripts.md)
- **目标**：理解并复用项目提供的实验脚本
- **内容**：5 个脚本的功能对比、参数配置、运行顺序、多 GPU 支持
- **实践**：修改 `scripts/eval_PAP.sh` 中的 n_way/k_shot 参数并运行

### [第28课：实验结果复现与可视化](./Lesson_28_Reproducing_Results.md)
- **目标**：学会复现论文中的实验结果
- **内容**：完整实验流程（预处理 → 预训练 → 元训练 → 评估）、TensorBoard 使用、结果对比
- **实践**：完成一次完整的 2-way 1-shot S3DIS 实验

---

## 第八阶段：拓展主题与项目扩展

### [第29课：MPTI — 多原型传导推理](./Lesson_29_MPTI.md)
- **目标**：理解另一种少样本推理范式
- **内容**：MPTI vs ProtoNet 的区别、FPS 子原型聚类、图构建与标签传播、闭合解推导
- **实践**：在 `models/mpti.py` 中调试标签传播矩阵

### [第30课：项目扩展方向与总结](./Lesson_30_Extensions_and_Summary.md)
- **目标**：总结整个教程，指引未来扩展方向
- **内容**：可扩展方向（新数据集、新骨干、3D Transformer、开放词汇分割）、关键设计思想回顾、课程总结
- **实践**：尝试将 DGCNN 替换为 PointNeXt / PointTransformer

---

## 附录

### 关键文件索引

| 文件 | 用途 | 对应课程 |
|---|---|---|
| `main.py` | 主入口 | 第4课 |
| `dataloaders/loader.py` | 数据加载 | 第7-8课 |
| `models/dgcnn.py` / `dgcnn_new.py` | DGCNN 骨干 | 第9-12课 |
| `models/protonet.py` | 基础原型网络 | 第14课 |
| `models/protonet_QGPA.py` | QGPA 增强版 | 第17-21课 |
| `models/protonet_FZ.py` | 零样本扩展版 | 第23-24课 |
| `models/attention.py` | 注意力模块 | 第17-18课 |
| `models/gmmn.py` | GMMN 生成器 | 第23课 |
| `models/proto_learner.py` | 训练/测试循环 | 第15课 |
| `models/proto_learner_FZ.py` | FZ 训练循环 | 第24课 |
| `runs/eval.py` | 评估工具 | 第26课 |
| `runs/pre_train.py` | 预训练 | 第12课 |
