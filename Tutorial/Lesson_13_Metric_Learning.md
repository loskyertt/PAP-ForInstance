# 第13课：度量学习与原型网络原理

## 1. 本节学习目标

- 理解度量学习（Metric Learning）的核心概念
- 掌握原型网络（Prototypical Network）的数学原理
- 理解嵌入空间、原型、相似度度量的关系
- 能够解释为什么原型网络适合少样本学习

---

## 2. 从分类到度量学习的范式转变

### 传统分类

```
传统方法:
  x → f_θ(x) → W · f_θ(x) + b → softmax → P(y|x)
  
  问题: W 和 b 是每个类别特有的参数
  → 遇到新类别时，需要新的 W_new 和 b_new
  → 但只有 1 个样本，无法训练这些参数！
```

### 度量学习

```
度量学习方法:
  x → f_θ(x) → 嵌入特征 z
  
  对于新类别 c:
    Prototype_c = mean{ f_θ(x) for x in support set of class c }
  
  预测查询点 x_q:
    P(y=c | x_q) = softmax( -distance( f_θ(x_q), Prototype_c ) )
```

**关键区别**：分类决策基于 **特征空间的欧氏距离/余弦相似度**，而非可学习的分类器权重。

---

## 3. 嵌入空间 (Embedding Space)

### 什么是好的嵌入空间

```
好的嵌入空间:
  ┌────────────────────┐
  │  * chair           │    同类的点聚在一起
  │    * chair         │    (类内紧凑)
  │                    │
  │        ● table     │    不同类分得开
  │        ● table     │    (类间分散)
  └────────────────────┘

差的嵌入空间:
  ┌────────────────────┐
  │  * chair  ● table  │    类别混在一起
  │  ● table  * chair  │    无法区分
  └────────────────────┘
```

### ProtoNet 中的嵌入空间

```python
# PAP-FZS3D 中嵌入空间的维度
embedding_dim = output_dim = 64  # 默认

# 点云 (2048, 9) → DGCNN → (2048, 256)
# BaseLearner 进一步压缩 → (2048, 64)
# 最终嵌入空间: ℝ^64
```

---

## 4. 原型 (Prototype) 的定义

### 数学定义

对于类别 c，给定 K 个 support 样本的特征 $\{z_1, z_2, ..., z_K\}$：

$$p_c = \frac{1}{|S_c|} \sum_{(x_i, y_i) \in S_c} f_\theta(x_i)$$

其中 $S_c$ 是类别 c 的 support 集合，$f_\theta$ 是特征提取器。

### 在 PAP 中的具体实现

由于是语义分割（逐点分类），原型是基于 **类别 mask** 的加权平均：

```python
# 伪代码
def compute_prototype(features, labels, class_idx):
    """
    features: (N, D)  一个 block 中所有点的特征
    labels:   (N,)    点的真实标签
    class_idx: int    目标类别
    """
    mask = (labels == class_idx)  # 属于该类别的点
    if mask.sum() == 0:
        return torch.zeros(D)     # 该类别没有点
    
    class_features = features[mask]     # (n_c, D)
    prototype = class_features.mean(dim=0)  # (D,)
    return prototype
```

### 多样本 Support 的原型

当 K > 1 时（多 shot），需要跨多个 support block 计算原型：

```python
def compute_prototype_multi_shot(support_features_list, support_labels_list):
    """
    support_features_list: [block1_feat, block2_feat, ..., blockK_feat]
    """
    all_class_features = []
    for features, labels in zip(support_features_list, support_labels_list):
        class_features = features[labels == class_idx]
        all_class_features.append(class_features)
    
    # 拼接所有 K 个 block 的该类别的点特征
    all_features = torch.cat(all_class_features, dim=0)  # (total_n_c, D)
    prototype = all_features.mean(dim=0)  # (D,)
    return prototype
```

---

## 5. 相似度度量

### PAP 使用的度量：余弦相似度

```python
def cosine_similarity(query_features, prototypes):
    """
    query_features: (N_q, D)  查询点的特征
    prototypes:     (C, D)    C 个类别的原型
    返回:           (N_q, C)  每个点对每个类的相似度
    """
    # L2 归一化
    query_norm = F.normalize(query_features, p=2, dim=-1)  # (N_q, D)
    proto_norm = F.normalize(prototypes, p=2, dim=-1)      # (C, D)
    
    # 点积 = 余弦相似度 (因为已归一化)
    similarity = query_norm @ proto_norm.T  # (N_q, C)
    
    return similarity
```

### 为什么用余弦相似度而非欧氏距离？

| 度量 | 公式 | 特点 |
|---|---|---|
| 欧氏距离 | $\|z_q - p_c\|_2$ | 对特征尺度敏感 |
| **余弦相似度** | $\frac{z_q \cdot p_c}{\|z_q\| \|p_c\|}$ | 只关心方向，尺度不变 |
| 负欧氏距离 | $-\|z_q - p_c\|_2$ | ProtoNet 论文原版 |

余弦相似度在特征归一化后等价于点积，计算高效且数值稳定。

---

## 6. 损失函数

### 原型网络的分类损失

```python
def prototypical_loss(similarity, query_labels):
    """
    similarity:  (N_q, C)  余弦相似度
    query_labels: (N_q,)   真实标签
    """
    # 相似度越高 → 概率越大
    # 使用 softmax 转换为概率分布
    logits = similarity * temperature  # 可选：控制分布锐度
    loss = F.cross_entropy(logits, query_labels)
    return loss
```

### 为什么不直接用欧氏距离

对比两种损失的等价性：

```python
# 当使用 softmax(-d^2) 时
P(y=c|z) = exp(-||z - p_c||^2) / sum_j exp(-||z - p_j||^2)

# 等价于
logits = 2 * z @ p_c.T - ||p_c||^2  # 线性分类器形式
```

这其实就是 Bregman 散度框架下的线性分类器。

---

## 7. 完整的预测流程

```python
def few_shot_predict(model, support_set, query_set):
    """
    完整的少样本预测流程
    """
    # Step 1: 编码所有点云
    support_feats = [model(s) for s in support_set]  # K 个 (N, D)
    query_feats = model(query_set)                     # (N_q, D)
    
    # Step 2: 计算每个类的原型
    prototypes = []
    for c in range(num_classes):
        p_c = compute_prototype(support_feats, support_labels, c)
        prototypes.append(p_c)
    prototypes = torch.stack(prototypes)  # (C, D)
    
    # Step 3: 计算相似度
    similarity = cosine_similarity(query_feats, prototypes)  # (N_q, C)
    
    # Step 4: 预测
    predictions = similarity.argmax(dim=-1)  # (N_q,)
    
    return predictions
```

---

## 8. 本节总结

| 概念 | 公式/要点 |
|---|---|
| 嵌入空间 | $f_\theta: \mathbb{R}^{N \times 9} \to \mathbb{R}^{N \times 64}$ |
| 原型 | $p_c = \frac{1}{|S_c|} \sum z_i$ |
| 余弦相似度 | $\cos(z_q, p_c) = \frac{z_q \cdot p_c}{\|z_q\|\|p_c\|}$ |
| 分类 | $P(y=c|z_q) = \text{softmax}(\cos(z_q, p_c))$ |
| 损失 | CrossEntropyLoss(similarity, labels) |

---

## 9. 课后练习

1. 用 PyTorch 手写一个简单的原型网络（1D 数据，2-way 1-shot），验证它能正确分类
2. 对比余弦相似度和欧氏距离在归一化嵌入上的等价性（数学推导）
3. 如果 support 中某类的点极少（<10个），原型计算的方差会如何变化？
4. 思考：为什么不用学习一个可训练的 "距离度量函数"，而用固定的余弦相似度？
