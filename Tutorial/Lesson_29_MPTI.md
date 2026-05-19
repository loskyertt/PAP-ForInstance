# 第29课：MPTI — 多原型传导推理

## 1. 本节学习目标

- 理解 MPTI (Multi-Prototype Transductive Inference) 的设计思想
- 掌握 FPS 子原型聚类和图标签传播
- 理解 MPTI 与 ProtoNet 的根本区别
- 能够阅读和理解 `models/mpti.py` 的核心实现

---

## 2. MPTI vs ProtoNet

### 范式差异

```
ProtoNet (归纳式, Inductive):
  Support → 特征 → 单一原型 → 与 query 对比 → 分类
  独立处理每个 query 点

MPTI (传导式, Transductive):
  Support → 特征 → 多个子原型 → 构建图 → 标签传播 → 分类
  同时利用所有 query 点的相互信息
```

### 核心思想

> Query 点之间也存在相似性关系。如果能利用 query 内部的相似性结构，就可以进行更准确的分类。

```python
# 直观例子
query 点 A: 特征 [0.8, 0.1, -0.2]  → 和 support 的 table 原型最近
query 点 B: 特征 [0.75, 0.15, -0.18] → 不太确定（table 和 chair 都接近）

ProtoNet: 独立分类 A 和 B
MPTI:    发现 A 和 B 很相似 → 互相增强置信度 → 都分类为 table
```

---

## 3. 子原型聚类 (FPS-based)

### 为什么需要多原型

```
单一原型的问题:
  table 类可能包含不同形状的桌子
  - 圆桌: 圆形平面
  - 方桌: 矩形平面
  - 会议桌: 大长条
  
  单一原型 = 所有形状的平均 → 可能不准确

多原型:
  用 FPS 在 support 特征空间中聚类
  每个子原型代表一种子类型
  → 更精细的表示
```

### FPS 子原型选择

```python
# models/mpti.py (简化)
from torch_cluster import fps

def compute_sub_prototypes(support_features, support_labels, 
                           num_sub_prototypes=10):
    """
    为每个类选择多个子原型
    """
    sub_prototypes = []
    sub_labels = []
    
    for c in range(num_classes):
        # 提取该类的 support 特征
        class_features = support_features[support_labels == c]  # (n_c, D)
        
        if class_features.shape[0] < num_sub_prototypes:
            # 不够选，直接用全部
            selected = class_features
        else:
            # FPS 选代表性的点
            # fps 返回索引（在特征空间中的 farthest point sampling）
            batch = torch.zeros(class_features.shape[0], dtype=torch.long)
            idx = fps(class_features, batch, 
                      ratio=num_sub_prototypes / class_features.shape[0])
            selected = class_features[idx]
        
        sub_prototypes.append(selected)
        sub_labels.extend([c] * len(selected))
    
    return torch.cat(sub_prototypes, dim=0), torch.tensor(sub_labels)
```

---

## 4. 图构建

### 节点

```
图节点 = [子原型节点] + [query 特征节点]

  原型1 (class 0) ──┐
  原型2 (class 0) ──┤
  原型3 (class 1) ──┤  support 侧 (有标签节点)
  原型4 (class 1) ──┘
  ─────────────────
  query 点 1        ──┐
  query 点 2        ──┤  query 侧 (无标签节点)
  ...               ──┘
```

### 邻接矩阵 (相似度)

```python
def build_adjacency_matrix(prototypes, query_features, sigma=1.0):
    """
    构建所有节点之间的相似度邻接矩阵
    
    Args:
        prototypes:     (N_p, D)   子原型特征
        query_features: (N_q, D)   query 特征
    
    Returns:
        W: (N_p + N_q, N_p + N_q)  相似度矩阵
    """
    all_features = torch.cat([prototypes, query_features], dim=0)
    N = all_features.shape[0]
    
    # 计算 pairwise 相似度
    all_norm = F.normalize(all_features, dim=-1)
    W = all_norm @ all_norm.T  # (N, N)
    
    # 高斯核归一化
    W = torch.exp(W / sigma)
    
    # 自连接为 0
    W.fill_diagonal_(0)
    
    return W
```

---

## 5. 标签传播 (Label Propagation)

### 数学公式

给定邻接矩阵 W，节点标签矩阵 Y：

1. 计算归一化拉普拉斯矩阵：
   $$S = D^{-1/2} W D^{-1/2}$$

2. 迭代传播或闭合解：
   $$F = (I - \alpha S)^{-1} Y$$

### 代码实现

```python
def label_propagation(W, Y_labeled, labeled_mask, alpha=0.99):
    """
    标签传播的闭合形式解
    
    Args:
        W:             (N, N) 相似度矩阵
        Y_labeled:     (N_labeled, C) 已标注节点的 one-hot
        labeled_mask:  (N,) bool, 哪些节点有标签
        alpha:         传播系数 (接近 1 → 强传播)
    
    Returns:
        F: (N, C) 所有节点的软标签
    """
    N = W.shape[0]
    C = Y_labeled.shape[1]
    
    # Step 1: 归一化图拉普拉斯
    D = W.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-10))
    S = D_inv_sqrt @ W @ D_inv_sqrt
    
    # Step 2: 构建初始标签矩阵
    Y = torch.zeros(N, C)
    Y[labeled_mask] = Y_labeled
    
    # Step 3: 闭合解 F = (I - αS)^{-1} Y
    I = torch.eye(N).cuda()
    F = torch.linalg.solve(I - alpha * S, Y)
    
    # Step 4: 归一化到 [0, 1]
    F = F / (F.sum(dim=1, keepdim=True) + 1e-10)
    
    return F
```

---

## 6. 完整 MPTI 流程

```python
# models/mpti.py
class MPTI(nn.Module):
    def forward(self, support_x, support_y, query_x):
        """
        多原型传导推理的完整流程
        """
        # Step 1: 特征提取
        all_x = torch.cat([support_x, query_x], dim=1)
        features = self.encoder(all_x)
        
        # Step 2: 计算子原型
        sub_protos, sub_labels = compute_sub_prototypes(
            features[:, :n_support], support_y
        )
        
        # Step 3: 构建邻接图
        query_features = features[:, n_support:]
        W = build_adjacency_matrix(sub_protos, query_features)
        
        # Step 4: 标签传播
        Y_labeled = F.one_hot(sub_labels, num_classes)
        labeled_mask = torch.zeros(N, dtype=bool)
        labeled_mask[:len(sub_protos)] = True
        
        F = label_propagation(W, Y_labeled, labeled_mask, alpha=0.99)
        
        # Step 5: 获取 query 的预测
        query_predictions = F[-n_query:]  # 最后 N_q 行的软标签
        
        return query_predictions
```

---

## 7. MPTI 的优劣势

### 优势

| 优势 | 说明 |
|---|---|
| **利用 query 结构** | 利用 query 点间的相似性增强分类 |
| **多原型表示** | 每个类用多个子原型，更灵活 |
| **闭合解** | 标签传播有解析解，计算高效 |
| **转导推理** | 在同一次推理中看到所有 query 点 |

### 劣势

| 劣势 | 说明 |
|---|---|
| **批量依赖** | 结果依赖同时处理的 query 点的组成 |
| **在线场景不适** | 不能逐个处理 query 点 |
| **图构建开销** | N² 的相似度矩阵（但点云本身就小） |
| **超参数敏感** | alpha 和 sigma 影响较大 |

---

## 8. MPTI 与 QGPA 的对比

| 特性 | QGPA (in PAP) | MPTI |
|---|---|---|
| 作用对象 | 原型 | 原型 + query |
| 机制 | Cross-attention | 图标签传播 |
| 关系建模 | Prototype ↔ Query | Prototype ↔ Query + Query ↔ Query |
| 输出 | 自适应原型 | query 软标签 |
| 计算复杂度 | O(C × N_q) | O((N_p + N_q)²) |

---

## 9. 本节总结

| 概念 | 要点 |
|---|---|
| MPTI | 多原型 + 转导推理 |
| 子原型 | FPS 在 support 特征中选择多个代表点 |
| 图构建 | 子原型 + query → 相似度邻接矩阵 |
| 标签传播 | (I - αS)^(-1) Y 闭合解 |
| 与 ProtoNet 区别 | MPTI 利用 query 内部关联，ProtoNet 逐点独立分类 |

---

## 10. 课后练习

1. 运行 `python main.py --phase mptitrain`，对比 MPTI 和 PAP 的性能
2. 修改子原型数量 (5, 10, 20)，观察对 MPTI 性能的影响
3. 实现迭代版标签传播（而非闭合解），对比两者的速度和精度
4. 分析：MPTI 在 K-shot 变大时是否有优势？为什么？
