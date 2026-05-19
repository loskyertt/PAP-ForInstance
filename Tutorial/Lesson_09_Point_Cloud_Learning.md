# 第9课：点云特征学习基础 — 从 PointNet 到 EdgeConv

## 1. 本节学习目标

- 理解点云的无序性和置换不变性
- 掌握 PointNet 的对称函数设计思想
- 理解 k-NN 局部图在点云中的应用
- 能实现简单的 k-NN 搜索

---

## 2. 点云的核心挑战：无序性

### 图像 vs 点云

```
图像 (有序):
┌───┬───┬───┐
│ 1 │ 2 │ 3 │  ← 像素有固定位置关系
├───┼───┼───┤
│ 4 │ 5 │ 6 │  ← 卷积核利用空间连续性
├───┼───┼───┤
│ 7 │ 8 │ 9 │
└───┴───┴───┘

点云 (无序):
P = { (x1,y1,z1), (x2,y2,z2), ..., (xN,yN,zN) }
                                         
如果打乱顺序: P' = { (xN,yN,zN), ..., (x1,y1,z1) }
                 P 和 P' 表示同一个点云！
```

### 置换不变性 (Permutation Invariance)

函数 f 对点云 P 必须是置换不变的：

```
f(p1, p2, p3) = f(p3, p1, p2) = f(p2, p3, p1) = ...
```

**为什么重要？** 因为改变点的输入顺序不应该改变语义分割的结果。

---

## 3. PointNet 的解决方案：对称函数

### 核心思想

PointNet 使用 **对称函数** 来聚合点特征：

```python
# PointNet 的核心公式
f({x1, ..., xN}) = max_pool { h(x1), h(x2), ..., h(xN) }
                     ^^^^^^^^
                     对称函数：对输入顺序不敏感
```

`max_pool` 是对称的：`max(a,b,c) = max(c,a,b)`。

### PointNet 架构

```
输入:  (N, 3)
  ↓
MLP → (N, 64)     # 逐点处理
  ↓
MLP → (N, 128)    # 共享权重
  ↓
Max Pool → (128)  # 对称聚合 → 全局特征
  ↓
MLP → (K)         # 分类输出
```

### PointNet 的局限

- **忽略局部结构**：Max Pool 只保留全局特征，丢失局部几何信息
- **无法捕获邻域关系**：每个点独立处理，不知道邻近点的信息
- **不适合语义分割**：分割需要逐点输出，PointNet 的设计更适合全局分类

---

## 4. 从全局到局部：k-NN 图

### 为什么需要局部结构

在 3D 空间中，点的语义由其局部邻域决定：

```
  ________
 / 桌面  /
 --------
    ↑
    这个点是"桌面"还是"地板"？
    → 看它周围的点：如果周围都是水平面的点 → 桌面
    
  |   |
  |   |  ← 这个点是"墙壁"还是"柱子"？
  |   |  → 看它周围的点：如果在一个平面上 → 墙壁
  ------
```

### k 近邻 (k-NN) 图

为每个点找 k 个最近的邻居：

```python
def knn_graph(points, k=20):
    """
    points: (N, 3) 点云坐标
    返回: (N, k) 每个点的 k 个邻居索引
    """
    N = points.shape[0]
    # 计算距离矩阵 (N, N)
    dist = torch.cdist(points, points)
    
    # 忽略自己 (距离=0)
    # 选最近的 k 个点
    _, indices = torch.topk(dist, k+1, largest=False)
    return indices[:, 1:]  # 去掉自己，返回 k 个邻居
```

---

## 5. 局部图 → EdgeConv

EdgeConv 的核心思想：对每条边提取特征，而非对每个点。

### 边特征定义

```
边 (i, j): 连接中心点 i 和邻居点 j

边特征 = concat[ p_j - p_i ,  p_i ]
         ^^^^^^^^^^^^  ^^^^
         局部差异      全局位置
```

```python
# 边特征计算 (简化)
def edge_feature(center_features, neighbor_features):
    """
    center_features:    (N, k, C)  中心点的特征
    neighbor_features:  (N, k, C)  邻居点的特征
    """
    # 1. 局部差异
    diff = neighbor_features - center_features  # (N, k, C)
    
    # 2. 拼接中心特征
    edge_feat = torch.cat([diff, center_features], dim=-1)  # (N, k, 2C)
    
    return edge_feat
```

### EdgeConv 的完整公式

```
对于点 i:
1. 找 k 个邻居: N(i) = {j1, ..., jk}
2. 对每条边 (i, j):
   e_ij = MLP( concat[ x_j - x_i , x_i ] )
3. 聚合所有边:
   x_i' = max_pool { e_i1, e_i2, ..., e_ik }
```

---

## 6. 对比：标准卷积 vs EdgeConv

| 方面 | 标准 2D 卷积 | EdgeConv |
|---|---|---|
| 输入域 | 固定网格 (image) | 动态图 (point cloud) |
| 邻域 | 固定 (3x3, 5x5...) | k-NN 动态邻域 |
| 操作 | 权重和 | MLP + max_pool |
| 置换不变性 | 天然具有 | 通过 max_pool 实现 |

---

## 7. 实践：实现 k-NN 搜索

```python
import torch

def knn(x, k):
    """
    x: (batch_size, num_points, num_dims)
    """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

# 测试
points = torch.randn(4, 2048, 3)  # batch=4, 2048 点, 3D
neighbors = knn(points, k=20)
print(f"Neighbors shape: {neighbors.shape}")  # (4, 2048, 20)
```

---

## 8. 本节总结

| 概念 | 要点 |
|---|---|
| 无序性 | 点云是集合，不是序列 |
| 置换不变性 | 输入顺序不影响输出 |
| 对称函数 | max/min/sum 等对输入顺序不敏感 |
| PointNet | MLP + MaxPool，全局特征学习 |
| k-NN 图 | 每个点找 k 个最近邻居 |
| EdgeConv | 提取边特征 (差异+中心)，MLP + MaxPool |

---

## 9. 课后练习

1. 实现 `knn` 函数，验证 `knn(x, k)` 对 x 的行置换是不变的
2. 比较 PointNet (MaxPool) 和 EdgeConv 在点云分类上的差异（纸上分析即可）
3. 写一个简单的 EdgeConv forward 函数（不需要可学习参数，仅手动计算边特征+聚合）
4. 思考：如果 k=1（只有最近邻），EdgeConv 退化为什么？
