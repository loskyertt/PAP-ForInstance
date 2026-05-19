# 第10课：EdgeConv — 动态边卷积详解

## 1. 本节学习目标

- 深入理解 EdgeConv 的数学原理和 PyTorch 实现
- 掌握动态图更新（Dynamic Graph Update）的概念
- 能够阅读和理解 `models/dgcnn.py` 中的 EdgeConv 类
- 理解 EdgeConv 为什么优于静态图方法

---

## 2. EdgeConv 的数学定义

设点云为 $X = \{x_1, x_2, ..., x_N\} \subset \mathbb{R}^F$，对于每个点 $x_i$：

1. **构建 k-NN 图**：找到 $x_i$ 的 k 个最近邻 $\{x_{i_1}, x_{i_2}, ..., x_{i_k}\}$

2. **边特征映射**：对每条边 $(i, j)$：
   $$e_{ij} = h_\Theta(x_j - x_i, x_i)$$

   其中 $h_\Theta$ 是一个共享的 MLP

3. **聚合**：
   $$x_i' = \max_{j:(i,j)\in\mathcal{E}} e_{ij}$$

### 完整公式（单层）
$$x_i' = \max_{j \in \mathcal{N}(i)} \text{ReLU}(\theta \cdot (x_j - x_i) + \phi \cdot x_i)$$

---

## 3. 代码实现逐行解读

### `models/dgcnn.py` 中的 EdgeConv

```python
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        # 卷积层: 对边特征 (2*in_channels) 做 1x1 卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        """
        x: (batch_size, num_points, in_channels)
        返回: (batch_size, num_points, out_channels)
        """
        batch_size, num_points, _ = x.shape
        
        # Step 1: 构建 k-NN 图
        # 计算 pairwise 距离矩阵
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(x ** 2, dim=-1, keepdim=True)
        pairwise_distance = -(x_inner + x_square + x_square.transpose(2, 1))
        
        # 取距离最近的 k 个点 (包含自身)
        _, nn_idx = pairwise_distance.topk(
            k=self.k, dim=-1
        )  # (B, N, k)
        
        # Step 2: 收集邻居特征
        # 用 nn_idx 索引邻居
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.k, -1)
        # (B, N, k, C)
        
        nn_feat = gather_feature(x, nn_idx)  # (B, N, k, C)
        
        # Step 3: 构建边特征
        # [中心特征, 邻居特征-中心特征]
        edge_feat = torch.cat([
            x_expanded,                # 中心特征
            nn_feat - x_expanded       # 差异特征
        ], dim=-1)  # (B, N, k, 2C)
        
        # Step 4: 逐边卷积 + 聚合
        edge_feat = edge_feat.permute(0, 3, 1, 2)  # (B, 2C, N, k)
        edge_feat = self.conv(edge_feat)             # (B, C', N, k)
        edge_feat = edge_feat.max(dim=-1)[0]         # (B, C', N)
        
        return edge_feat
```

### 关键辅助函数：gather_feature

```python
def gather_feature(x, idx):
    """
    x:   (B, N, C)  特征
    idx: (B, N, k)  邻居索引
    返回: (B, N, k, C)  邻居特征
    """
    B, N, C = x.shape
    k = idx.shape[-1]
    
    # 扩展 batch 索引
    batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, N, k)
    
    return x[batch_idx, idx, :]
```

---

## 4. 张量形状追踪

以典型配置为例：`B=1, N=2048, C_in=9, C_out=64, k=20`

```
输入 x:        (1, 2048, 9)

Step 1 - k-NN:
  pairwise_distance: (1, 2048, 2048)
  nn_idx:            (1, 2048, 20)

Step 2 - 收集邻居:
  x_expanded:  (1, 2048, 20, 9)
  nn_feat:     (1, 2048, 20, 9)

Step 3 - 边特征:
  edge_feat:   (1, 2048, 20, 18)  # 9+9=18

Step 4 - 卷积+聚合:
  permute:     (1, 18, 2048, 20)
  conv:        (1, 64, 2048, 20)
  max pool:    (1, 64, 2048)

输出:          (1, 2048, 64)
```

---

## 5. 动态图更新 (Dynamic Graph Update)

### 什么是"动态图"

每层 EdgeConv 的 k-NN 图是在 **当前层的特征空间** 中计算的，而非固定的 XYZ 空间。

```python
# 第 1 层 EdgeConv:
#   k-NN 在原始坐标空间 (xyz) 中计算
#   → 捕获几何上的邻居关系

# 第 2 层 EdgeConv:
#   k-NN 在特征空间 (经过第1层后的特征) 中计算
#   → 可能连接"语义上相似"但"空间上远离"的点
```

### 动态图的优势

```
几何空间:            特征空间 (动态):
                      
● ← 近邻 == 几何近        ★ ← 近邻 == "几何远但语义近"
●                       
    ●                        ★  (被识别为同一类)
```

**例子**：
- 房间两端的椅子在几何上相距很远
- 但第 2 层 EdgeConv 的特征空间中，它们可能成为近邻
- 这样信息可以在远距离间传播 → 增大感受野

---

## 6. EdgeConv vs 其他卷积操作

| 操作 | 邻域定义 | 是否动态 | 适用域 |
|---|---|---|---|
| **标准 2D Conv** | 固定网格窗口 | 否 | 图像 |
| **3D Voxel Conv** | 固定体素窗口 | 否 | 体素化点云 |
| **PointNet++ Conv** | 球查询 + FPS | 否 (几何空间) | 点云 |
| **EdgeConv** | k-NN 在特征空间 | **是** | 点云 |
| **PointConv** | 加权球查询 | 否 | 点云 |

EdgeConv 的核心优势：动态图使模型能够学习 **任务自适应的邻域关系**。

---

## 7. 实践：可视化 EdgeConv 的邻居关系

```python
import numpy as np
import matplotlib.pyplot as plt

# 模拟点云
points = np.random.randn(100, 2) * 2  # 2D 便于可视化
center_idx = 0  # 观察中心点

# 计算最近邻
dist = np.sum((points - points[center_idx])**2, axis=1)
nn_idx = np.argsort(dist)[:6]  # k=5 + 自己

# 可视化
plt.scatter(points[:, 0], points[:, 1], alpha=0.5)
plt.scatter(points[center_idx, 0], points[center_idx, 1], 
            c='red', s=100, label='Center')
plt.scatter(points[nn_idx[1:], 0], points[nn_idx[1:], 1], 
            c='green', s=80, label='Neighbors')

for j in nn_idx[1:]:
    plt.plot([points[center_idx, 0], points[j, 0]],
             [points[center_idx, 1], points[j, 1]], 
             'g--', alpha=0.5)

plt.legend()
plt.title('EdgeConv: Center Point and its k-NN')
plt.show()
```

---

## 8. 常见错误

### Q1: k 值选择

```python
# 错误: k=N (全连接)
nn_idx = pairwise_distance.topk(k=N)  # 失去局部性

# 错误: k=1
nn_idx = pairwise_distance.topk(k=1)  # 太局部，忽略邻域信息

# 正确: k=20~30 (任务相关)
nn_idx = pairwise_distance.topk(k=20)
```

### Q2: 忘记 BatchNorm

```python
# 错误: 没有在 Conv 后加 BN
self.conv = nn.Conv2d(...)

# 正确: Conv → BN → Activation
self.conv = nn.Sequential(
    nn.Conv2d(...), 
    nn.BatchNorm2d(...), 
    nn.LeakyReLU()
)
```

---

## 9. 本节总结

| 概念 | 要点 |
|---|---|
| k-NN 图 | 在特征空间中为每个点找 k 个最近邻 |
| 边特征 | concat[中心, 差异] 捕获局部模式 |
| 1x1 Conv(Conv2d) | 对每条边独立处理（共享权重） |
| Max Pool | 对称聚合 → 置换不变性 |
| 动态图更新 | 每层重新计算 k-NN，特征空间邻域 ≠ 几何邻域 |

---

## 10. 课后练习

1. 在 `models/dgcnn.py` 的 EdgeConv.forward 中插入 print(x.shape)，验证每步的形状变化
2. 修改 k 值（5, 10, 20, 50），观察对模型参数量和推理速度的影响
3. 尝试用 AvgPool 替代 MaxPool，分析对 EdgeConv 置换不变性的影响
4. 画图说明：动态图更新如何帮助"椅子 A"和"椅子 B"在特征空间中成为近邻
