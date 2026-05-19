# 第14课：ProtoNet 模型的代码实现

## 1. 本节学习目标

- 逐行阅读 `models/protonet.py` 中的 ProtoNet 实现
- 理解 forward() 的完整流程
- 掌握 masked average pooling 的实现细节
- 能够对比基础版本和增强版本的区别

---

## 2. ProtoNet 模型结构

### 类定义

```python
# models/protonet.py (简化)
class ProtoNet(nn.Module):
    """
    基础的原型网络，用于少样本 3D 点云语义分割
    """
    def __init__(self, args):
        super().__init__()
        
        # 特征编码器: DGCNN
        self.encoder = DGCNN_semseg(args)
        
        # 特征精炼: BaseLearner (Conv1d 层)
        self.base_widths = args.base_widths  # e.g. [128, 64]
        self.output_dim = args.output_dim    # e.g. 64
        
        self.base_learner = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),          # 输出 channel = output_dim
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
    
    def forward(self, support_x, support_y, query_x, query_y):
        ...
```

### 模型参数

```python
# 默认配置下的模型输出
encoder:      DGCNN_semseg → (B, N, 256)
base_learner: Conv1d(256→128→64) → (B, N, 64)
最终嵌入:     (B, N, 64)
```

---

## 3. forward() 完整流程解析

```python
def forward(self, support_x, support_y, query_x, query_y):
    """
    Args:
        support_x: (B, N_way*K_shot, pc_npts, dim)  support 点云
        support_y: (B, N_way*K_shot)                 support 标签
        query_x:   (B, N_way*Q, pc_npts, dim)        query 点云
        query_y:   (B, N_way*Q)                      query 标签
    
    Returns:
        loss:      scalar
        acc:       scalar (query 上的准确率)
    """
    B = support_x.size(0)
    n_support = support_x.size(1)
    n_query = query_x.size(1)
    
    # ============ Step 1: 合并 support 和 query ============
    # 为了提高效率，将 support 和 query 一起编码
    all_x = torch.cat([support_x, query_x], dim=1)
    # all_x: (B, n_support + n_query, pc_npts, dim)
    
    # 展平 batch 和 block 维度
    all_x = all_x.view(-1, self.pc_npts, self.input_dim)
    # all_x: (B * (n_support + n_query), pc_npts, dim)
    
    # ============ Step 2: 特征编码 ============
    feat = self.encoder(all_x)      # (B*(S+Q), pc_npts, 256)
    feat = self.base_learner(feat.transpose(1,2)).transpose(1,2)
    # feat: (B*(S+Q), pc_npts, 64)
    
    # ============ Step 3: 分离 support 和 query 特征 ============
    feat = feat.view(B, n_support + n_query, self.pc_npts, -1)
    
    support_feat = feat[:, :n_support]    # (B, S, pc_npts, 64)
    query_feat = feat[:, n_support:]      # (B, Q, pc_npts, 64)
    
    # ============ Step 4: 计算原型 (Masked Avg Pool) ============
    prototypes = self._compute_prototypes(
        support_feat, support_y, n_support
    )  # (B, n_way, 64)
    
    # ============ Step 5: 计算相似度 ============
    query_feat = query_feat.view(B, -1, self.output_dim)  # (B, Q*pc_npts, 64)
    
    # 余弦相似度
    prototypes = F.normalize(prototypes, dim=-1)
    query_feat = F.normalize(query_feat, dim=-1)
    
    similarity = torch.matmul(query_feat, prototypes.transpose(1,2))
    # (B, Q*pc_npts, n_way)
    
    # ============ Step 6: 计算损失 ============
    query_y = query_y.unsqueeze(-1).expand(-1, -1, self.pc_npts)
    query_y = query_y.reshape(B, -1)  # (B, Q*pc_npts)
    
    loss = F.cross_entropy(
        similarity.transpose(1,2),  # (B, n_way, N)
        query_y
    )
    
    # 准确率
    pred = similarity.argmax(dim=-1)
    acc = (pred == query_y).float().mean()
    
    return loss, acc
```

---

## 4. Masked Average Pooling 详解

这是原型计算中最关键的一步：

```python
def _compute_prototypes(self, support_feat, support_y, n_support):
    """
    通过 masked avg pooling 计算每个类的原型
    
    Args:
        support_feat: (B, n_support, pc_npts, D)  特征
        support_y:    (B, n_support)               每 block 的类别标签
        n_support:    int = n_way * k_shot
    
    Returns:
        prototypes: (B, n_way, D)
    """
    B, S, N, D = support_feat.shape
    n_way = S // self.k_shot  # 类别数
    
    prototypes = torch.zeros(B, n_way, D).cuda()
    
    for i in range(S):
        # 确定当前 block 属于哪个类别
        class_idx = support_y[0, i].item()  # 假设 batch 内类别相同
        block_feat = support_feat[:, i]     # (B, N, D)
        
        # 创建前景 mask
        # 实际代码中，每个点有前景/背景标签
        # 这里简化为：所有点都是前景
        fg_mask = torch.ones(B, N, 1).cuda()  # (B, N, 1)
        
        # 加权平均
        masked_feat = block_feat * fg_mask    # (B, N, D)
        sum_feat = masked_feat.sum(dim=1)     # (B, D)
        count = fg_mask.sum(dim=1)           # (B, 1)
        
        prototypes[:, class_idx] += sum_feat / (count + 1e-8)
    
    # 对多 shot 情况取平均
    prototypes /= self.k_shot
    
    return prototypes
```

### 为什么用 Masked Avg Pooling？

在语义分割中，每个 block 可能包含多个类别的点。我们需要只聚合 **某个类别** 的点的特征：

```
Block A (chair):
  [●, ●, ○, ○, ●, ...]   # ● = chair 点, ○ = 其他类点
  → Mask = [1, 1, 0, 0, 1, ...]
  → Prototype_chair = mean(● 点的特征)
```

---

## 5. 张量形状全追踪

以默认配置 `B=1, n_way=2, k_shot=1, n_queries=1, pc_npts=2048`：

```
Step 1: 合并
  all_x: (1, 3, 2048, 9) → view → (3, 2048, 9)

Step 2: 编码
  feat: (3, 2048, 256) → base_learner → (3, 2048, 64)

Step 3: 分离
  feat: (1, 3, 2048, 64)
  support_feat: (1, 2, 2048, 64)
  query_feat:   (1, 1, 2048, 64)

Step 4: 原型
  prototypes: (1, 2, 64)   # 2 个类，每个 64 维

Step 5: 相似度
  query_feat: (1, 2048, 64)
  similarity: (1, 2048, 2)  # 每个点对 2 个类的相似度

Step 6: 损失
  loss: scalar
  acc:  scalar
```

---

## 6. 基础版本 vs 增强版本

| 模块 | ProtoNet | ProtoNetAlignQGPASR | ProtoNetAlignFZ |
|---|---|---|---|
| DGCNN encoder | ✅ | ✅ | ✅ |
| BaseLearner | ✅ | ✅ | ✅ |
| Self-Attention | ❌ | ✅ | ✅ |
| QGPA Transformer | ❌ | ✅ | ✅ |
| Alignment Loss | ❌ | ✅ | ✅ |
| Self-Reconstruction | ❌ | ✅ | ✅ |
| GMMN Generator | ❌ | ❌ | ✅ |

---

## 7. 本节总结

| 步骤 | 操作 | 输入形状 | 输出形状 |
|---|---|---|---|
| 合并 | cat(support, query) | — | (B, S+Q, N, 9) |
| 编码 | DGCNN + BaseLearner | (B*(S+Q), N, 9) | (B*(S+Q), N, 64) |
| 分离 | 切片 | (B, S+Q, N, 64) | support + query |
| 原型 | Masked Avg Pool | (B, S, N, 64) | (B, C, 64) |
| 相似度 | Cosine Similarity | (B, QN, 64) | (B, QN, C) |
| 损失 | CrossEntropy | (B, C, QN) | scalar |

---

## 8. 课后练习

1. 在 `models/protonet.py` 的 forward() 中每步插入 print(x.shape)，验证形状追踪
2. 手工计算：n_way=3, k_shot=5, pc_npts=1024 时，每步的形状
3. 实现一个简化版 ProtoNet（不含 DGCNN，用随机特征），验证 13 讲中的原理
4. 为什么 Step 1 要把 support 和 query 合并编码？分开编码有什么问题？
