# 第20课：Self-Reconstruction 损失

## 1. 本节学习目标

- 理解 Self-Reconstruction (SR) 损失的动机与原理
- 掌握 `--use_supervise_prototype` 的工作机制
- 理解 Encoder-Decoder 重建在原型学习中的作用
- 能够阅读和理解 `protonet_QGPA.py` 中的 reconstruction loss

---

## 2. 动机：原型需要更强的监督

### 问题

在基础 ProtoNet 中，原型的学习**仅依赖分类损失**：

```python
loss = CrossEntropy(similarity, labels)
```

这意味着：

- 原型只要能让分类正确就行，不一定"真的好"
- 原型可能退化：把所有点推到同一个区域，分类靠微小的差异
- 原型缺乏"重构自身"的能力——它只是一个任务驱动的统计量

### Self-Reconstruction 的思路

> 好的原型应该能够"重建"产生它的 support 特征。

```
如果原型 p_c 真的是类别 c 的好代表：
  那么 p_c 应该能通过 decoder 重建出该类别的 support 特征
  
损失: L_sr = || decoder(p_c) - support_feat[c] ||^2
```

---

## 3. 数学定义

### Self-Reconstruction Loss

$$\mathcal{L}_{SR} = \sum_{c=1}^{C} \frac{1}{|S_c|} \sum_{i \in S_c} \| g(p_c) - z_i \|_2^2$$

其中：
- $p_c$：类别 c 的原型
- $g(\cdot)$：重建 decoder（一个小 MLP）
- $z_i$：support 集合中属于类别 c 的特征
- $S_c$：support 中类别 c 的样本集合

### 总损失

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{SR} \cdot \mathcal{L}_{SR}$$

---

## 4. 代码实现

### 重建头 (Reconstruction Head)

```python
# models/protonet_QGPA.py (简化)
class ProtoNetAlignQGPASR(nn.Module):
    def __init__(self, args):
        super().__init__()
        ...
        
        # Self-Reconstruction Decoder
        if args.use_supervise_prototype:
            self.reconstructor = nn.Sequential(
                nn.Linear(args.output_dim, 128),  # 64 → 128
                nn.ReLU(),
                nn.Linear(128, args.output_dim)   # 128 → 64 (重建)
            )
    
    def forward(self, support_x, support_y, query_x, query_y):
        ...
        # 计算原型
        prototypes = self._compute_prototypes(...)  # (B, C, 64)
        
        # 计算 SR 损失
        if self.args.use_supervise_prototype:
            sr_loss = self._compute_sr_loss(
                prototypes, support_feat, support_y
            )
        else:
            sr_loss = 0.0
        
        # 分类损失 + SR 损失
        total_loss = ce_loss + sr_weight * sr_loss
        
        return total_loss, acc
```

### _compute_sr_loss 详解

```python
def _compute_sr_loss(self, prototypes, support_feat, support_y):
    """
    Args:
        prototypes:  (B, C, D)   类别原型
        support_feat: (B, S, N, D) support 特征
        support_y:    (B, S)       support 标签
    
    Returns:
        sr_loss: scalar
    """
    B, C, D = prototypes.shape
    
    # Step 1: 用 decoder 重建原型
    reconstructed = self.reconstructor(prototypes)  # (B, C, D)
    
    # Step 2: 收集每个类别的真实 support 特征
    sr_loss = 0.0
    count = 0
    
    for c in range(C):
        # 找到属于类别 c 的 support block
        class_mask = (support_y == c)  # (B, S)
        
        if class_mask.sum() == 0:
            continue
        
        # 获取该类别的 support 特征
        class_feat = support_feat[class_mask]  # (n_blocks, N, D)
        class_feat = class_feat.view(-1, D)    # (n_blocks * N, D)
        
        # 重建的原型 (扩展以匹配)
        recon_c = reconstructed[:, c:c+1, :]  # (B, 1, D)
        recon_c = recon_c.expand(-1, class_feat.shape[0], -1)  # (B, M, D)
        recon_c = recon_c.reshape(-1, D)       # (B*M, D)
        
        # Step 3: 计算 MSE 损失
        loss_c = F.mse_loss(recon_c, class_feat)
        sr_loss += loss_c
        count += 1
    
    return sr_loss / max(count, 1)
```

---

## 5. SR 损失的直观理解

### 理想情况

```
原型 p_chair = [0.8, 0.1, -0.3, ...]  # 64维

Reconstructor:
  p_chair → MLP → p'_chair = [0.78, 0.12, -0.28, ...]

真实的 chair 点特征:
  z1 = [0.81, 0.09, -0.31, ...]
  z2 = [0.79, 0.11, -0.29, ...]
  z3 = [0.77, 0.13, -0.30, ...]

SR loss = mean(||p'_chair - z_i||^2) → 接近 0

说明: 原型包含了足够的信息来重建 chair 的特征
     → 原型确实是 chair 的好代表
```

### 退化情况

```
原型 p_chair = [0.5, 0.5, 0.5, ...]  # 全是均值，信息量低

Reconstructor 无法重建出多样的 chair 特征
SR loss 很高 → 惩罚原型
→ 促使原型包含更多判别性信息
```

---

## 6. SR vs 其他重建损失

| 方法 | 重建目标 | 损失 |
|---|---|---|
| **SR (PAP)** | 原型 → 原始特征 | MSE |
| Autoencoder | 输入 → 输出 | MSE/BCE |
| VAE | 输入 → 分布参数 | KL + MSE |
| Masked AE | 可见部分 → 掩码部分 | MSE |

SR 的特殊之处：
- 不是重建整个输入，而是验证原型的信息量
- 间接监督：通过原型质量保证分类效果
- 轻量：decoder 只有 2 层 MLP

---

## 7. SR 权重的影响

```python
# SR 损失的权重 (通常较小)
sr_weight = 0.1  # 或更小

# 权重实验
sr_weight = 0.0   → 等价于没有 SR
sr_weight = 0.01  → 轻微正则化
sr_weight = 0.1   → 推荐值，平衡分类和重建
sr_weight = 1.0   → 太强，原型过于保守，分类性能下降
```

### 权重选择的原则

```
比例: CE_loss ≈ 1.0~2.0 (few-shot 早期)
       SR_loss ≈ 0.01~0.05

sr_weight = 0.1:
  有效 SR = 0.1 × 0.05 = 0.005
  相对 CE: 0.005 / 1.5 ≈ 0.3%
  → SR 作为辅助任务，不主导训练
```

---

## 8. 本节总结

| 概念 | 要点 |
|---|---|
| SR 动机 | 原型仅靠分类损失可能退化 |
| SR 定义 | `MSE(decoder(prototype), support_features)` |
| Decoder | 2 层 MLP: 64→128→64 |
| 总损失 | CE + λ × SRLoss |
| 权重建议 | λ = 0.1 作为辅助监督 |

---

## 9. 课后练习

1. 在 `protonet_QGPA.py` 中添加 SR 损失的 TensorBoard 日志
2. 训练时分别记录 CE Loss 和 SR Loss，观察曲线的变化趋势
3. 尝试不同的 sr_weight (0.01, 0.05, 0.1, 0.5, 1.0)，绘制性能-sr_weight 图
4. 设计一个实验验证：SR 损失是否真的提高了原型的差异度（inter-class distance）
