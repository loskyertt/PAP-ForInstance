# 第17课：自注意力机制 (Self-Attention)

## 1. 本节学习目标

- 理解 Self-Attention 在点云中的应用
- 掌握 Scaled Dot-Product Attention 的原理
- 能够阅读和理解 `models/attention.py` 中的 SelfAttention 类
- 理解注意力机制如何增强点云特征

---

## 2. 为什么要在点云中使用注意力

### 点云特征的问题

```
DGCNN 输出:
  每个点的特征包含局部几何信息
  但点与点之间的关系是"间接"的（通过多层传递）
  
注意力机制:
  让每个点"直接看到"所有其他点
  → 可以建模全局依赖关系
```

### 直观理解

```
没有注意力:
  点 A  ---> 局部邻域 A  ---> 局部邻域 B  ---> 点 B
  (信息通过 2-3 跳传播)

有注意力:
  点 A  ←────── 注意力加权 ──────→  点 B
  (信息直接传播，一步到位)
```

---

## 3. Scaled Dot-Product Attention 原理

### 标准公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 直观解释

```python
# 伪代码
def attention(Q, K, V):
    # Q (Query):  每个点"问"其他点——我想知道什么？
    # K (Key):    每个点"回答"其他点——我有什么？
    # V (Value):  每个点的实际内容
    
    # Step 1: 计算注意力分数
    scores = Q @ K.T / sqrt(d_k)       # (N, N)  # 每对点的相关性
    
    # Step 2: Softmax 归一化
    weights = softmax(scores, dim=-1)  # (N, N)  # 每行和为 1
    
    # Step 3: 加权聚合
    output = weights @ V               # (N, d_v) # 信息聚合
    
    return output
```

### 在点云中的含义

```
Q_i @ K_j  =  点 i 和点 j 的特征相似度
           =  "点 j 对点 i 有多相关"

softmax(Q_i @ K / sqrt(d)) = 注意力权重分布

加权 V_j  = 从所有点聚合信息
```

---

## 4. SelfAttention 类代码解读

### 完整实现

```python
# models/attention.py
class SelfAttention(nn.Module):
    """
    点云中的自注意力机制
    让每个点都能"关注"到其他点
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Q, K, V 的投影
        self.q_conv = nn.Conv1d(channels, channels // 8, 1)
        self.k_conv = nn.Conv1d(channels, channels // 8, 1)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        # 输出投影
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        x: (B, N, C)  输入特征
        返回: (B, N, C)  增强特征
        """
        B, N, C = x.shape
        x_t = x.transpose(1, 2)  # (B, C, N)
        
        # Step 1: 计算 Q, K, V
        q = self.q_conv(x_t).transpose(1, 2)  # (B, N, C//8)
        k = self.k_conv(x_t).transpose(1, 2)  # (B, N, C//8)
        v = self.v_conv(x_t).transpose(1, 2)  # (B, N, C)
        
        # Step 2: 计算注意力
        energy = torch.bmm(q, k.transpose(1, 2))  # (B, N, N)
        scale = (self.channels // 8) ** 0.5
        attention = self.softmax(energy / scale)  # (B, N, N)
        
        # Step 3: 加权聚合
        out = torch.bmm(attention, v)  # (B, N, C)
        
        # Step 4: 残差连接
        out = out + x
        
        return out
```

### 张量形状追踪

```
输入 x:        (B, N, C=64)

Q, K:          (B, N, C//8=8)   # 降维减少计算量
V:             (B, N, C=64)     # 保持原始维度

energy (QK^T): (B, N, N)        # 每对点的相似度矩阵
attention:     (B, N, N)        # softmax 归一化

output:        (B, N, C=64)     # 注意力输出
+ x (残差):    (B, N, C=64)     # 最终输出
```

---

## 5. 设计细节分析

### 为什么 Q, K 用 C//8 通道？

```
原始通道 C=64
Q, K 用 C//8=8 通道

好处:
  1. 减少 QK^T 的计算量: 64×64 → 8×8 (64x 加速)
  2. 减少参数量
  3. 类似于 Multi-Head 的"瓶颈"设计

Trade-off:
  太小的瓶颈可能丢失信息，8 是一个经验平衡点
```

### 残差连接的作用

```python
# 残差连接
out = attention_out + x

# 为什么?
# 1. 梯度流: 即使 attention 不好，梯度仍可通过捷径回传
# 2. 特征保持: attention 增强了特征但不破坏原始信息
# 3. 训练稳定: 残差使 attention 学习"残差"而非"完全替换"
```

### 自注意力 vs 卷积

| 特性 | Conv1d | SelfAttention |
|---|---|---|
| 感受野 | 局部 (kernel_size) | 全局 (N × N) |
| 权重 | 固定（空间位置） | 动态（基于特征） |
| 参数量 | C_in × C_out × k | 3 × C × C/8 (较少) |
| 计算复杂度 | O(N × C²) | O(N² × C) |
| 适用场景 | 局部模式 | 全局依赖 |

---

## 6. 注意力权重可视化

```python
# 提取注意力权重
def visualize_attention(model, x):
    with torch.no_grad():
        # 注入 hook 捕获 attention matrix
        attention_matrix = None
        def hook_fn(module, input, output):
            nonlocal attention_matrix
            # 手动计算 attention
            q = model.q_conv(x.transpose(1,2)).transpose(1,2)
            k = model.k_conv(x.transpose(1,2)).transpose(1,2)
            energy = torch.bmm(q, k.transpose(1,2))
            attention_matrix = F.softmax(energy / 8**0.5, dim=-1)
        
        handle = model.attention.register_forward_hook(hook_fn)
        out = model(x)
        handle.remove()
        
        return attention_matrix[0].cpu().numpy()  # (N, N)

# 可视化
attn = visualize_attention(model, point_cloud)
plt.imshow(attn[:100, :100])  # 前 100 个点的注意力
plt.colorbar()
plt.title('Self-Attention Matrix (first 100 points)')
```

---

## 7. 本节总结

| 概念 | 要点 |
|---|---|
| Q, K, V | Query(查询), Key(键), Value(值) 投影 |
| Scaled Dot-Product | `softmax(QK^T/√d) V` |
| 自注意力 | Q, K, V 都来自同一输入 → 自己关注自己 |
| 残差连接 | 输出 = attention_out + x |
| 复杂度 | O(N² × C)，N=2048 时约 4M 次操作 |

---

## 8. 课后练习

1. 在 SelfAttention.forward() 中打印 attention matrix 的形状和对角线值
2. 对比有/无残差连接的训练稳定性
3. 修改 Q, K 的通道倍数（C/4, C/2, C），测试对性能/速度的影响
4. 实现 Multi-Head Self Attention（head=4），对比与单头版本的区别
