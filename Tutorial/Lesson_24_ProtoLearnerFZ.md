# 第24课：ProtoLearnerFZ — 联合少样本与零样本

## 1. 本节学习目标

- 理解 ProtoLearnerFZ 与 ProtoLearner 的区别
- 掌握零样本训练中 seen/unseen 类的处理逻辑
- 理解 GMMN 损失与分类损失的平衡
- 能够运行和理解 `train_PAPFZ.sh` 脚本

---

## 2. 两阶段对比

| 特性 | ProtoLearner | ProtoLearnerFZ |
|---|---|---|
| 处理类 | 全部 novel (少样本) | seen (少样本) + unseen (零样本) |
| 模型类 | ProtoNetAlignQGPASR | ProtoNetAlignFZ |
| 原型来源 | 仅 support 特征 | support 特征 + GMMN 生成器 |
| 额外损失 | SR + Align | SR + Align + GMMN |
| 词嵌入 | 不需要 | 必需 |

---

## 3. ProtoLearnerFZ.train() 核心流程

```python
# models/proto_learner_FZ.py (简化)
class ProtoLearnerFZ(ProtoLearner):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.seen_classes = args.seen_classes    # 有 support 样本的类
        self.unseen_classes = args.unseen_classes # 没有 support 样本的类
        self.all_classes = self.seen_classes + self.unseen_classes
    
    def train_epoch(self, dataloader):
        """
        每 episode: 随机划分 seen/unseen
        """
        for support_x, support_y, query_x, query_y in dataloader:
            # 数据移到 GPU
            ...
            
            # 获取所有类的词嵌入
            word_embeddings = self._get_word_embeddings(self.all_classes)
            
            # 前向传播 (包含 GMMN 生成)
            loss, ce_loss, gmmn_loss, acc = self.model(
                support_x, support_y,
                query_x, query_y,
                word_embeddings,
                seen_indices,
                unseen_indices
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

---

## 4. 前向传播中的 Seen/Unseen 处理

```python
# models/protonet_FZ.py (简化)
class ProtoNetAlignFZ(ProtoNetAlignQGPASR):
    def forward(self, support_x, support_y, query_x, query_y,
                word_embeddings, seen_idx, unseen_idx):
        
        # ===== Step 1: 编码 (与 PAP 相同) =====
        feat = self.encoder(all_x)
        feat = self.base_learner(feat)
        support_feat, query_feat = split(feat)
        
        # ===== Step 2: 计算 seen 类的真实原型 =====
        real_prototypes = self._compute_prototypes(
            support_feat, support_y
        )  # (B, |seen|, D)
        
        # ===== Step 3: GMMN 生成所有类的伪原型 =====
        noise = torch.randn(B, len(self.all_classes), self.noise_dim)
        fake_prototypes = self.generator(word_embeddings, noise)
        # (B, |all|, D)
        
        # ===== Step 4: 计算 GMMN 损失 (只对 seen 类) =====
        seen_fake = fake_prototypes[:, seen_idx]  # 取出 seen 的生成结果
        gmmn_loss = self.gmmn_loss(real_prototypes, seen_fake)
        
        # ===== Step 5: 构建最终原型集 =====
        final_prototypes = self._build_prototypes(
            real_prototypes, fake_prototypes, seen_idx, unseen_idx
        )
        # final_prototypes: (B, |all|, D)
        # seen 位置: 用真实原型 (从 support 计算)
        # unseen 位置: 用生成原型 (从词嵌入生成)
        
        # ===== Step 6: QGPA 自适应 =====
        if self.use_transformer:
            final_prototypes, attn = self.transformer(
                query_feat, final_prototypes
            )
        
        # ===== Step 7: 分类损失 =====
        similarity = cosine(query_feat, final_prototypes)
        ce_loss = F.cross_entropy(similarity, query_labels)
        
        # ===== Step 8: 辅助损失 =====
        align_loss = self._compute_align_loss(real_prototypes, ...)
        sr_loss = self._compute_sr_loss(real_prototypes, ...)
        
        # ===== Step 9: 总损失 =====
        total_loss = ce_loss \
                   + self.sr_weight * sr_loss \
                   + self.align_weight * align_loss \
                   + self.gmmn_weight * gmmn_loss
        
        return total_loss, ce_loss, gmmn_loss, acc
```

---

## 5. _build_prototypes() — Seen/Unseen 混合

```python
def _build_prototypes(self, real_prototypes, fake_prototypes, 
                      seen_idx, unseen_idx):
    """
    构建最终的混合原型集
    
    Args:
        real_prototypes: (B, |seen|, D)    从 support 计算
        fake_prototypes: (B, |all|, D)     GMMN 生成
        seen_idx:        list              有 support 的类索引
        unseen_idx:      list              零样本的类索引
    
    Returns:
        hybrid_prototypes: (B, |all|, D)
    """
    B, _, D = fake_prototypes.shape
    n_all = len(seen_idx) + len(unseen_idx)
    
    hybrid = torch.zeros(B, n_all, D).cuda()
    
    # 对于 seen 类: 使用真实原型
    hybrid[:, seen_idx] = real_prototypes
    
    # 对于 unseen 类: 使用生成原型
    hybrid[:, unseen_idx] = fake_prototypes[:, unseen_idx]
    
    return hybrid
```

---

## 6. 训练策略

### GMMN 权重的动态调整

```python
# 训练早期: GMMN 需要大权重 (生成器还没学好)
# 训练后期: GMMN 权重可以减小 (生成器已收敛)

# 方法1: 固定权重
gmmn_weight = 0.1

# 方法2: 余弦退火
gmmn_weight = 0.5 * (1 + math.cos(epoch / total_epochs * math.pi))

# 方法3: 阶段策略
if epoch < 10:
    gmmn_weight = 0.5   # 早期强监督
elif epoch < 30:
    gmmn_weight = 0.1   # 正常
else:
    gmmn_weight = 0.05  # 弱化
```

### 评估时的处理

```python
def test_few_shot_FZ(self, dataloader):
    for support_x, support_y, query_x, query_y in dataloader:
        # 评估时不需要噪声 —— 生成器用 0 噪声
        noise = torch.zeros(B, n_all, noise_dim)
        
        fake_prototypes = self.generator(word_embeddings, noise)
        
        # 注意: 评估时 unseen 类只能用生成的原型
        # query 中可能包含 unseen 类的点
        ...
```

---

## 7. 脚本解析：train_PAPFZ.sh

```bash
# scripts/train_PAPFZ.sh

# Step 1: 生成词嵌入 (首次需要)
python dataloaders/get_embedding.py --dataset S3DIS

# Step 2: 训练
python main.py \
    --phase prototrain \
    --dataset S3DIS \
    --cvfold 0 \
    --n_way 4 \              # 比 PAP 更多类 (seen + unseen)
    --k_shot 1 \
    --n_queries 1 \
    --use_high_dgcnn \
    --use_attention \
    --use_transformer \
    --use_align \
    --use_supervise_prototype \
    --use_linear_proj \
    --use_zero \             # ★ 开启 FZ 模式
    --gmmn_weight 0.1 \
    --noise_dim 300 \
    --episodes_per_epoch 200
```

### 与 train_PAP.sh 的差异

| 参数 | PAP | PAP-FZ |
|---|---|---|
| `n_way` | 2 | 4~6 (含 seen+unseen) |
| `use_zero` | ❌ | ✅ |
| `gmmn_weight` | — | 0.1 |
| `noise_dim` | — | 300 |
| 预处理 | 无 | get_embedding.py |
| 模型类 | ProtoNetAlignQGPASR | ProtoNetAlignFZ |

---

## 8. 本节总结

| 概念 | 要点 |
|---|---|
| FZ 模式 | `--use_zero` 启用 |
| 原型混合 | seen → 真实原型, unseen → GMMN 生成 |
| GMMN 权重 | 0.1，与 CE 平衡 |
| 词嵌入 | 必须预生成 (get_embedding.py) |
| 评估 | 生成器用零噪声，保证确定性 |

---

## 9. 课后练习

1. 运行 `train_PAPFZ.sh`，记录 GMMN 损失的收敛曲线
2. 对比 seen-only 和 seen+unseen 配置下的分类准确率
3. 尝试在评估时用不同噪声种子多次采样，观察结果方差
4. 设计实验：固定 GMMN 权重，改变 seen/unseen 比例，观察对 unseen 性能的影响
