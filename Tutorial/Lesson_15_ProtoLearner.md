# 第15课：ProtoLearner — 训练与测试循环

## 1. 本节学习目标

- 理解 ProtoLearner 如何组织训练和评估
- 掌握 train() 和 test_few_shot() 的完整流程
- 理解 Episode 训练中 loss 和 accuracy 的计算
- 能够修改和扩展训练逻辑

---

## 2. ProtoLearner 架构

### 类定义

```python
# models/proto_learner.py (简化)
class ProtoLearner:
    """
    封装了 prototypical network 的训练和测试逻辑
    """
    def __init__(self, args, model, optimizer=None):
        self.args = args
        self.model = model              # ProtoNet 或 ProtoNetAlignQGPASR
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), lr=args.lr
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, train_loader, val_loader, epochs):
        """完整训练流程"""
        ...
    
    def train_epoch(self, dataloader):
        """单个 epoch 的训练"""
        ...
    
    def test_few_shot(self, dataloader):
        """在验证/测试集上评估"""
        ...
```

---

## 3. train() — 完整训练流程

```python
def train(self, train_loader, val_loader, epochs):
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # ===== 训练阶段 =====
        self.model.train()
        train_loss, train_acc = self.train_epoch(train_loader)
        
        # ===== 日志 =====
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        
        # ===== 验证阶段 =====
        if epoch % self.args.eval_interval == 0:
            self.model.eval()
            with torch.no_grad():
                val_loss, val_acc = self.test_few_shot(val_loader)
            
            print(f"  Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            
            # ===== 保存最佳模型 =====
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc, 'best_model.pth')
            
            # TensorBoard 记录
            self.writer.add_scalar('Val/Acc', val_acc, epoch)
    
    print(f"Training done. Best val acc: {best_val_acc:.4f}")
```

---

## 4. train_epoch() — 单轮训练

```python
def train_epoch(self, dataloader):
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(dataloader):
        # 数据移到 GPU
        support_x = support_x.cuda()
        support_y = support_y.cuda()
        query_x = query_x.cuda()
        query_y = query_y.cuda()
        
        # 前向传播
        loss, acc = self.model(
            support_x, support_y, query_x, query_y
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 累积统计
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches
```

### 关键细节

```python
# 数据形状
# support_x: (B=1, S=N*K, pc_npts=2048, dim=9)
# support_y: (B=1, S=N*K)
# query_x:   (B=1, Q=N*Q, pc_npts=2048, dim=9)
# query_y:   (B=1, Q=N*Q)

# batch_size=1 时，实际上每个 episode 包含了多个点云块
# 模型内部处理了 S+Q 个块的批处理
```

---

## 5. test_few_shot() — 评估函数

```python
def test_few_shot(self, dataloader):
    """
    在验证集上评估少样本性能
    """
    total_loss = 0.0
    total_acc = 0.0
    all_predictions = []
    all_labels = []
    num_episodes = 0
    
    for support_x, support_y, query_x, query_y in dataloader:
        support_x = support_x.cuda()
        support_y = support_y.cuda()
        query_x = query_x.cuda()
        query_y = query_y.cuda()
        
        # 前向传播（不计算梯度）
        loss, pred, labels = self.model.forward_with_pred(
            support_x, support_y, query_x, query_y
        )
        
        total_loss += loss.item()
        total_acc += (pred == labels).float().mean().item()
        
        all_predictions.append(pred.cpu())
        all_labels.append(labels.cpu())
        num_episodes += 1
    
    # 计算详细指标
    all_preds = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    ious = self._compute_iou(all_preds, all_labels)
    
    return total_loss / num_episodes, total_acc / num_episodes, ious
```

---

## 6. 优化器与学习率调度

```python
# 不同的参数组使用不同的学习率（可选）
def _setup_optimizer(self):
    # 方法1: 统一学习率
    optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
    
    # 方法2: 分层学习率
    optimizer = torch.optim.Adam([
        {'params': self.model.encoder.parameters(), 'lr': args.lr * 0.1},
        {'params': self.model.base_learner.parameters(), 'lr': args.lr},
        {'params': self.model.attention.parameters(), 'lr': args.lr},
    ])
    
    # 学习率衰减
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )
    
    return optimizer, scheduler
```

### 为什么 encoder 用更小的学习率？

```
预训练的 encoder 已经学好了通用的几何特征
→ 大幅修改会破坏已有的知识
→ 用小学习率微调即可
→ base_learner 和 attention 是随机初始化的
→ 需要用正常学习率
```

---

## 7. train() vs test_few_shot() 对比

| 方面 | train() | test_few_shot() |
|---|---|---|
| 模型模式 | `model.train()` | `model.eval()` |
| 梯度 | 计算并回传 | `torch.no_grad()` |
| BatchNorm | 使用 batch 统计 | 使用全局统计 |
| 数据增强 | 可能开启 | 关闭 |
| Episode 数量 | `episodes_per_epoch` (如 100) | 全部预生成 (如 1000) |
| 指标 | Loss + Acc | Loss + Acc + IoU |

---

## 8. 实践：自定义训练监控

```python
# 在 train_epoch 中添加监控
def train_epoch_with_monitor(self, dataloader):
    for batch_idx, data in enumerate(dataloader):
        ...
        
        # 每 10 个 batch 打印梯度范数
        if batch_idx % 10 == 0:
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Batch {batch_idx}: Grad Norm = {total_norm:.4f}")
        
        # 每 50 个 batch 保存一次 checkpoint
        if batch_idx % 50 == 0:
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'batch': batch_idx
            }, f'checkpoint_batch_{batch_idx}.pth')
```

---

## 9. 本节总结

| 函数 | 职责 | 调用频率 |
|---|---|---|
| `train()` | 组织 epoch 循环，验证，保存模型 | 一次 |
| `train_epoch()` | 遍历 episode，更新参数 | 每 epoch 一次 |
| `test_few_shot()` | 在验证/测试集评估 | 每 N epoch 一次 |
| `_save_checkpoint()` | 保存模型 state_dict | 当 val_acc 提升时 |

---

## 10. 课后练习

1. 在 ProtoLearner 中添加学习率 warmup 逻辑（前 5 个 epoch 线性增长）
2. 修改 train_epoch()，让每个 episode 打印 support 与 query 特征的余弦相似度分布
3. 对比有/无 encoder 低学习率策略的验证曲线
4. 实现一个 early stopping 机制：如果 val_acc 连续 10 epoch 不提升就停止训练
