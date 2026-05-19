# 第25课：Episode 训练策略与超参数

## 1. 本节学习目标

- 掌握元训练（Meta-Training）的超参数调优技巧
- 理解 Episode 数量、验证频率、模型保存策略
- 掌握学习率调度和梯度裁剪
- 能够根据 GPU 资源调整训练配置

---

## 2. 元训练的超参数全景

| 参数 | 默认值 | 作用 | 调优建议 |
|---|---|---|---|
| `lr` | 0.001 | 学习率 | 0.0005~0.002 |
| `epochs` | 50 | 训练轮数 | 30~100 |
| `episodes_per_epoch` | 100 | 每轮 episode 数 | 50~500 |
| `eval_interval` | 5 | 验证频率 (epochs) | 1~10 |
| `n_way` | 2 | 每 episode 类数 | 1~6 |
| `k_shot` | 1 | 每类 support 数 | 1,5 |
| `pc_npts` | 2048 | 采样点数 | 512~4096 |
| `grad_clip` | — | 梯度裁剪 | 1.0~5.0 |

---

## 3. Episode 训练策略

### Episode 数量与泛化

```
每个 epoch 的 episode 数量决定模型看到的"任务多样性":

episodes_per_epoch = 100:
  - 每轮看到 100 个不同的 (n_way, k_shot) 组合
  - 适合快速验证 (开发阶段)
  
episodes_per_epoch = 500:
  - 更多样化的任务 → 更好的泛化
  - 适合最终训练
  
episodes_per_epoch = 2000:
  - 最大多样性
  - 可能过拟合 episode 分布（反而不利于泛化）
```

### 验证策略

```python
# 推荐配置
if args.debug:  # 开发阶段
    eval_interval = 1
    episodes_per_epoch = 20
    epochs = 5

elif args.fast:  # 快速实验
    eval_interval = 5
    episodes_per_epoch = 100
    epochs = 30

else:  # 完整训练
    eval_interval = 5
    episodes_per_epoch = 200
    epochs = 50
```

---

## 4. 学习率调度策略

### 阶梯衰减 (StepLR)

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.5
)
# lr: 0.001 → 0.0005 (epoch 20) → 0.00025 (epoch 40)
```

### 余弦退火 (CosineAnnealingLR)

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
# 平滑下降，适合最终精细调优
```

### 分层学习率

```python
# encoder 用小学习率 (预训练权重不能大幅改变)
# attention/QGPA 用大学习率 (随机初始化)
optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': args.lr * 0.1},
    {'params': model.attention.parameters(), 'lr': args.lr},
    {'params': model.transformer.parameters(), 'lr': args.lr},
    {'params': model.base_learner.parameters(), 'lr': args.lr},
])
```

---

## 5. 梯度裁剪

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# 在 train_epoch 中的应用
for batch in dataloader:
    loss = model(...)
    optimizer.zero_grad()
    loss.backward()
    
    # 裁剪梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    
    optimizer.step()
```

---

## 6. 训练监控与调试

### 关键指标

```python
# 每 N 个 episode 记录
self.writer.add_scalar('Train/Loss', loss, global_step)
self.writer.add_scalar('Train/Accuracy', acc, global_step)

# 每 eval_interval 验证
self.writer.add_scalar('Val/Loss', val_loss, epoch)
self.writer.add_scalar('Val/Accuracy', val_acc, epoch)

# 学习率
self.writer.add_scalar('Train/LR', 
    optimizer.param_groups[0]['lr'], epoch)

# 梯度范数
total_norm = sum(p.grad.norm()**2 for p in model.parameters())**0.5
self.writer.add_scalar('Train/GradNorm', total_norm, global_step)
```

### Watch Dog (训练异常检测)

```python
def check_training_health(loss, acc, grad_norm):
    """检测训练是否异常"""
    warnings = []
    
    if loss > 10.0:
        warnings.append("Loss too high")
    if acc < 0.2 and epoch > 10:
        warnings.append("Accuracy stuck low")
    if grad_norm > 100:
        warnings.append("Gradient explosion")
    if torch.isnan(loss):
        warnings.append("NaN loss detected!")
    
    for w in warnings:
        print(f"⚠️ {w}")
    
    return len(warnings) == 0
```

---

## 7. Checkpoint 管理

### 保存策略

```python
def manage_checkpoints(save_dir, max_keep=5):
    """
    保留最近 max_keep 个 checkpoint，删除旧的
    """
    ckpts = sorted(glob.glob(f'{save_dir}/*.pth'))
    if len(ckpts) > max_keep:
        for ckpt in ckpts[:-max_keep]:
            os.remove(ckpt)
            print(f"Removed old checkpoint: {ckpt}")
```

### Checkpoint 内容

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    'acc': acc,
    'args': args,  # 记录训练参数，便于复现
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
```

---

## 8. 资源管理

### GPU 内存优化

```python
# 1. 减少采样点数
--pc_npts 1024   # 从 2048 减少

# 2. 减少 episode 中的类数
--n_way 2        # 从 5 减少

# 3. 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    loss = model(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 4. 梯度累积 (模拟大 batch)
accumulation_steps = 4
for i, data in enumerate(dataloader):
    loss = model(...) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 9. 本节总结

| 超参数 | 推荐范围 | 影响 |
|---|---|---|
| lr | 0.0005~0.002 | 过高震荡，过低收敛慢 |
| episodes/epoch | 100~500 | 高→泛化好但慢 |
| eval_interval | 1~5 | 频繁验证便于观察 |
| grad_clip | 1~5 | 防止梯度爆炸 |
| pc_npts | 1024~2048 | 高→精细但吃内存 |
| 分层 lr | encoder×0.1 | 保护预训练知识 |

---

## 10. 课后练习

1. 设计一个网格搜索实验：lr ∈ {0.0005, 0.001, 0.002} × episodes_per_epoch ∈ {50, 200, 500}
2. 实现余弦退火学习率调度，对比与 StepLR 的差异
3. 在训练中添加梯度范数监控，找出梯度爆炸的 episode
4. 使用 TensorBoard 对比 3 种不同训练策略的验证曲线
