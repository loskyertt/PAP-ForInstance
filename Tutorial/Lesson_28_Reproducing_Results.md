# 第28课：实验结果复现与可视化

## 1. 本节学习目标

- 学会复现论文中的完整实验结果
- 掌握 TensorBoard 的可视化技巧
- 能够分析和对比不同配置的实验结果
- 培养科学的实验管理习惯

---

## 2. 完整复现流程

### S3DIS 2-way 1-shot 完整复现

```bash
# === Step 1: 预处理 ===
# (如果有 S3DIS 原始数据)
python preprocess/collect_s3dis_data.py
python preprocess/room2blocks.py

# === Step 2: 预训练 ===
bash scripts/pretrain_segmentor.sh
# 检查输出: runs/pretrain_s3dis_fold0/best_model.pth

# === Step 3: 训练 PAP ===
bash scripts/train_PAP.sh
# 检查输出: runs/PAP_s3dis_fold0_2way1shot/best_model.pth

# === Step 4: 评估 ===
bash scripts/eval_PAP.sh
# 输出: Per-class IoU 和 Mean IoU

# === Step 5: 验证结果 ===
# 论文 S3DIS 2-way 1-shot mIoU: 约 54.8%
# 自己运行应在 ±2% 范围内
```

---

## 3. 实验管理最佳实践

### 目录结构

```
runs/
├── pretrain_s3dis_fold0/
│   ├── events.out.tfevents.xxx   # TensorBoard 文件
│   ├── best_model.pth            # 最佳模型
│   └── checkpoint_epoch_50.pth   # 定期 checkpoint
│
├── PAP_s3dis_fold0_2way1shot/
│   ├── events.out.tfevents.xxx
│   ├── best_model.pth
│   └── args.txt                   # 记录参数
│
├── PAP_s3dis_fold0_3way1shot/    # 其他配置
└── eval_PAP_s3dis_fold0/
    └── results.json               # 评估结果
```

### 参数记录

```python
# 训练开始时自动保存参数
def save_args(args, save_path):
    with open(os.path.join(save_path, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

# 或使用 JSON
import json
with open('runs/exp_config.json', 'w') as f:
    json.dump(vars(args), f, indent=2)
```

---

## 4. TensorBoard 使用指南

### 启动 TensorBoard

```bash
# 启动 (在项目根目录)
tensorboard --logdir runs/ --port 6006

# 访问: http://localhost:6006

# 多实验对比
tensorboard --logdir runs/ --port 6006 --bind_all
```

### 需要记录的标量 (Scalars)

```python
# 训练指标
writer.add_scalar('Train/Loss', train_loss, epoch)
writer.add_scalar('Train/Accuracy', train_acc, epoch)
writer.add_scalar('Train/LearningRate', lr, epoch)
writer.add_scalar('Train/GradNorm', grad_norm, epoch)

# 验证指标
writer.add_scalar('Val/Loss', val_loss, epoch)
writer.add_scalar('Val/Accuracy', val_acc, epoch)
writer.add_scalar('Val/MeanIoU', val_miou, epoch)

# FZ 特有
writer.add_scalar('Train/GMMNLoss', gmmn_loss, epoch)
writer.add_scalar('Train/SRLoss', sr_loss, epoch)
writer.add_scalar('Train/AlignLoss', align_loss, epoch)

# Per-class IoU
for c in range(num_classes):
    writer.add_scalar(f'Val/IoU_Class_{c}', per_class_iou[c], epoch)
```

### 直方图 (Histograms)

```python
# 记录参数分布
for name, param in model.named_parameters():
    writer.add_histogram(f'Params/{name}', param, epoch)
    if param.grad is not None:
        writer.add_histogram(f'Grads/{name}', param.grad, epoch)

# 记录原型分布
writer.add_histogram('Prototypes/original', original_prototypes, epoch)
writer.add_histogram('Prototypes/adapted', adapted_prototypes, epoch)
```

---

## 5. 结果对比与分析

### 消融实验结果表

```python
# 自动化收集结果
import json
import glob

def collect_results(exp_dirs):
    """收集所有实验的结果"""
    results = {}
    for exp_name, exp_dir in exp_dirs.items():
        result_file = os.path.join(exp_dir, 'results.json')
        if os.path.exists(result_file):
            with open(result_file) as f:
                results[exp_name] = json.load(f)
    
    # 打印对比表
    print(f"{'Experiment':30s} {'mIoU':8s}")
    print("-" * 40)
    for name, res in results.items():
        print(f"{name:30s} {res['mean_iou']:.4f}")
    
    return results

# 收集
exp_dirs = {
    'ProtoNet': 'runs/ProtoNet_s3dis_fold0',
    'PAP (w/o SR)': 'runs/PAP_noSR_s3dis_fold0',
    'PAP (full)': 'runs/PAP_s3dis_fold0_2way1shot',
}
collect_results(exp_dirs)
```

### 可视化对比

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_per_class_iou(results, class_names):
    """绘制各类别 IoU 的对比柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    colors = ['#4C72B0', '#55A868', '#C44E52']
    
    for i, (name, res) in enumerate(results.items()):
        ax.bar(x + i * width, res['per_class_iou'], 
               width, label=name, color=colors[i])
    
    ax.set_ylabel('IoU')
    ax.set_xlabel('Class')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.set_title('Per-Class IoU Comparison')
    
    plt.tight_layout()
    plt.savefig('per_class_iou_comparison.png', dpi=150)
```

---

## 6. 训练曲线诊断

### 正常曲线

```
正常训练的特征:
  - Loss 单调下降，逐渐趋于平稳
  - Val accuracy 先快后慢上升
  - Train 和 Val 之间的 gap 不大 (没有严重过拟合)
  - 梯度范数稳定在 0.1~10 之间
```

### 异常曲线诊断

| 现象 | 可能原因 | 解决方案 |
|---|---|---|
| Loss 不下降 | 学习率太小 | 增大 lr |
| Loss 震荡剧烈 | 学习率太大 | 减小 lr |
| Train acc 高, Val acc 低 | 过拟合 | 增加 episode 多样性, 减小模型 |
| Loss → NaN | 梯度爆炸 | 添加 grad clip, 减小 lr |
| Val acc 波动大 | episode 太少 | 增加 episodes_per_epoch 或 n_episodes_test |
| Acc 始终很低 | 模型容量不足 | 增大 output_dim, 使用 QGPA |

---

## 7. 论文复现 Checklist

```
□ 数据预处理完成
  □ collect_s3dis_data.py (或 collect_scannet_data.py)
  □ room2blocks.py
  □ class2scans.pkl 生成正确

□ 预训练完成
  □ base class 准确率 > 80%
  □ encoder checkpoint 保存

□ 元训练完成
  □ 所有 flags 与论文一致
  □ SR/Align/GMMN 损失正常收敛
  □ val accuracy 持续提升

□ 评估完成
  □ 使用独立的 test episodes
  □ 结果在论文 ±2% 范围内
  □ per-class IoU 分布合理
```

---

## 8. 本节总结

| 工具 | 用途 |
|---|---|
| TensorBoard | 实时监控训练曲线 |
| args.txt / JSON | 记录实验配置 |
| results.json | 保存评估结果 |
| Matplotlib | 论文级可视化 |
| Shell 脚本 | 自动化实验流水线 |

---

## 9. 课后练习

1. 用 TensorBoard 对比两组实验（有/无 QGPA）的训练曲线
2. 编写 Python 脚本自动收集所有实验结果并生成对比表格
3. 绘制不同 k_shot (1, 5, 10) 下的 per-class IoU 雷达图
4. 实现一个函数，自动判断训练是否异常并发送通知
