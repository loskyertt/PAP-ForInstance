# 第26课：评估指标 — IoU 计算与解释

## 1. 本节学习目标

- 深入理解语义分割的评估标准 IoU
- 掌握 Per-class IoU 和 Mean IoU (mIoU) 的区别
- 能够阅读 `runs/eval.py` 中的 IoU 代码
- 理解混淆矩阵与 IoU 的关系

---

## 2. IoU 定义

### Intersection over Union

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$

其中：
- $A$ 是预测为该类的点集
- $B$ 是真实属于该类的点集
- $TP$ (True Positive)：预测正确
- $FP$ (False Positive)：预测为该类，实际不是
- $FN$ (False Negative)：实际是该类，预测为其他

### 直观理解

```
真实:     [● ● ○ ○ ○ ● ● ●]  (4 个点属于 class A)
预测:     [● ○ ● ○ ○ ● ○ ●]  (4 个点被预测为 class A)

TP = 2  (位置 1 和 6，预测对)
FP = 2  (位置 3 和 7，预测为 A 但不是)
FN = 2  (位置 2 和 8，是 A 但被预测为其他)

IoU = 2 / (2 + 2 + 2) = 0.333
```

---

## 3. IoU 代码实现

### runs/eval.py 中的实现

```python
def compute_iou(predictions, labels, num_classes):
    """
    Args:
        predictions: (N,)  预测类别
        labels:      (N,)  真实标签
        num_classes: int   类别总数
    
    Returns:
        per_class_iou: (num_classes,)
        mean_iou:      scalar
    """
    ious = []
    
    for cls in range(num_classes):
        # 属于该类的预测
        pred_mask = (predictions == cls)
        # 属于该类的真实标签
        label_mask = (labels == cls)
        
        # 交集: 预测为 cls 且真是 cls
        intersection = (pred_mask & label_mask).sum().float()
        
        # 并集: 预测为 cls 或真是 cls
        union = (pred_mask | label_mask).sum().float()
        
        if union == 0:
            ious.append(float('nan'))  # 该类别不存在
        else:
            ious.append(intersection / union)
    
    ious = torch.tensor(ious)
    
    # mean IoU: 忽略不存在的类
    valid = ~torch.isnan(ious)
    mean_iou = ious[valid].mean()
    
    return ious, mean_iou
```

### 完整的评估流程

```python
# runs/eval.py
def evaluate(model, test_loader, num_classes):
    all_predictions = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for support_x, support_y, query_x, query_y in test_loader:
            # 少样本推理
            support_feat = model.encoder(support_x)
            prototypes = compute_prototypes(support_feat, support_y)
            
            query_feat = model.encoder(query_x)
            similarity = cosine_similarity(query_feat, prototypes)
            predictions = similarity.argmax(dim=-1)
            
            all_predictions.append(predictions.view(-1))
            all_labels.append(query_y.view(-1))
    
    # 计算 IoU
    all_preds = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    per_class_iou, mean_iou = compute_iou(
        all_preds, all_labels, num_classes
    )
    
    return per_class_iou, mean_iou
```

---

## 4. Per-class IoU vs Mean IoU

### 为什么要分开看

```
数据集不平衡时:
  Class A (wall):     10000 点 → IoU=0.95
  Class B (chair):    200 点   → IoU=0.40
  Class C (column):   50 点    → IoU=0.10

整体 accuracy = (9500+80+5) / 10250 = 93.5%  (看起来很好!)
mean IoU = (0.95+0.40+0.10) / 3 = 48.3%     (揭示问题)

→ mean IoU 对稀有类更公平
```

### 输出格式

```python
def print_results(per_class_iou, mean_iou, class_names):
    print("=" * 50)
    print("Per-class IoU:")
    for i, name in enumerate(class_names):
        if not torch.isnan(per_class_iou[i]):
            print(f"  {name:15s}: {per_class_iou[i]:.4f}")
        else:
            print(f"  {name:15s}: N/A")
    print("-" * 50)
    print(f"Mean IoU: {mean_iou:.4f}")
    print("=" * 50)
```

---

## 5. 混淆矩阵

```python
def compute_confusion_matrix(predictions, labels, num_classes):
    """
    返回 (num_classes, num_classes) 的混淆矩阵
    confusion[i, j]: 真实为 i, 预测为 j 的点数
    """
    confusion = torch.zeros(num_classes, num_classes)
    
    for t, p in zip(labels, predictions):
        confusion[t.long(), p.long()] += 1
    
    return confusion

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt

cm = compute_confusion_matrix(preds, labels, num_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm / cm.sum(dim=1, keepdim=True), 
            annot=True, fmt='.2f',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

---

## 6. 评估中的特殊考虑

### 背景类处理

```python
# S3DIS 中的 clutter 类
# 通常被排除在计算之外
valid_classes = [0, 1, 2, 3, 4, 5]  # 不包括 clutter
per_class_iou = per_class_iou[valid_classes]
mean_iou = per_class_iou[valid_classes].mean()
```

### 多 episode 聚合

```python
# 测试多个 episode 时的正确聚合方式
# 错误: 每个 episode 算 IoU 然后平均
# 正确: 聚合所有 prediction→label 对，然后统一算 IoU

# 具体操作
total_intersection = torch.zeros(num_classes)
total_union = torch.zeros(num_classes)

for episode in test_loader:
    preds, labels = model(episode)
    for c in range(num_classes):
        total_intersection[c] += (preds == c & labels == c).sum()
        total_union[c] += (preds == c | labels == c).sum()

per_class_iou = total_intersection / (total_union + 1e-10)
mean_iou = per_class_iou[total_union > 0].mean()
```

---

## 7. 论文中的评估报告

```
PAP-FZS3D 论文典型的评估表格:

S3DIS, 2-way 1-shot:
┌────────────┬──────────┬──────────┐
│ Class      │ IoU(%)   │          │
├────────────┼──────────┼──────────┤
│ beam       │  48.3    │          │
│ board      │  45.7    │          │
│ bookcase   │  51.2    │          │
│ ceiling    │  85.1    │          │
│ chair      │  55.4    │          │
│ column     │  42.9    │          │
├────────────┼──────────┼──────────┤
│ mean IoU   │  54.8    │          │
└────────────┴──────────┴──────────┘
```

---

## 8. 本节总结

| 指标 | 公式 | 意义 |
|---|---|---|
| IoU | TP/(TP+FP+FN) | 单类分割精度 |
| mIoU | mean(per-class IoU) | 整体分割性能（公平考虑各类） |
| Accuracy | TP+TN / Total | 容易被大类主导 |
| Confusion Matrix | C×C | 看出混淆模式 |

---

## 9. 课后练习

1. 实现 `compute_iou` 函数并用模拟数据测试（含极端情况）
2. 用混淆矩阵分析：哪些类别在少样本设置下最容易混淆
3. 计算同一模型在不同 episode 上的 IoU 标准差，评估模型稳定性
4. 实现 F1-score 和 Dice coefficient，对比与 IoU 的关系
