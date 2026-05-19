# 第7课：Episode 采样机制

## 1. 本节学习目标

- 理解 N-way K-shot Episode 的完整构建过程
- 掌握 MyDataset 类的设计
- 理解 support/query 划分与数据增强的区别
- 能够追踪一次 __getitem__ 的完整调用

---

## 2. 为什么需要 Episode 采样

常规监督学习的 DataLoader：

```python
for x, y in dataloader:
    loss = model(x, y)  # x 和 y 一一对应
```

少样本学习的 DataLoader：

```python
for support_x, support_y, query_x, query_y in episode_loader:
    # support: 用于"学习"（计算原型）
    # query:   用于"测试"（计算损失）
    prototypes = compute_prototypes(model(support_x), support_y)
    predictions = model(query_x) @ prototypes.T
    loss = CrossEntropy(predictions, query_y)
```

**根本区别**：少样本学习中，每次迭代都在模拟一个"小规模学习任务"。

---

## 3. Episode 的结构

一个完整的 Episode = 一个 N-way K-shot 任务：

```
Episode (N=2, K=1, Q=1):
┌──────────────────────────────────────┐
│ Support Set (有标签)                  │
│   类 A: [样本A1]                      │  ← K=1 个样本
│   类 B: [样本B1]                      │  ← K=1 个样本
├──────────────────────────────────────┤
│ Query Set (无标签，用于计算损失)        │
│   类 A: [样本A1_q]                    │  ← Q=1 个样本
│   类 B: [样本B1_q]                    │  ← Q=1 个样本
└──────────────────────────────────────┘
```

总计: N×K + N×Q 个点云块 = 2×1 + 2×1 = 4 个 block

---

## 4. MyDataset 类详解

### 核心属性

```python
# dataloaders/loader.py (简化)
class MyDataset(Dataset):
    def __init__(self, dataset_path, classes, class2scans,
                 n_way, k_shot, n_queries, pc_npts, pc_augm):
        self.dataset_path = dataset_path
        self.classes = classes           # novel classes
        self.class2scans = class2scans   # {class_name: [(file, idx), ...]}
        
        self.n_way = n_way               # 每 episode 的类别数
        self.k_shot = k_shot             # 每类的 support 样本数
        self.n_queries = n_queries       # 每类的 query 样本数
        self.pc_npts = pc_npts           # 每块的点数 (2048)
        
    def __len__(self):
        # 理论上无限，通常设置一个较大的数
        return 100000  # 或 episodes_per_epoch
```

### __getitem__ 完整流程

```python
def __getitem__(self, index):
    # === Step 1: 随机选择 N 个类别 ===
    episode_classes = np.random.choice(
        self.classes,        # 从 novel classes 中选
        self.n_way,          # 选 N 个
        replace=False        # 不重复
    )
    
    # === Step 2: 为每个类别采样 support + query ===
    support_clouds = []  # support 点云
    support_labels = []  # support 标签
    query_clouds = []    # query 点云
    query_labels = []    # query 标签
    
    for cls_idx, cls_name in enumerate(episode_classes):
        # 获取该类所有的 block 列表
        available_blocks = self.class2scans[cls_name]
        
        # 随机选择 K + Q 个不重复的 block
        selected = random.sample(
            available_blocks, 
            self.k_shot + self.n_queries
        )
        
        # 前 K 个 → support, 后 Q 个 → query
        for i, (file_path, label_idx) in enumerate(selected):
            cloud = self._load_and_sample(file_path, label_idx)
            
            if i < self.k_shot:
                # Support: 可以使用数据增强
                if self.pc_augm:
                    cloud = augment_support(cloud)
                support_clouds.append(cloud)
                support_labels.append(cls_idx)
            else:
                # Query: 不使用数据增强 (评估要稳定)
                query_clouds.append(cloud)
                query_labels.append(cls_idx)
    
    # === Step 3: 组装成 batch ===
    # support: (N*K, pc_npts, dim)
    # query:   (N*Q, pc_npts, dim)
    return (
        torch.stack(support_clouds),
        torch.tensor(support_labels),
        torch.stack(query_clouds),
        torch.tensor(query_labels)
    )
```

---

## 5. Support 与 Query 的增强策略差异

| 属性 | Support | Query |
|---|---|---|
| 数据增强 | ✅ 使用 (如果 --pc_augm) | ❌ 不使用 |
| 原因 | 增强泛化能力 | 保持评估一致性 |
| 类比 | 训练集增广 | 验证集保持原样 |

### 代码层面的实现

```python
# dataloaders/loader.py (简化)
def _get_item(self, ...):
    # Support 分支
    if is_support and self.pc_augm:
        points = self.augment(points)  # 旋转/缩放/抖动
    
    # Query 分支
    elif is_query:
        # 不做任何增强，保持原始数据
        pass
```

---

## 6. 单次 __getitem__ 的数据形状

以默认配置为例：

```
n_way=2, k_shot=1, n_queries=1, pc_npts=2048, pc_attribs='xyzrgbXYZ'

返回值:
  support_clouds:  torch.Size([2, 2048, 9])   # N*K=2 个 block
  support_labels:  torch.Size([2])             # [0, 1]
  query_clouds:    torch.Size([2, 2048, 9])   # N*Q=2 个 block
  query_labels:    torch.Size([2])             # [0, 1]
```

---

## 7. Episode 采样与 Collate Function

由于每个 episode 的形状是固定的，collate_fn 比较简单：

```python
# 不需要特殊的 collate，默认的 default_collate 就能处理
# 但如果使用 DataParallel，需要注意 batch size = 1 (一个 episode)
```

对于实际的 batch 训练：

```python
# 虽然 DataLoader 的 batch_size 设为 1，但一个 episode 内部
# 已经包含了 N*K + N*Q 个点云块
dataloader = DataLoader(
    dataset,
    batch_size=1,        # 一次只取 1 个 episode
    shuffle=True,        # episode 顺序随机
    num_workers=4        # 多进程加载
)
```

---

## 8. 实践：追踪 __getitem__

在 `dataloaders/loader.py` 中插入调试代码：

```python
def __getitem__(self, index):
    # 添加调试打印 (仅 first call)
    if index == 0:
        print(f"Episode classes: {episode_classes}")
        print(f"Selected blocks per class: {len(selected)}")
        print(f"Support clouds shape: {len(support_clouds)}")
        print(f"Query clouds shape: {len(query_clouds)}")
    ...
```

---

## 9. 本节总结

| 概念 | 要点 |
|---|---|
| Episode | N 类 × K 个 support × Q 个 query |
| 类选择 | 从 novel classes 中随机不重复选 N 个 |
| 块选择 | 从 class2scans 中随机不重复选 K+Q 个块 |
| 增强差异 | support 增强，query 不增强 |
| 数据形状 | support (N*K, P, D), query (N*Q, P, D) |

---

## 10. 课后练习

1. 如果 `class2scans['table']` 只有 2 个 block，但 `k_shot=5`，会发生什么？如何解决？
2. 修改 Episode 采样，确保 support 和 query 来自 **不同的房间**，测试对性能的影响
3. 为 Episode 添加"难例挖掘"——优先选择分类困难的类别组合
4. 实现一个可视化函数，将一个 Episode 中的所有点云渲染到同一坐标系
