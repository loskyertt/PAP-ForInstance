# 第8课：HDF5 数据缓存与 Test Dataset

## 1. 本节学习目标

- 理解为什么测试数据需要预生成和缓存
- 掌握 HDF5 文件格式的基础操作
- 理解 MyTestDataset 的完整设计
- 能够分析测试集的去重与索引机制

---

## 2. 问题：测试时的重复 I/O

在测试阶段，需要评估模型在大量 episode 上的表现。如果每个 episode 都实时采样：

```python
# 低效方式
for episode in range(1000):
    # 每次都要: 读磁盘 → 选类 → 选块 → 采样 → normalize
    support, query = real_time_sample()
```

**问题**：每个 block 可能被多个 episode 重复使用，导致重复 I/O 和预处理。

---

## 3. 解决方案：预生成 + HDF5 缓存

### 设计思路

```
预处理阶段 (一次性):
  1. 枚举所有可能的 (class_combination, episode) 组合
  2. 对每个 episode，采样并预处理点云块
  3. 存入 HDF5 文件 (.h5)
  4. 记录索引元数据 (pickle)

测试阶段:
  1. 打开 HDF5 文件 (只读)
  2. 根据索引直接读取预处理的 Tensor
  3. 零 I/O 开销
```

### 优势

| 属性 | 实时采样 | HDF5 缓存 |
|---|---|---|
| 磁盘 I/O | 每次读取 | 一次性读取 |
| 预处理 | 每次重复 | 预先完成 |
| 可复现性 | 依赖 seed | 完全确定 |
| 内存占用 | 低 | 中等 (预加载) |

---

## 4. HDF5 文件格式简介

HDF5 是一种高效的分层数据格式，适合存储大型多维数组：

```python
import h5py

# 创建 HDF5 文件
with h5py.File('test_episodes.h5', 'w') as f:
    # 创建数据集
    f.create_dataset('support_clouds', 
                     data=...,
                     shape=(num_episodes, N*K, pc_npts, dim))
    f.create_dataset('support_labels', 
                     data=..., 
                     shape=(num_episodes, N*K))
    f.create_dataset('query_clouds', 
                     data=...,
                     shape=(num_episodes, N*Q, pc_npts, dim))
    f.create_dataset('query_labels', 
                     data=..., 
                     shape=(num_episodes, N*Q))

# 读取时
with h5py.File('test_episodes.h5', 'r') as f:
    episode_0_support = f['support_clouds'][0]  # 高效读取
```

---

## 5. MyTestDataset 类详解

### Episode 组合枚举

```python
# dataloaders/loader.py (简化)
class MyTestDataset(Dataset):
    def __init__(self, dataset_path, classes, class2scans, 
                 n_way, k_shot, n_queries, pc_npts, 
                 n_episodes=100, cache_dir='test_cache'):
        
        self.cache_path = os.path.join(
            cache_dir, 
            f'test_{len(classes)}way_{k_shot}shot.h5'
        )
        
        if not os.path.exists(self.cache_path):
            self._generate_and_cache(
                classes, class2scans, n_way, k_shot, 
                n_queries, pc_npts, n_episodes
            )
        
        # 打开 HDF5 (只读模式)
        self.h5file = h5py.File(self.cache_path, 'r')
    
    def _generate_and_cache(self, classes, class2scans, 
                            n_way, k_shot, n_queries, 
                            pc_npts, n_episodes):
        """
        预生成所有测试 episode 并存入 HDF5
        """
        # Step 1: 生成所有 N-way 组合
        from itertools import combinations
        class_combos = list(combinations(range(len(classes)), n_way))
        
        # Step 2: 为每个组合生成 episodes
        all_support = []
        all_query = []
        
        episodes_per_combo = n_episodes // len(class_combos)
        
        for combo in class_combos:
            for _ in range(episodes_per_combo):
                support, query = self._sample_episode(
                    combo, class2scans, k_shot, n_queries, pc_npts
                )
                all_support.append(support)
                all_query.append(query)
        
        # Step 3: 写入 HDF5
        with h5py.File(self.cache_path, 'w') as f:
            f.create_dataset('support', data=np.array(all_support))
            f.create_dataset('query', data=np.array(all_query))
    
    def __getitem__(self, index):
        # 直接从缓存读取，极快
        support = torch.from_numpy(self.h5file['support'][index])
        query = torch.from_numpy(self.h5file['query'][index])
        return support, query
```

---

## 6. 去重策略

### Block 级别的去重

```python
def _sample_episode(self, combo, class2scans, k_shot, n_queries, pc_npts):
    """
    确保同一个 episode 中不会重复使用同一个 block
    """
    used_blocks = set()
    
    for cls_idx in combo:
        cls_name = self.classes[cls_idx]
        available = [b for b in class2scans[cls_name] 
                     if b not in used_blocks]
        
        selected = random.sample(available, k_shot + n_queries)
        used_blocks.update(selected)
    ...
```

### Episode 级别的去重

```python
def _generate_episodes(self, class_combos, n_episodes):
    """
    确保不会生成完全相同的 episode
    """
    seen_episodes = set()
    episodes = []
    
    while len(episodes) < n_episodes:
        combo = random.choice(class_combos)
        blocks = tuple(self._select_blocks(combo))
        
        if blocks not in seen_episodes:
            seen_episodes.add(blocks)
            episodes.append(blocks)
    
    return episodes
```

---

## 7. 实践：分析缓存文件

```python
import h5py
import numpy as np

# 查看测试缓存文件的结构
cache_path = 'test_cache/test_6way_1shot.h5'

with h5py.File(cache_path, 'r') as f:
    print("=== HDF5 文件结构 ===")
    f.visit(print)  # 打印所有数据集
    
    print("\n=== 数据集详情 ===")
    for key in f.keys():
        print(f"{key}: shape={f[key].shape}, dtype={f[key].dtype}")
    
    # 读取第一个 episode
    support = f['support'][0]
    query = f['query'][0]
    print(f"\nEpisode 0: support={support.shape}, query={query.shape}")
```

---

## 8. Pickle 索引文件

除了 HDF5，还会生成 `class2scans.pkl` 来索引文件位置：

```python
import pickle

# 保存
with open('class2scans.pkl', 'wb') as f:
    pickle.dump(class2scans, f)

# 加载
with open('class2scans.pkl', 'rb') as f:
    class2scans = pickle.load(f)
```

作用：避免每次启动都重新遍历所有 `.npy` 文件。

---

## 9. 本节总结

| 概念 | 要点 |
|---|---|
| 预生成 | 一次性枚举所有测试 episode 组合 |
| HDF5 | 高效存储多维数组，支持随机索引读取 |
| 去重 | block 级别 + episode 级别双重去重 |
| Pickle | 索引 file→class 映射，避免重复扫描 |
| 性能 | 测试时零 I/O 开销，适合大规模评估 |

---

## 10. 课后练习

1. 用 Python 脚本创建一个简单的 HDF5 文件，存储随机数据，然后读取
2. 计算：N=6, K=1 时，如果随机采样 1000 个 episode，不重复 episode 的概率是多少？
3. 修改 MyTestDataset，添加一个"按类别分组"的读取模式，支持 per-class 分析
4. 比较 HDF5 缓存 vs 实时采样的测试时间差异
