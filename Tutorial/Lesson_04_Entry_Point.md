# 第4课：main.py 入口与命令行参数系统

## 1. 本节学习目标

- 理解 `main.py` 的完整结构
- 掌握 argparse 参数系统的设计模式
- 理解 phase 路由机制的实现
- 能够追踪不同 phase 的代码执行路径

---

## 2. main.py 的整体结构

`main.py` 是项目的唯一入口，遵循"参数解析 → 路由分发"的经典模式：

```python
# main.py 伪代码结构
def main():
    args = get_args()       # Step 1: 解析命令行参数
    set_seed(args.seed)     # Step 2: 设置随机种子
    setup_logger(args)      # Step 3: 配置日志
    
    if args.phase == 'pretrain':     # Step 4: 路由分发
        pretrain_main(args)
    elif args.phase == 'prototrain':
        proto_main(args)
    elif args.phase == 'protoeval':
        eval_main(args)
    elif args.phase == 'mptitrain':
        mpti_main(args)
    elif args.phase == 'mptieval':
        mpti_eval_main(args)
    elif args.phase == 'finetune':
        finetune_main(args)
    else:
        raise ValueError('Unknown phase!')
```

---

## 3. 参数系统详解

### 3.1 参数分组

`get_args()` 函数使用 `argparse` 定义了 40+ 个参数，可分为以下组：

```python
def get_args():
    parser = argparse.ArgumentParser()
    
    # === 第1组: 基础配置 ===
    parser.add_argument('--phase', type=str, required=True,
                        choices=['pretrain', 'prototrain', 'protoeval', 
                                 'mptitrain', 'mptieval', 'finetune'])
    parser.add_argument('--dataset', type=str, default='S3DIS',
                        choices=['S3DIS', 'ScanNet'])
    parser.add_argument('--cvfold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    
    # === 第2组: 数据配置 ===
    parser.add_argument('--n_way', type=int, default=2)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--n_queries', type=int, default=1)
    parser.add_argument('--pc_npts', type=int, default=2048)
    parser.add_argument('--pc_attribs', type=str, default='xyzrgbXYZ')
    parser.add_argument('--pc_augm', action='store_true')
    
    # === 第3组: 模型结构 ===
    parser.add_argument('--dgcnn_k', type=int, default=20)
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--train_dim', type=int, default=320)
    parser.add_argument('--noise_dim', type=int, default=300)
    
    # === 第4组: 模型开关 (Feature Flags) ===
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--use_transformer', action='store_true')
    parser.add_argument('--use_align', action='store_true')
    parser.add_argument('--use_supervise_prototype', action='store_true')
    parser.add_argument('--use_high_dgcnn', action='store_true')
    parser.add_argument('--use_linear_proj', action='store_true')
    parser.add_argument('--use_zero', action='store_true')
    
    # === 第5组: 训练超参数 ===
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    
    # === 第6组: 路径配置 ===
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--pretrain_checkpoint', type=str)
    
    return parser.parse_args()
```

### 3.2 Feature Flags 的默认值

所有 `action='store_true'` 的参数默认为 `False`。这意味着：

```
默认模型: ProtoNet (最简版本, 无注意力/无QGPA/无对齐/无SR)

python main.py --phase prototrain
→ 使用最基础的 ProtoNet

增强模型:
python main.py --phase prototrain \
    --use_high_dgcnn \          # 使用 DGCNN_semseg 骨干
    --use_attention \           # 启用自注意力
    --use_transformer \         # 启用 QGPA
    --use_align \               # 启用对齐损失
    --use_supervise_prototype \ # 启用自重建
    --use_linear_proj           # 启用线性投影
→ 使用完整的 PAP (ProtoNetAlignQGPASR)
```

---

## 4. Phase 路由详解

### Phase 机制

`--phase` 参数决定了程序执行哪个任务：

```python
# main.py 中的路由逻辑 (简化)
if __name__ == '__main__':
    args = get_args()
    
    # 设置 CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.phase == 'pretrain':
        from runs.pre_train import main as pretrain_main
        pretrain_main(args)
        
    elif args.phase == 'prototrain':
        from runs.proto_train import proto_main
        proto_main(args)
        
    elif args.phase == 'protoeval':
        from runs.eval import main as eval_main
        eval_main(args)
        
    elif args.phase == 'mptitrain':
        from runs.mpti_train import main as mpti_train_main
        mpti_train_main(args)
        
    elif args.phase == 'mptieval':
        from runs.mpti_train import eval_main as mpti_eval_main
        mpti_eval_main(args)
        
    elif args.phase == 'finetune':
        from runs.fine_tune import main as finetune_main
        finetune_main(args)
```

### 各 Phase 的模型选择逻辑

不同 phase 会使用不同的模型类：

| Phase | 使用的模型 | 文件 |
|---|---|---|
| `pretrain` | `DGCNNSeg` | `runs/pre_train.py` |
| `prototrain` | `ProtoNetAlignQGPASR` 或 `ProtoNetAlignFZ` | `runs/proto_train.py` |
| `protoeval` | 同上 | `runs/eval.py` |
| `mptitrain` | `MPTI` 模型 | `runs/mpti_train.py` |
| `mptieval` | `MPTI` 模型 | `runs/mpti_train.py` |
| `finetune` | `DGCNNSeg` (微调) | `runs/fine_tune.py` |

**FZ 模式的选择**：

```python
# runs/proto_train.py 中的模型选择逻辑 (简化)
if args.use_zero:
    model = ProtoNetAlignFZ(args)      # 含 GMMN 生成器
    learner = ProtoLearnerFZ(args)
else:
    model = ProtoNetAlignQGPASR(args)   # 纯少样本
    learner = ProtoLearner(args)
```

---

## 5. 实践：追踪执行路径

### 任务1：列出 prototrain phase 的所有函数调用

从 `main.py` 开始，运行：

```bash
python main.py --phase prototrain --dataset S3DIS --cvfold 0 --n_way 2 --k_shot 1
```

执行路径：

```
main.py
  → runs/proto_train.py:proto_main(args)
      → models/protonet_QGPA.py:ProtoNetAlignQGPASR.__init__()
          → models/dgcnn_new.py:DGCNN_semseg.__init__()
          → models/attention.py:SelfAttention.__init__()
          → models/attention.py:QGPA.__init__()
      → models/proto_learner.py:ProtoLearner.__init__()
      → dataloaders/loader.py:MyDataset.__init__()
      → dataloaders/loader.py:MyTestDataset.__init__()
      → proto_learner.train()
          → proto_learner.train_epoch()
          → proto_learner.test_few_shot()  # 定期验证
```

### 任务2：理解 --use_high_dgcnn 的影响

```python
# models/protonet_QGPA.py (简化)
if args.use_high_dgcnn:
    self.encoder = DGCNN_semseg(args)  # 64→64→64 → 192 concat
else:
    self.encoder = DGCNN(args)         # 不同通道配置
```

跟踪这个 flag 如何影响模型结构：
1. 打开 `models/dgcnn_new.py`，对比 `DGCNN_cls` 和 `DGCNN_semseg`
2. 打开 `models/protonet_QGPA.py`，搜索 `use_high_dgcnn`

---

## 6. Shell 脚本的参数解析

Shell 脚本实现了与 Python 相同的参数逻辑：

```bash
# scripts/train_PAP.sh (简化)
python main.py \
    --phase prototrain \
    --dataset S3DIS \
    --cvfold 0 \
    --n_way 2 \
    --k_shot 1 \
    --n_queries 1 \
    --pc_npts 2048 \
    --pc_augm \
    --use_high_dgcnn \
    --use_attention \
    --use_transformer \
    --use_align \
    --use_supervise_prototype \
    --use_linear_proj
```

---

## 7. 常见错误

### Q1: 忘记设置 --phase 参数

```bash
$ python main.py
# 错误: the following arguments are required: --phase
```

### Q2: Phase 和参数不匹配

```bash
$ python main.py --phase pretrain --use_transformer
# pretrain 不使用 transformer, 该参数被忽略
```

### Q3: 脚本中的路径错误

```bash
# 确保 scripts/ 中的路径指向正确的数据集目录
--data_dir ./datasets/S3DIS
```

---

## 8. 本节总结

| 概念 | 要点 |
|---|---|
| 参数系统 | argparse, 40+ 参数分 6 组 |
| Feature Flags | `action='store_true'`, 默认关闭所有增强 |
| Phase 路由 | `if/elif` 链, 7 种 phase |
| 模型选择 | 通过 flags 动态选择模型类和配置 |
| Shell 脚本 | 封装参数组合, 方便复现实验 |

---

## 9. 课后练习

1. 运行 `python main.py --help`，列出所有参数及其默认值
2. 写一个最小的 prototrain 命令（只用必需参数），并预测会使用哪个模型类
3. 在 `main.py` 中添加一个 `print(args)` 语句，观察所有解析后的参数值
4. 修改 `scripts/train_PAP.sh`，关闭 `--use_attention`，观察对模型选择的影响
