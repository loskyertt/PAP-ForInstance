# 第27课：Shell 实验脚本解析

## 1. 本节学习目标

- 理解 5 个 Shell 脚本的功能和区别
- 掌握脚本中的参数配置逻辑
- 能够修改脚本以适配不同的实验设置
- 理解完整的实验运行顺序

---

## 2. 脚本全景

| 脚本 | Phase | 作用 | 运行顺序 |
|---|---|---|---|
| `pretrain_segmentor.sh` | pretrain | 预训练 backbone | ① 第一步 |
| `train_PAP.sh` | prototrain | 训练 PAP (少样本) | ② 第二步 |
| `train_PAPFZ.sh` | prototrain | 训练 PAP-FZ (少+零) | ②' 第二步(替代) |
| `eval_PAP.sh` | protoeval | 评估 PAP | ③ 第三步 |
| `eval_PAPFZ.sh` | protoeval | 评估 PAP-FZ | ③' 第三步(替代) |

---

## 3. pretrain_segmentor.sh 详解

```bash
#!/bin/bash
# 预训练 DGCNN 语义分割器

python main.py \
    --phase pretrain \                      # ★ 预训练阶段
    --dataset S3DIS \                       # 数据集
    --cvfold 0 \                            # fold 0 (base/novel 划分)
    --use_high_dgcnn \                      # 使用 DGCNN_semseg
    --pc_augm \                             # 数据增强
    --pc_npts 2048 \                        # 采样点数
    --n_epochs 100 \                        # 训练轮数
    --batch_size 8 \                        # batch size
    --lr 0.001 \                            # 学习率
    --log_dir runs/pretrain_s3dis_fold0     # 日志目录
```

### 关键参数

```bash
--batch_size 8
# 每个 batch 包含 8 个 block → 8 × 2048 = 16384 点
# 需要约 4GB GPU 内存

--n_epochs 100
# 100 epoch 通常在 S3DIS 上达到 85%+ 的 base class 准确率

--pc_augm
# 开启数据增强，增加训练样本多样性
```

---

## 4. train_PAP.sh 详解

```bash
#!/bin/bash
# 训练 PAP (少样本)

python main.py \
    --phase prototrain \                    # ★ 元训练阶段
    --dataset S3DIS \
    --cvfold 0 \
    --n_way 2 \                             # 2-way 任务
    --k_shot 1 \                            # 1-shot
    --n_queries 1 \                         # 每个类 1 个 query block
    --pc_npts 2048 \
    --pc_augm \
    --use_high_dgcnn \                      # 模型: DGCNN_semseg
    --use_attention \                       # 模型: Self-Attention
    --use_transformer \                     # 模型: QGPA
    --use_align \                           # 损失: Alignment Loss
    --use_supervise_prototype \             # 损失: SR Loss
    --use_linear_proj \                      # 模型: Linear Projection
    --output_dim 64 \                       # 嵌入维度
    --train_dim 320 \                       # 拼接后维度
    --episodes_per_epoch 100 \              # 每 epoch 100 个 episode
    --lr 0.001 \
    --epochs 50 \
    --log_dir runs/PAP_s3dis_fold0_2way1shot
```

### 与 pretrain 的配合

```bash
# pretrain 预训练 → 保存 checkpoint
# train_PAP 加载 checkpoint
--pretrain_checkpoint runs/pretrain_s3dis_fold0/best_model.pth
```

---

## 5. train_PAPFZ.sh 详解

```bash
#!/bin/bash
# 训练 PAP-FZ (少样本 + 零样本)

# Step 0: 生成词嵌入 (如果还没生成)
python dataloaders/get_embedding.py --dataset S3DIS

# Step 1: 训练
python main.py \
    --phase prototrain \
    --dataset S3DIS \
    --cvfold 0 \
    --n_way 4 \                             # ★ 更多类 (seen + unseen)
    --k_shot 1 \
    --n_queries 1 \
    --use_high_dgcnn \
    --use_attention \
    --use_transformer \
    --use_align \
    --use_supervise_prototype \
    --use_linear_proj \
    --use_zero \                            # ★ 开启 FZ 模式
    --gmmn_weight 0.1 \                     # GMMN 损失权重
    --noise_dim 300 \                       # 噪声维度
    --embedding_type word2vec \             # 词嵌入类型
    --episodes_per_epoch 200 \
    --lr 0.001 \
    --epochs 50
```

### FZ 特有参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--use_zero` | flag | 开启零样本生成器 |
| `--gmmn_weight` | 0.1 | GMMN 损失权重 |
| `--noise_dim` | 300 | 生成器噪声维度 |
| `--embedding_type` | word2vec | 语义嵌入类型 |
| `--n_way` | 4~6 | PC 需要更多类 (seen + unseen) |

---

## 6. eval_PAP.sh 和 eval_PAPFZ.sh

```bash
#!/bin/bash
# 评估 PAP

python main.py \
    --phase protoeval \                     # ★ 评估阶段
    --dataset S3DIS \
    --cvfold 0 \
    --n_way 2 \
    --k_shot 1 \
    --n_queries 1 \
    --use_high_dgcnn \
    --use_attention \
    --use_transformer \
    --use_linear_proj \
    --model_checkpoint \                    # ★ 加载训练好的模型
        runs/PAP_s3dis_fold0_2way1shot/best_model.pth \
    --n_episodes_test 1000 \                # 测试 episode 数
    --log_dir runs/eval_PAP_s3dis_fold0

# PAP-FZ 版本额外需要:
#   --use_zero
#   --embedding_type word2vec
```

### 评估参数

```bash
--n_episodes_test 1000
# 测试 1000 个随机 episode
# 越多越稳定，但越慢
# 论文通常用 1000~2000

--model_checkpoint <path>
# 必须指定训练好的模型路径
# 否则使用随机权重 → 结果无意义
```

---

## 7. 完整的实验流水线

```bash
#!/bin/bash
# 完整实验: S3DIS, fold_0, 2-way 1-shot

# Step 1: 预训练 (约 4 小时, GPU)
bash scripts/pretrain_segmentor.sh

# Step 2: 元训练 (约 2 小时, GPU)
bash scripts/train_PAP.sh

# Step 3: 评估 (约 30 分钟, GPU)
bash scripts/eval_PAP.sh

# 结果保存在 runs/eval_PAP_s3dis_fold0/
```

### 多配置并行

```bash
# 同时运行多个实验 (需要多个 GPU)
CUDA_VISIBLE_DEVICES=0 bash scripts/train_PAP.sh \
    --n_way 2 --k_shot 1 &
CUDA_VISIBLE_DEVICES=1 bash scripts/train_PAP.sh \
    --n_way 2 --k_shot 5 &
CUDA_VISIBLE_DEVICES=2 bash scripts/train_PAP.sh \
    --n_way 3 --k_shot 1 &
CUDA_VISIBLE_DEVICES=3 bash scripts/train_PAP.sh \
    --n_way 3 --k_shot 5 &
wait  # 等待全部完成
```

---

## 8. 自定义脚本模板

```bash
#!/bin/bash
# 自定义消融实验: 测试 Self-Attention 的影响

DATASET="S3DIS"
FOLD=0
NWAY=2
KSHOT=1

BASE_ARGS="
    --phase prototrain
    --dataset $DATASET
    --cvfold $FOLD
    --n_way $NWAY
    --k_shot $KSHOT
    --use_high_dgcnn
    --use_transformer
    --use_align
    --use_supervise_prototype
    --use_linear_proj
"

# 实验 1: 无 Attention
python main.py $BASE_ARGS \
    --log_dir runs/ablation_no_attn

# 实验 2: 有 Attention
python main.py $BASE_ARGS --use_attention \
    --log_dir runs/ablation_with_attn
```

---

## 9. 本节总结

| 脚本 | 阶段 | 输出 | GPU 耗时 (估计) |
|---|---|---|---|
| pretrain_segmentor.sh | 预训练 | encoder.pth | ~4h |
| train_PAP.sh | 元训练 | PAP_model.pth | ~2h |
| train_PAPFZ.sh | 元训练+FZ | PAPFZ_model.pth | ~3h |
| eval_PAP.sh | 评估 | IoU metrics | ~30min |
| eval_PAPFZ.sh | 评估+FZ | IoU metrics | ~40min |

---

## 10. 课后练习

1. 修改 `train_PAP.sh`，添加 `--k_shot 5` 并运行
2. 写一个新脚本 `train_PAP_ScanNet.sh`，使用 ScanNet 数据集
3. 创建一个批量运行脚本，循环执行 2-way, 3-way, 5-way 的实验
4. 在 eval 脚本中添加自动画 IoU 柱状图的功能
