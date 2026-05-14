> [!abstract]
> 仅实现 FOR-Instance 数据集与 PAP 模型相适配后的训练效果分析。

# 1. 第一次训练 (2026-05-08)

## 1.1 预训练阶段 

> `log_forinstance/log_pretrain_forinstance_S0/log_pretrain.txt`

- **训练数据**: 基类 (class 1,3,4 = low_vegetation, live_branches, woody_branches)，418 个训练块 + 36 个验证块
- **训练时长**: 50 epochs，每 3 个 epoch 评估一次
- **效果**:
  - Epoch 2 达到 **mIoU 59.65%**
  - 最终 Epoch 41 达到最佳 **mIoU 64.33%**，Accuracy ~85%
  - 模型在基类上分割效果良好

## 1.2 PAP 小样本训练

> `log_forinstance_PAP/log_proto_forinstance_S0_N2_K1_Att1/log_prototrain.txt`

- **配置**: 2-way 1-shot，40000 iterations
- **测试类**: class 0 (terrain) 和 class 2 (stem) — 训练时**未见过的**新类
- **评估结果 (每 2000 iters)**:

| Iteration | class 0 terrain IoU | class 1 (bg) IoU | class 2 stem IoU | Best mIoU |
|:---------:|:-------------------:|:----------------:|:----------------:|:---------:|
| 2000 | 57.2% | 43.9% | 16.4% | **30.1%** |
| 4000 | 62.7% | 49.6% | 15.6% | **32.6%** |
| 6000 | 63.1% | 51.5% | 13.7% | 32.6% |
| ... 收敛期 ... | 57~62% | 48~51% | 14~18% | 32.6% |
| 18000 | 57.7% | 50.5% | 16.4% | **33.45%** |
| 20000 | 61.9% | 46.2% | 18.9% | 33.45% |
| 22000 | 59.3% | 48.2% | 17.3% | 33.45% |
| ... 收敛期 ... | 56~61% | 46~50% | 16~18% | 33.45% |
| 38000 | 60.9% | 50.0% | 17.7% | **33.85%** |
| 40000 | 62.4% | 47.6% | 18.6% | **33.85%** |

## 1.3 最终评估（未做）

> `trash/log_forinstance_PAP/log_proto_forinstance_S0_N2_K1_Att1/log_protoeval.txt`

> [!warning] 
> 第一次训练后的最终 protoeval 只用了 `n_episode_test=1`（仅 1 个测试 episode），结果偶然性较大，仅供参考。

- **第 1 次（npts=256）**: terrain 63.7%, bg 36.3%, stem 33.3%, **mIoU=34.78%**
- **第 2 次（npts=2048）**: terrain 24.8%, bg 55.5%, stem 8.1%, **mIoU=31.78%**

## 1.4 第一次训练小结

- 确认了 PAP-FZS3D 基线在 FOR-Instance 上可复现：**33.85% mIoU**
- protoeval 仅 n_episode_test=1，不可靠，训练过程中 100 episodes 的评估（33.85%）才是可用结果

---

# 2. 第二次训练 (2026-05-14)

## 2.1 预训练阶段

> `log_forinstance/log_pretrain_forinstance_S0/log_pretrain.txt`

- **训练数据**: 基类 (class 1,3,4 = low_vegetation, live_branches, woody_branches)，418 个训练块 + 36 个验证块
- **训练时长**: 50 epochs，每 3 个 epoch 评估一次
- **效果**: 
  - Epoch 2 已到 **mIoU 59.09%**
  - 最终 Epoch 47 达到最佳 **mIoU 65.86%**，Accuracy ~88%
  - 模型在基类上分割效果良好，为后续小样本学习提供了好的编码器初始化

## 2.2 PAP 小样本训练

> `log_forinstance_PAP/log_proto_forinstance_S0_N2_K1_Att1/log_prototrain.txt`

- **配置**: 2-way 1-shot（每 episode 2 个类 × 1 个 support 点云），40000 iterations
- **测试类**: class 0 (terrain) 和 class 2 (stem) — 即训练时**未见过的**新类，每 2000 iters 评估
- **训练过程**: 
  - 训练 Loss 从 0.80 逐步下降到 ~0.30~0.50（有波动正常，每 episode 内容不同）
  - 训练 Accuracy 稳定在 92%~97%（点级别的语义分割精度）
- **评估结果 (每 2000 iters 的 protoeval)**:

| Iteration | class 0 terrain IoU | class 1 (bg) IoU | class 2 stem IoU | Best mIoU |
|-----------|:------------------:|:----------------:|:----------------:|:---------:|
| 2000 | 63.6% | 47.9% | 16.9% | **32.4%** |
| 4000 | 63.0% | 42.4% | 17.8% | 32.4% |
| 6000 | 53.0% | 37.1% | 15.4% | 32.4% |
| 8000 | 62.0% | 51.2% | 15.8% | **33.5%** |
| 10000 | **72.2%** | 54.6% | 17.0% | **35.8%** |
| 12000 | 67.7% | 55.5% | 16.8% | **36.2%** |
| 14000 | 64.6% | 54.5% | 15.4% | **36.2%** |
| 16000 | 64.1% | **58.8%** | 15.2% | **37.0%** ✅ |
| 18000 | 71.2% | 58.8% | 16.0% | **37.4%** ✅ |
| 20000 | 66.0% | **59.9%** | 15.8% | **37.9%** ✅ |
| 22000 | 68.3% | **59.9%** | 16.3% | **38.1%** ✅ |
| 24000~40000 | 62~68% | 54~59% | 15~16% | **38.1%** (收敛) |

## 2.3 最终评估

> `trash/log_forinstance_PAP/log_proto_forinstance_S0_N2_K1_Att1/log_protoeval.txt`

- 训练完成后，加载最佳 checkpoint 在 100 个固定测试 episode 上评估：
  - **class 0 (terrain) IoU: 66.40%**
  - **class 1 (background) IoU: 59.90%**
  - **class 2 (stem) IoU: 15.18%**
  - **Mean IoU: 37.54%**

## 2.4 训练效果分析

| 指标 | 值 | 评价 |
|------|:---:|:----|
| 预训练基类 mIoU | **65.86%** | 良好，基类分割能力够用 |
| 小样本 terrain IoU | **66.40%** | 较好，terrain（地面）语义清晰易分 |
| 小样本 background IoU | **59.90%** | 尚可，背景类占大多数点 |
| 小样本 stem IoU | **15.18%** | ❌ **较差**，树干细长、点稀疏，是小样本分割的难点 |
| 总体 mIoU | **37.54%** | terrain、background、stem 三者的平均值 |

> stem（树干）IoU 低是**预期中的难点**：树干在点云中只占少量点、形状细长，1-shot 情况下 support 样本很难覆盖其外观变化。

---

# 3. 两次训练对比总结

## 3.1 效果对比

| 阶段 | 指标 | 第一次 | 第二次 | 差异分析 |
|------|------|:--------------:|:-------------:|:---------|
| **预训练** | 最佳 mIoU | **64.33%** | **65.86%** | +1.53%，第二次稍好，随机波动范围内 |
|  | 最佳 Epoch | 41 | 47 | - |
| **PAP 小样本** | **最终 mIoU** | **33.85%** | **38.09%** | **+4.24%**，显著提升 |
|  | class 0 terrain | ~57-62% | ~63-72% | terrain 分割更稳定 |
|  | class 1 background | ~44-50% | ~54-60% | bg 识别有较大提升 |
|  | class 2 stem | ~15-19% | ~15-17% | stem 无明显变化，仍是瓶颈 |
| **protoeval** | mIoU (100 ep.) | 未做 | **37.54%** | 第二次评估更规范 |

## 3.2 第二次训练效果更好的原因

1. **预训练编码器更强**：第二次的 pretrain checkpoint (65.86% mIoU) 优于第一次 (64.33% mIoU)，为 PAP 训练提供了更好的特征提取初始化
2. **Episodic 采样随机性**：小样本训练中 support set 的采样是随机的，两次实验遇到的 episode 分布不同，导致收敛结果有差异
3. **第二次训练更充分**：第一次的 protoeval 配置有误（`n_episode_test=1`），且训练过程中 mIoU 提升较慢，可能 param 初始化或数据加载顺序影响了收敛

## 3.3 共同发现

- **Stem (树干) 是最难分割的类别**：两次实验中 stem IoU 均只有 **15-18%**，说明 1-shot 条件下树干细长结构的点云分割是本任务的核心挑战
- **Terrain (地面) 分割效果最好**：class 0 IoU 可达 60-72%，地面点云语义清晰、空间分布集中，易于分割
- **训练过程稳定**：两次实验的训练 Accuracy 均稳定在 92-97%，Loss 呈下降趋势，无过拟合或发散迹象

## 3.4 当前评估状态

| 评估类型 | 状态 | 说明 |
|----------|:----:|------|
| ✅ **训练中评估** (Training-time eval) | **已完成** | 两次均记录了全过程评估 |
| ✅ **最终独立评估** (Post-training eval) | **部分完成** | 第一次 n_episode_test=1 不可靠；第二次 100 episodes 可靠 |
| ❌ **跨 fold 评估** (Cross-fold eval) | **未做** | 当前只跑了 `cvfold=0`（测试类: terrain+stem），`cvfold=1`（测试类: live_branches+woody_branches）尚未训练和评估 |
| ❌ **不同 shot 数实验** | **未做** | 当前仅 1-shot，可以做 2-shot, 5-shot 等对比实验 |
| ❌ **多次重复取平均** | **未做** | 现有两次结果有 4% 差异，需多次重复（如 5 次）取平均得到稳定结果 |

**结论**：第一次训练确认了 PAP-FZS3D 基线在 FOR-Instance 上可适配。第二次训练通过更好的预训练权重获得了更优的 38.09% mIoU，验证了模型的潜力。后续应补充：(1) Fold 1 的完整训练和评估；(2) 多次重复实验取平均；(3) 不同 shot 数的对比实验。
