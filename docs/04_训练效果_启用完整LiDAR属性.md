> [!abstract]
> 启用完整 LiDAR 属性（`xyzIRNAXYZ`，10 通道）的实验分析。
> 对比原 `xyzIXYZ`（7 通道）配置，新增 return_number（R）、number_of_returns（N）、scan_angle / scan_angle_rank（A）三个属性。
> 其余主要配置保持一致：高度分层原型 `lambda=0.05`，`B=3`，未启用类别平衡损失。

# 1. 实验说明

将 `--pc_attribs` 从 `xyzIXYZ` 改为 `xyzIRNAXYZ`，点云编码器输入从 7 通道扩展为 **10 通道**：

| 通道 | 含义 | 来源列 |
|:----:|------|:------:|
| xyz | 原始坐标 | col 0-2 |
| I | intensity 强度 | col 3 |
| **R** | **return_number 回波次序** | **col 4** |
| **N** | **number_of_returns 总回波数** | **col 5** |
| **A** | **scan_angle / scan_angle_rank 扫描角度** | **col 6** |
| XYZ | 归一化坐标 | 由 xyz 计算 |

> [!warning] 注意
> `protoeval` / `prototrain` 日志中的 `[class 0] [class 1] [class 2]` 是 few-shot episode 内部标签，不是 FOR-Instance 原始类别编号。当前 `cvfold=0` 的 `Test Classes: [0, 2]`，因此：
>
> - `[class 0]` = episodic background，不参与 Mean IoU 计算；
> - `[class 1]` = dataset class 0，即 terrain；
> - `[class 2]` = dataset class 2，即 stem；
> - 日志里的 `Mean IoU` 只对 `[class 1]` 和 `[class 2]` 求平均。

# 2. 训练配置

| 参数 | 值 |
|------|:---:|
| `--pc_attribs` | **xyzIRNAXYZ**（10 通道） |
| `pc_in_dim` | **10** |
| `--use_height_proto` | True |
| `--height_proto_bins` | 3 |
| `--height_proto_weight` | 0.05 |
| `--use_balanced_loss` | False |
| 预训练 checkpoint | `./logs/log_forinstance_full_lidar/log_pretrain_forinstance_S0`（重新训练，10 通道） |
| PAP checkpoint | `./logs/log_forinstance_hproto_full_lidar/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.05` |

# 3. 实验结果

## 3.1 预训练阶段

日志文件：

`logs/log_forinstance_full_lidar/log_pretrain_forinstance_S0/log_pretrain.txt`

| 指标 | xyzIXYZ (7ch) | xyzIRNAXYZ (10ch) | 差异 |
|:----|:-------------:|:-----------------:|:----:|
| 最佳 mIoU | **65.86%** | **63.46%** | -2.40% |
| 达到 Epoch | 47 | 47 | 一致 |

10 通道预训练的最佳结果来自 Epoch 47：Accuracy 81.79%，mIoU 63.4564%。与 7 通道历史结果相比略低，说明新增 R/N/A 后，当前预训练设置下没有带来编码器性能提升。

## 3.2 PAP 小样本训练

日志文件：

`logs/log_forinstance_hproto_full_lidar/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.05/log_prototrain.txt`

| Iteration | episodic bg IoU | terrain IoU | stem IoU | Mean IoU | Best mIoU |
|:---------:|:---------------:|:-----------:|:--------:|:--------:|:---------:|
| 2000 | 46.34% | 42.84% | 15.57% | 29.20% | **29.20%** |
| 4000 | 43.15% | 41.94% | 15.96% | 29.95% | 29.20% |
| 6000 | 41.71% | **46.21%** | 15.00% | **30.60%** | **30.60%** |
| 8000 | 40.39% | 40.68% | 15.38% | 28.03% | 30.60% |
| 10000 | 43.42% | 43.34% | 15.67% | 29.50% | 30.60% |
| 34000 | **47.93%** | 40.18% | 15.05% | 27.61% | 30.60% |
| 40000 | 43.17% | 43.63% | 14.84% | 29.23% | 30.60% |

训练期间最佳 mIoU 为 **30.6041%**，出现在 6000 iteration。之后验证 mIoU 没有超过该值，整体在约 27.0% 到 30.6% 之间波动。

## 3.3 最终评估

日志文件：

`logs/log_forinstance_hproto_full_lidar/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.05/log_protoeval.txt`

- **episodic background IoU: 47.92%**
- **terrain IoU: 42.76%**
- **stem IoU: 15.66%**
- **Mean IoU: 29.21%**

这里的 Mean IoU = `(terrain IoU + stem IoU) / 2`，不包含 episodic background。

# 4. 对比分析

## 4.1 与 `lambda=0.05` 的 7 通道组对比

> 当前 `logs/` 目录下可直接核验的是完整 LiDAR 组；7 通道 `lambda=0.05` 数值来自已有消融实验记录，下表用于横向分析。

| 指标 | `lambda=0.05` (xyzIXYZ) | `lambda=0.05` (xyzIRNAXYZ) | 差异 |
|:----|:----------------------:|:--------------------------:|:----:|
| 训练最佳 mIoU | **32.81%** | 30.60% | -2.21% |
| 最终 Eval mIoU | **29.59%** | 29.21% | -0.38% |
| Eval episodic bg IoU | 61.29% | 47.92% | -13.37% |
| Eval terrain IoU | **44.18%** | 42.76% | -1.42% |
| Eval stem IoU | 15.00% | **15.66%** | +0.66% |

下降最大是 **episodic background**，terrain 仅小幅下降，stem 小幅上升，但这两项的变化都不足以带来总体 mIoU 改善。

## 4.2 与已有实验综合对比

| 实验 | 预训练 mIoU | PAP 训练最佳 mIoU | Eval mIoU | stem IoU |
|:----|:-----------:|:-----------------:|:---------:|:--------:|
| adapted 基线 (7ch) | **65.86%** | **38.09%** | **37.54%** | 15.18% |
| `lambda=0.05` (7ch) | 65.86% | 32.81% | 29.59% | 15.00% |
| `lambda=0.05` (10ch) | 63.46% | 30.60% | 29.21% | 15.66% |
| `lambda=0.20` (7ch) | 65.86% | 31.64% | **32.58%** | **16.95%** |

# 5. 分析与结论

## 5.1 关键发现

1. **完整 LiDAR 属性未带来总体提升**：启用 `xyzIRNAXYZ` 后，最终 Eval mIoU 为 29.21%，略低于 `lambda=0.05` 的 7 通道组 29.59%。

2. **background 下降最明显**：按日志真实含义，`[class 0]` 是 episodic background，其 IoU 从 61.29% 降至 47.92%。这说明新增属性可能影响了前景/背景的 episode 内部区分，而不是直接说明 terrain 表达显著退化。

3. **terrain 基本持平略降**：terrain IoU 从 44.18% 降至 42.76%，下降 1.42%，幅度较小。

4. **stem 小幅提升但不足以改变总体结果**：stem IoU 从 15.00% 升至 15.66%，提升 0.66%，仍然是当前 fold 的主要瓶颈。

5. **训练最佳点出现较早**：10 通道组在 6000 iteration 达到最佳 30.60%，后续没有继续突破，说明当前训练设置下新增属性没有转化为稳定收益。

## 5.2 可能原因

- **预训练质量略降**：10 通道预训练最佳 mIoU 为 63.46%，低于 7 通道历史结果 65.86%，下游少样本训练的编码器初始化可能略弱。
- **新增属性未被有效建模**：R/N/A 是传感器采集属性，直接与 xyz/I 拼接后进入 DGCNN，不一定能自动转化为有利的语义特征。
- **属性归一化和融合方式仍较粗**：当前是简单拼接，缺少针对 LiDAR 属性的单独编码、门控或注意力融合，可能导致有效几何信息被稀释。
- **单次实验不足以证明属性本身无效**：目前只完成了 `lambda=0.05` 的单次实验，还不能排除随机种子、预训练波动和超参数未适配的影响。

## 5.3 当前评估状态

| 评估类型 | 状态 | 说明 |
|----------|:----:|------|
| 启用完整 LiDAR 属性 + `lambda=0.05` | 已完成 | `xyzIRNAXYZ`，10 通道 |
| 重新预训练 10 通道编码器 | 已完成 | 最佳 mIoU 63.46% |
| 最终 protoeval | 已完成 | mIoU 29.21% |
| 与类别平衡损失组合 | 未做 | `xyzIRNAXYZ + BLoss` 待验证 |
| 不同 `lambda` 搭配完整 LiDAR | 未做 | 目前只测了 `lambda=0.05` |

## 5.4 结论

**在当前配置下，启用完整 LiDAR 属性（`xyzIRNAXYZ`）没有带来性能提升。** 相比 7 通道 `lambda=0.05` 组，10 通道组的最终 mIoU 基本持平但略低，主要问题体现在 episodic background IoU 明显下降；terrain 小幅下降，stem 小幅上升但仍处于低位。

后续若继续探索完整 LiDAR 属性，建议：

1. **先做单属性消融**：分别测试 `xyzIRXYZ`、`xyzINXYZ`、`xyzIAXYZ`，判断 R/N/A 哪个真正有效。
2. **调整属性融合方式**：不要只做通道拼接，可考虑为 R/N/A 增加轻量 MLP、门控融合或 attention 权重。
3. **延长或重跑预训练**：确认 10 通道预训练 mIoU 下降不是单次随机波动。
4. **再与类别平衡损失组合**：stem 仍是瓶颈，完整 LiDAR 属性单独使用收益有限，可以考虑与类别平衡损失联合验证。
