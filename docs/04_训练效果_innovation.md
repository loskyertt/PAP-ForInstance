> [!abstract]
> 引入创新点后的 PAP 模型后的训练效果分析。

# 1. 第一次训练 — 高度分层残差原型 (2026-05-14)

对应 `logs/log_forinstance_hproto/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.20/` 目录。

## 1.1 配置说明

| 参数 | 值 | 说明 |
|------|:---:|------|
| `--use_height_proto` | True | 启用高度分层原型 |
| `--height_proto_bins` | 3 | 垂直分层数 B=3 |
| `--height_proto_weight` | 0.2 | 高度相似度融合权重 λ=0.2 |
| `--use_balanced_loss` | False | 未启用类别平衡损失 |
| `--pc_attribs` | xyzIXYZ | 7 通道（与 adapted 基线一致） |
| `pretrain_checkpoint` | `./logs/log_forinstance/` | 使用与 adapted 第二次训练相同的预训练权重 |
| 其他参数 | — | 与 adapted 基线完全一致（2-way 1-shot, 40000 iters, 含 QGPA/SR/Alignment） |

## 1.2 PAP 小样本训练

> `logs/log_forinstance_hproto/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.20/log_prototrain.txt`

- **测试类**: class 0 (terrain) 和 class 2 (stem)
- **训练过程**: 训练 Loss 在 0.2~1.0 之间波动，Accuracy 稳定在 90%-97%
- **评估结果 (每 2000 iters)**:

| Iteration | class 0 terrain IoU | class 1 (bg) IoU | class 2 stem IoU | Best mIoU |
|:---------:|:-------------------:|:----------------:|:----------------:|:---------:|
| 2000 | 59.1% | 40.3% | 17.5% | **28.9%** |
| 4000 | 61.6% | 41.6% | 18.0% | **29.8%** |
| 6000 | 57.4% | 38.6% | 18.4% | 29.8% |
| 8000 | **62.3%** | 41.5% | **20.7%** | **31.1%** |
| 10000 | 59.3% | 39.5% | 21.0% | 31.1% |
| 12000 | 61.0% | 40.4% | 20.2% | 31.1% |
| 14000 | 54.7% | 43.1% | 18.7% | 31.1% |
| 16000 | 55.9% | 43.5% | 19.0% | **31.2%** |
| 18000 | 56.3% | **44.7%** | 18.6% | **31.6%** |
| 20000 | 56.3% | 44.7% | 18.6% | 31.6% |
| ... 收敛期 ... | 53~60% | 41~44% | 18~21% | **31.6%** (收敛) |
| 40000 | 53.7% | 42.6% | 18.8% | 31.6% |

## 1.3 最终评估

> `logs/log_forinstance_hproto/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.20/log_protoeval.txt`

训练完成后，加载最佳 checkpoint 在 100 个固定测试 episode 上评估：

- **class 0 (terrain) IoU: 50.29%**
- **class 1 (background) IoU: 48.21%**
- **class 2 (stem) IoU: 16.95%**
- **Mean IoU: 32.58%**

## 1.4 训练效果分析

| 指标 | 值 | 评价 |
|------|:---:|:----|
| 小样本 terrain IoU | **50.29%** | 一般，低于 adapted 基线的 66.40% |
| 小样本 background IoU | **48.21%** | 一般，低于 adapted 基线的 59.90% |
| 小样本 stem IoU | **16.95%** | ❌ **较差**，与 adapted 基线（15.18%）接近 |
| 总体 mIoU | **32.58%** | 低于 adapted 基线的 37.54% |

---

# 2. 与 adapted 基线对比分析

## 2.1 效果对比

| 指标 | adapted 基线 | 创新点 (hproto λ=0.2) | 差异 |
|------|:-----------:|:---------------------:|:----:|
| **训练中最佳 mIoU** | **38.09%** | **31.64%** | **-6.45%** |
| **最终 eval mIoU (100 ep.)** | **37.54%** | **32.58%** | **-4.96%** |
| class 0 terrain | 66.40% | 50.29% | -16.11% |
| class 1 background | 59.90% | 48.21% | -11.69% |
| class 2 stem | 15.18% | 16.95% | +1.77% |

## 2.2 结果分析

当前创新点训练结果**低于 adapted 基线**，可能原因：

### 1. 高度原型融合权重 λ=0.2 过大

当前使用 `--height_proto_weight 0.2`，论文中最佳设置为 λ=0.05。λ 过大时高度相似度会过度主导最终得分，削弱 QGPA 精炼原型的判别性特征，导致 terrain 和 background 的 IoU 显著下降。

### 2. 未启用完整的 LiDAR 属性

训练使用 `xyzIXYZ`（7 通道），而非完整 `xyzIRNAXYZ`（11 通道）。`return_number`、`number_of_returns`、`scan_angle_rank` 等 LiDAR 属性未参与特征编码，可能限制了高度原型的有效性。

### 3. 未启用类别平衡损失

`--use_balanced_loss` 为 False，stem 类的类别不平衡问题未得到缓解。

### 4. Stem 类略有提升

Stem IoU 从 15.18% 提升到 16.95%（+1.77%），说明高度分层原型对**树干**这类贯穿多个高度层的结构提供了额外信息，但提升幅度有限。

## 2.3 当前评估状态

| 评估类型 | 状态 | 说明 |
|----------|:----:|------|
| ✅ **训练中评估** (Training-time eval) | **已完成** | 全过程记录 |
| ✅ **最终独立评估** (Post-training eval) | **已完成** | 100 episodes，可靠 |
| ❌ **跨 fold 评估** (Cross-fold eval) | **未做** | 仅 fold 0 |
| ❌ **消融实验** | **未做** | 需测试不同 λ/ B 值组合 |
| ❌ **启用完整 LiDAR 属性** | **未做** | 当前仍用 xyzIXYZ 而非 xyzIRNAXYZ |

---

# 3. 结论与建议

**当前结论**：高度分层残差原型（λ=0.2）在 FOR-Instance 2-way 1-shot 任务上获得 **32.58% mIoU**，低于 adapted 基线的 37.54%。stem 类有微弱提升（+1.77%），但 terrain 和 background 下降明显。

**后续优化方向**：

1. **调低 λ 权重**：将 `--height_proto_weight` 从 0.2 降至 0.05（论文推荐值），使高度相似度作为温和的残差修正而非主导信号
2. **启用完整 LiDAR 属性**：使用 `--pc_attribs xyzIRNAXYZ`，利用 return_number、number_of_returns、scan_angle_rank 提供更丰富的特征
3. **启用类别平衡损失**：添加 `--use_balanced_loss`，缓解 stem 类点数稀少的问题
4. **分步消融**：逐一启用上述改进，观察各模块的独立贡献
