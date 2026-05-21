# ============================================================================
# FORInstance 预训练脚本
# 在点云分割数据上预训练 DGCNN 骨干网络，学习点云特征表示。
# 预训练权重会被下游的 PAP 小样本训练加载使用。
# ============================================================================

# --- GPU & 数据集配置 ---
GPU_ID=0

DATASET='forinstance'
SPLIT=0                                                                         # 交叉验证 fold
DATA_PATH='./datasets/FORInstance/blocks_dev_bs10.0_s5.0'                       # 分块后的数据集路径
SAVE_PATH='./logs/log_forinstance/'                                              # 模型保存目录

# --- 点云模型结构参数 ---
NUM_POINTS=2048                                                                  # 每个 block 采样点数
PC_ATTRIBS='xyzIXYZ'                                                             # 点云属性: xyz=坐标, I=强度, XYZ=归一化坐标
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'                                 # EdgeConv 每层输出通道数, 共3层
MLP_WIDTHS='[512, 256]'                                                          # 聚合多层特征后的 MLP 隐藏层宽度
K=20                                                                             # kNN 近邻数

# --- 预训练超参数 ---
EVAL_INTERVAL=3                                                                  # 每隔 N 个 epoch 评估一次
BATCH_SIZE=16
NUM_WORKERS=4
NUM_EPOCHS=50
LR=0.001
WEIGHT_DECAY=0.0001
DECAY_STEP=50                                                                    # 学习率衰减步长 (epoch)
DECAY_RATIO=0.5                                                                  # 学习率衰减因子 (gamma)

args=(--phase 'pretrain' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path "$DATA_PATH" --save_path "$SAVE_PATH"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K
      --dgcnn_mlp_widths "$MLP_WIDTHS" --use_high_dgcnn --pc_augm_scale 1.25 --pc_augm_shift 0.1
      --n_iters $NUM_EPOCHS --eval_interval $EVAL_INTERVAL
      --batch_size $BATCH_SIZE --n_workers $NUM_WORKERS
      --pretrain_lr $LR --pretrain_weight_decay $WEIGHT_DECAY
      --pretrain_step_size $DECAY_STEP --pretrain_gamma $DECAY_RATIO)

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
