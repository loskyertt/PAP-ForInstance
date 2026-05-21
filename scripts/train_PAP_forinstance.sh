# ============================================================================
# FORInstance PAP 小样本训练脚本
# 在预训练骨干网络基础上进行原型对齐训练 (Prototype Alignment Pretraining),
# 使点云特征在小样本分割任务上具有更好的判别能力。
# ============================================================================

# --- GPU & 数据集配置 ---
GPU_ID=0

DATASET='forinstance'
SPLIT=0                                                                         # 交叉验证 fold
DATA_PATH='./datasets/FORInstance/blocks_dev_bs10.0_s5.0'                       # 分块后的数据集路径
SAVE_PATH='./logs/log_forinstance_PAP/'                                          # 模型保存目录

# --- 点云模型结构参数 ---
NUM_POINTS=2048                                                                  # 每个 block 采样点数
PC_ATTRIBS='xyzIXYZ'                                                             # 点云属性: xyz=坐标, I=强度, XYZ=归一化坐标
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'                                 # EdgeConv 每层输出通道数, 共3层
MLP_WIDTHS='[512, 256]'                                                          # 聚合多层特征后的 MLP 隐藏层宽度
K=20                                                                             # kNN 近邻数
BASE_WIDTHS='[128, 64]'                                                          # 原型投影头 MLP 宽度

# --- 小样本 episode 配置 ---
PRETRAIN_CHECKPOINT='./logs/log_forinstance/log_pretrain_forinstance_S0'         # 预训练模型权重路径
N_WAY=2                                                                          # episode 类别数
K_SHOT=1                                                                         # support set 每类样本数
N_QUESIES=1                                                                      # query set 每类样本数
N_TEST_EPISODES=100                                                              # 测试 episode 数

# --- 训练超参数 ---
NUM_ITERS=25000                                                                  # 训练迭代次数
EVAL_INTERVAL=2000                                                               # 每隔 N 次迭代评估一次
LR=0.001
DECAY_STEP=5000                                                                  # 学习率衰减步长 (iteration)
DECAY_RATIO=0.5                                                                  # 学习率衰减因子 (gamma)

args=(--phase 'prototrain' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path "$DATA_PATH" --save_path "$SAVE_PATH"
      --use_transformer --use_supervise_prototype
      --pretrain_checkpoint_path "$PRETRAIN_CHECKPOINT" --use_attention
      --use_align --use_high_dgcnn --pc_augm_shift 0.1
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K --use_linear_proj
      --dgcnn_mlp_widths "$MLP_WIDTHS" --base_widths "$BASE_WIDTHS"
      --n_iters $NUM_ITERS --eval_interval $EVAL_INTERVAL --batch_size 1
      --lr $LR --step_size $DECAY_STEP --gamma $DECAY_RATIO
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUESIES --n_episode_test $N_TEST_EPISODES
      --trans_lr 1e-4)

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
