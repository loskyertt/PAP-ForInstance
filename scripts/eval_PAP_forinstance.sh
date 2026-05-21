# ============================================================================
# FORInstance PAP 评估脚本
# 加载训练好的 PAP 模型, 在小样本 episode 上评估点云实例分割性能。
# 运行阶段为 protoeval, 仅前向推理, 不更新模型参数。
# ============================================================================

# --- GPU & 数据集配置 ---
GPU_ID=0

DATASET='forinstance'
SPLIT=0                                                                         # 交叉验证 fold
DATA_PATH='./datasets/FORInstance/blocks_dev_bs10.0_s5.0'                       # 分块后的数据集路径

# --- 点云模型结构参数 ---
NUM_POINTS=2048                                                                  # 每个 block 采样点数
PC_ATTRIBS='xyzIXYZ'                                                             # 点云属性: xyz=坐标, I=强度, XYZ=归一化坐标
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'                                 # EdgeConv 每层输出通道数, 共3层
MLP_WIDTHS='[512, 256]'                                                          # 聚合多层特征后的 MLP 隐藏层宽度
K=20                                                                             # kNN 近邻数
BASE_WIDTHS='[128, 64]'                                                          # 原型投影头 MLP 宽度

# --- 模型加载 ---
MODEL_CHECKPOINT='./logs/log_forinstance_PAP/log_proto_forinstance_S0_N2_K1_Att1'  # PAP 训练后的模型权重路径
PRETRAIN_CHECKPOINT='./logs/log_forinstance/log_pretrain_forinstance_S0'            # 预训练权重路径 (用于初始化骨干网络)

# --- 小样本 episode 配置 ---
N_WAY=2                                                                          # episode 类别数
K_SHOT=1                                                                         # support set 每类样本数
N_QUESIES=1                                                                      # query set 每类样本数
N_TEST_EPISODES=100                                                              # 测试 episode 数

# --- 评估参数 ---
NUM_ITERS=40000                                                                  # 评估迭代次数
EVAL_INTERVAL=2000                                                               # 每隔 N 次迭代输出一次结果
LR=0.001
DECAY_STEP=5000
DECAY_RATIO=0.5

args=(--phase 'protoeval' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path "$DATA_PATH" --save_path "$MODEL_CHECKPOINT"
      --model_checkpoint_path "$MODEL_CHECKPOINT"
      --use_transformer --use_supervise_prototype
      --pretrain_checkpoint_path "$PRETRAIN_CHECKPOINT" --use_attention
      --use_align --use_high_dgcnn --pc_augm_shift 0.1
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS"
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K --use_linear_proj
      --dgcnn_mlp_widths "$MLP_WIDTHS" --base_widths "$BASE_WIDTHS"
      --n_iters $NUM_ITERS --eval_interval $EVAL_INTERVAL --batch_size 1
      --lr $LR --step_size $DECAY_STEP --gamma $DECAY_RATIO
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUESIES --n_episode_test $N_TEST_EPISODES
      --trans_lr 1e-4)

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
