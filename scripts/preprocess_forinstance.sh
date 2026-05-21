# ============================================================================
# FORInstance 数据预处理脚本
# 步骤1: 从原始数据集收集指定 collection 的点云场景数据
# 步骤2: 将场景切分为固定大小的 block, 供模型训练使用
# ============================================================================

RAW_DATA_PATH='./data/FORinstance_dataset'

# 步骤1: 收集原始点云数据, 按 collection 和 split 筛选
python preprocess/collect_forinstance_data.py \
  --data_path "$RAW_DATA_PATH" \
  --collections NIBIO SCION \
  --split dev

# 步骤2: 将场景切分为 block
#   block_size: 每个 block 边长 (米)
#   stride:    相邻 block 中心间距; stride < block_size 时产生重叠
#   min_npts:  过滤掉点数过少的 block
python preprocess/forinstance2blocks.py \
  --data_path './datasets/FORInstance/scenes_dev' \
  --block_size 10.0 \
  --stride 5.0 \
  --min_npts 1024 \
  --overwrite
