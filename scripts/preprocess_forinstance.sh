RAW_DATA_PATH='./data/FORinstance_dataset'

python preprocess/collect_forinstance_data.py \
  --data_path "$RAW_DATA_PATH" \
  --collections NIBIO SCION \
  --split dev

python preprocess/forinstance2blocks.py \
  --data_path './datasets/FORInstance/scenes_dev' \
  --block_size 10.0 \
  --stride 5.0 \
  --min_npts 1024 \
  --overwrite \
  --tag _rna
