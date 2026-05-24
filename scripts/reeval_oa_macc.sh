#!/bin/bash
# ============================================================================
# Batch re-eval script: re-run protoeval for all experiments with OA/mAcc logging
#
# Prerequisites:
#   1. runs/eval.py has been modified to compute OA and mAcc
#   2. Data symlinks exist in datasets/FORInstance/
#   3. CUDA is available
#
# Usage: bash scripts/reeval_oa_macc.sh [--dry-run]
# ============================================================================
set -uo pipefail

DRY_RUN="${1:---no-dry-run}"
if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "=== DRY RUN — commands will be printed, not executed ==="
fi

ORIG_BRANCH=$(git rev-parse --abbrev-ref HEAD)
PROJECT_ROOT=$(pwd)

# Save modified eval.py (with OA/mAcc computation)
cp runs/eval.py /tmp/eval_oa_macc.py

# ---- Core flags shared by all experiments ----
CORE_FLAGS=(
    --dataset forinstance --cvfold 0 --phase protoeval
    --use_transformer --use_supervise_prototype --use_attention --use_align
    --use_high_dgcnn --use_linear_proj
    --pc_npts 2048 --n_way 2 --k_shot 1 --n_queries 1 --n_episode_test 100
    --edgeconv_widths '[[64,64],[64,64],[64,64]]' --dgcnn_k 20
    --dgcnn_mlp_widths '[512,256]' --base_widths '[128,64]'
    --pc_augm_shift 0.1 --lr 0.001 --step_size 5000 --gamma 0.5 --trans_lr 1e-4
)

# ---- Utility functions ----
run_eval() {
    local branch="$1"
    local log_dir="$2"
    local data_path="$3"
    local pc_attribs="$4"
    shift 4
    local extra_flags=("$@")

    echo ""
    echo "====================================================================="
    echo "  Branch:     $branch"
    echo "  Log dir:    $log_dir"
    echo "  Data:       $data_path"
    echo "  Attribs:    $pc_attribs"
    echo "  Extra:      ${extra_flags[*]:-(none)}"
    echo "====================================================================="

    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "  [DRY-RUN] Would run protoeval for this experiment"
        return 0
    fi

    # Backup old log
    mkdir -p "$log_dir"
    if [ -f "$log_dir/log_protoeval.txt" ]; then
        cp "$log_dir/log_protoeval.txt" "$log_dir/log_protoeval.txt.bak.$(date +%Y%m%d_%H%M%S)"
        echo "  Backed up: $log_dir/log_protoeval.txt"
    fi

    # Switch branch and inject eval.py
    echo "  Checking out: $branch"
    git checkout "$branch"
    cp /tmp/eval_oa_macc.py runs/eval.py

    # Run protoeval
    echo "  Running protoeval..."
    python main.py "${CORE_FLAGS[@]}" \
        --save_path "$log_dir" \
        --model_checkpoint_path "$log_dir" \
        --save_path "$log_dir" \
        --data_path "$data_path" \
        --pc_attribs "$pc_attribs" \
        "${extra_flags[@]}" 2>&1 | tee -a "$log_dir/reeval_$(date +%Y%m%d_%H%M%S).log"

    # Restore branch's original eval.py
    git checkout -- runs/eval.py

    # Clean Python cache
    find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

    echo "  Done: $log_dir"
    echo ""
}

# ======================================================================
# 1. Baseline — for-instance/base (5 runs, on current branch)
# ======================================================================
echo ">>> Phase 1: Baseline runs (for-instance/base) <<<"
for run in 01 02 03 04 05; do
    run_eval "for-instance/base" \
        "logs/base/log_forinstance_PAP/log_proto_forinstance_S0_N2_K1_Att1_${run}" \
        "datasets/FORInstance/blocks_dev_bs10.0_s5.0" \
        "xyzIXYZ"
done

# ======================================================================
# 2. Full-LiDAR 9ch — exp/xyzIRNXYZ
# ======================================================================
echo ">>> Phase 2: Full-LiDAR 9ch (exp/xyzIRNXYZ) <<<"
run_eval "exp/xyzIRNXYZ" \
    "logs/full-lidar-attribs/log_forinstance_PAP_rn/log_proto_forinstance_S0_N2_K1_Att1" \
    "datasets/FORInstance/blocks_dev_bs10.0_s5.0_rn" \
    "xyzIRNXYZ"

# ======================================================================
# 3. Full-LiDAR 10ch — exp/xyzIRNAXYZ
# ======================================================================
echo ">>> Phase 3: Full-LiDAR 10ch (exp/xyzIRNAXYZ) <<<"
run_eval "exp/xyzIRNAXYZ" \
    "logs/full-lidar-attribs/log_forinstance_PAP_rna/log_proto_forinstance_S0_N2_K1_Att1" \
    "datasets/FORInstance/blocks_dev_bs10.0_s5.0_rna" \
    "xyzIRNAXYZ"

# ======================================================================
# 4. Height-Aware — for-instance/implement-height-aware-metric
# ======================================================================
echo ">>> Phase 4: Height-Aware (for-instance/implement-height-aware-metric) <<<"
run_eval "for-instance/implement-height-aware-metric" \
    "logs/implement-height-aware-metric/log_forinstance_haware/log_proto_forinstance_S0_N2_K1_Att1_HAware" \
    "datasets/FORInstance/blocks_dev_bs10.0_s5.0" \
    "xyzIXYZ" \
    "--use_height_aware"

run_eval "for-instance/implement-height-aware-metric" \
    "logs/implement-height-aware-metric/log_forinstance_haware_rerun/log_proto_forinstance_S0_N2_K1_Att1_HAware" \
    "datasets/FORInstance/blocks_dev_bs10.0_s5.0" \
    "xyzIXYZ" \
    "--use_height_aware"

# ======================================================================
# 5. HProto Ablation — exp/hpproto-ablation (5 λ values)
# ======================================================================
echo ">>> Phase 5: HProto ablation (exp/hpproto-ablation) <<<"
H_SHARED="--use_height_proto --height_proto_bins 3"

run_eval "exp/hpproto-ablation" \
    "logs/hpproto/hpproto-ablation/log_forinstance_hproto_lambda_001/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.01" \
    "datasets/FORInstance/blocks_dev_bs10.0_s5.0" \
    "xyzIXYZ" \
    $H_SHARED "--height_proto_weight" "0.01"

run_eval "exp/hpproto-ablation" \
    "logs/hpproto/hpproto-ablation/log_forinstance_hproto_lambda_003/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.03" \
    "datasets/FORInstance/blocks_dev_bs10.0_s5.0" \
    "xyzIXYZ" \
    $H_SHARED "--height_proto_weight" "0.03"

run_eval "exp/hpproto-ablation" \
    "logs/hpproto/hpproto-ablation/log_forinstance_hproto_lambda_005/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.05" \
    "datasets/FORInstance/blocks_dev_bs10.0_s5.0" \
    "xyzIXYZ" \
    $H_SHARED "--height_proto_weight" "0.05"

run_eval "exp/hpproto-ablation" \
    "logs/hpproto/hpproto-ablation/log_forinstance_hproto_lambda_010/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.10" \
    "datasets/FORInstance/blocks_dev_bs10.0_s5.0" \
    "xyzIXYZ" \
    $H_SHARED "--height_proto_weight" "0.10"

run_eval "exp/hpproto-ablation" \
    "logs/hpproto/hpproto-ablation/log_forinstance_hproto_lambda_020/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.20" \
    "datasets/FORInstance/blocks_dev_bs10.0_s5.0" \
    "xyzIXYZ" \
    $H_SHARED "--height_proto_weight" "0.20"

# ======================================================================
# 6. Balanced Loss — exp/hpproto-balanced-loss
# ======================================================================
echo ">>> Phase 6: Balanced Loss (exp/hpproto-balanced-loss) <<<"
run_eval "exp/hpproto-balanced-loss" \
    "logs/hpproto/hpproto-balanced-loss/log_forinstance_hproto_balanced/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.05_BLoss_M5.0" \
    "datasets/FORInstance/blocks_dev_bs10.0_s5.0" \
    "xyzIXYZ" \
    $H_SHARED "--height_proto_weight" "0.05" \
    "--use_balanced_loss" "--balanced_loss_max_weight" "5.0"

# ======================================================================
# 7. HProto + Full-LiDAR 10ch — exp/hpproto-xyzIRNAXYZ
# ======================================================================
echo ">>> Phase 7: HProto + Full-LiDAR (exp/hpproto-xyzIRNAXYZ) <<<"
run_eval "exp/hpproto-xyzIRNAXYZ" \
    "logs/hpproto/hpproto-xyzIRNAXYZ/log_forinstance_hproto_full_lidar/log_proto_forinstance_S0_N2_K1_Att1_HProto_B3_W0.05" \
    "datasets/FORInstance/blocks_dev_bs10.0_s5.0_rna" \
    "xyzIRNAXYZ" \
    $H_SHARED "--height_proto_weight" "0.05"

# ======================================================================
# Done — return to original branch
# ======================================================================
echo ""
echo "=============================================="
echo "  All experiments completed!"
echo "=============================================="

# Restore original branch
git checkout "$ORIG_BRANCH"

# Restore modified eval.py on original branch
cp /tmp/eval_oa_macc.py runs/eval.py

# Clean up
rm -f /tmp/eval_oa_macc.py

echo "  Returned to branch: $ORIG_BRANCH"
echo "  eval.py modification restored."
echo ""
echo "  New log_protoeval.txt files contain OA and mAcc."
echo "  Backup files (*.bak.*) saved alongside originals."
