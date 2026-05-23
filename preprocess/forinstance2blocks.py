"""
Step 2: Split per-plot .npy files into fixed-size overlapping blocks.

Mirrors the role of room2blocks.py, adapted for large-scale outdoor forest plots.

Key differences from room2blocks.py
-------------------------------------
  room2blocks.py          forinstance2blocks.py
  block_size = 1 m        block_size = 10 m  (forest plots are 20–50 m wide)
  stride     = 1 m        stride     = 5 m   (50 % overlap to avoid cutting trees)
  min_npts   = 1000       min_npts   = 1024  (same intent, outdoor scenes are sparser)

Input  (output of collect_forinstance_data.py):
    datasets/FORInstance/scenes_dev/data/*.npy
    Each file: float32 [N, 8]
        col 0-2 : x, y, z              (plot-local coordinates, meters)
        col 3   : intensity             (normalized)
        col 4   : return_number         (normalized)
        col 5   : number_of_returns     (normalized)
        col 6   : scan_angle_rank       (normalized)
        col 7   : label                 (0-4, stored as float32)

Output:
    datasets/FORInstance/blocks_dev_bs<size>_s<stride>/data/*.npy
    Each file: float32 [M, 8]   (M varies; dataloader samples to fixed pc_npts)
    Same column layout as input — no additional normalization here.
    Block-level XY centering is performed in the dataloader, not here,
    to keep this script consistent with room2blocks.py.

Usage:
    python preprocess/forinstance2blocks.py \
        --data_path datasets/FORInstance/scenes_dev \
        --block_size 10.0 \
        --stride 5.0 \
        --min_npts 1024
"""

import os
import sys
import glob
import argparse
import shutil

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


# -----------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------


def plot2blocks(
    data: np.ndarray, block_size: float, stride: float, min_npts: int
) -> list:
    """
    Split one plot's point cloud into overlapping blocks using a sliding window.

    Mirrors room2blocks() in room2blocks.py with the same algorithm;
    only the default parameters differ (larger block / stride for outdoor scenes).

    Args:
        data      : float32 [N, 8] — x, y, z, intensity, return_number,
                    number_of_returns, scan_angle_rank, label
        block_size: physical edge length of each block (meters)
        stride    : sliding step (meters); stride <= block_size gives overlap
        min_npts  : discard blocks with fewer points than this threshold

    Returns:
        list of float32 arrays, each [M, 8]  (M varies per block)
    """
    assert stride <= block_size, (
        f"stride ({stride}) must be <= block_size ({block_size})"
    )

    # Work on XY only for the sliding window; Z is kept as-is.
    # Shift the entire plot to origin first (same as room2blocks.py).
    # Note: xyz is a VIEW of data[:, :3], so this modifies data in-place.
    xyz = data[:, :3]
    xyz_min = xyz.min(axis=0)
    xyz -= xyz_min  # shift plot to local origin
    xyz_max = xyz.max(axis=0)

    # Number of blocks along X and Y
    num_block_x = max(1, int(np.ceil((xyz_max[0] - block_size) / stride)) + 1)
    num_block_y = max(1, int(np.ceil((xyz_max[1] - block_size) / stride)) + 1)

    blocks = []
    for i in range(num_block_x):
        for j in range(num_block_y):
            xbeg = i * stride
            ybeg = j * stride
            xcond = (xyz[:, 0] >= xbeg) & (xyz[:, 0] <= xbeg + block_size)
            ycond = (xyz[:, 1] >= ybeg) & (xyz[:, 1] <= ybeg + block_size)
            mask = xcond & ycond

            if mask.sum() < min_npts:
                continue

            blocks.append(data[mask, :].copy())  # copy to avoid view issues

    return blocks


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Split FOR-Instance per-plot .npy into blocks"
    )
    parser.add_argument(
        "--data_path",
        default="datasets/FORInstance/scenes_dev",
        help="Path to the scenes directory produced by collect_forinstance_data.py "
        "(must contain a data/ subdirectory with *.npy files)",
    )
    parser.add_argument(
        "--block_size",
        type=float,
        default=10.0,
        help="Block edge length in meters. "
        "Use 10 m for typical forest plots (default: 10.0)",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=5.0,
        help="Sliding step in meters. "
        "stride < block_size gives overlapping blocks (default: 5.0 = 50 %% overlap)",
    )
    parser.add_argument(
        "--min_npts",
        type=int,
        default=1024,
        help="Minimum number of points a block must contain; "
        "sparser blocks are discarded (default: 1024)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output data directory before writing blocks. "
        "Recommended after changing split/collections/block parameters.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix appended to the output directory name, "
        "e.g. --tag _rn  →  blocks_dev_bs10.0_s5.0_rn. "
        "Use this to separate blocks with different feature sets.",
    )
    args = parser.parse_args()

    # ---- paths ----
    DATA_PATH = (
        os.path.join(ROOT_DIR, args.data_path)
        if not os.path.isabs(args.data_path)
        else args.data_path
    )
    scene_tag = os.path.basename(os.path.normpath(DATA_PATH))
    if scene_tag.startswith("scenes_"):
        split_tag = scene_tag.replace("scenes_", "", 1)
        blocks_name = f"blocks_{split_tag}_bs{args.block_size}_s{args.stride}{args.tag}"
    elif scene_tag == "scenes":
        blocks_name = f"blocks_bs{args.block_size}_s{args.stride}{args.tag}"
    else:
        blocks_name = f"{scene_tag}_blocks_bs{args.block_size}_s{args.stride}{args.tag}"

    SAVE_PATH = os.path.join(os.path.dirname(DATA_PATH), blocks_name, "data")
    if os.path.exists(SAVE_PATH) and args.overwrite:
        shutil.rmtree(SAVE_PATH)
    os.makedirs(SAVE_PATH, exist_ok=True)

    file_paths = sorted(glob.glob(os.path.join(DATA_PATH, "data", "*.npy")))
    if not file_paths:
        raise FileNotFoundError(
            f"No .npy files found under {DATA_PATH}/data/\n"
            f"Run collect_forinstance_data.py first."
        )

    print(f"Input dir   : {os.path.join(DATA_PATH, 'data')}")
    print(f"Output dir  : {SAVE_PATH}")
    print(f"block_size  : {args.block_size} m")
    print(
        f"stride      : {args.stride} m  ({100 * (1 - args.stride / args.block_size):.0f} %% overlap)"
    )
    print(f"min_npts    : {args.min_npts}")
    print(f"Plots found : {len(file_paths)}")
    print()

    total_blocks = 0

    for fp in file_paths:
        plot_name = os.path.basename(fp)[:-4]  # strip .npy
        data = np.load(fp)  # [N, 8]

        blocks = plot2blocks(data, args.block_size, args.stride, args.min_npts)

        print(f"  {plot_name}: {len(data):>8,} pts  →  {len(blocks):>4} blocks")

        for k, blk in enumerate(blocks):
            out_path = os.path.join(SAVE_PATH, f"{plot_name}_block_{k}.npy")
            np.save(out_path, blk.astype(np.float32))

        total_blocks += len(blocks)

    print()
    print("=" * 45)
    print("  Block splitting complete")
    print("=" * 45)
    print(f"  Total blocks saved : {total_blocks}")
    print(f"  Output directory   : {SAVE_PATH}")
    print("=" * 45)


if __name__ == "__main__":
    main()
