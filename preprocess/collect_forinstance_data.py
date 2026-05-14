"""
Step 1: Convert FOR-Instance .las files to per-plot .npy files.

Mirrors the role of collect_s3dis_data.py / collect_scannet_data.py.
Only responsible for: read LAS → filter → remap labels → normalize → save .npy
Block-cutting is handled separately by forinstance2blocks.py.

Output format per file: float32 array of shape [N, 8]
    col 0-2 : x, y, z             (shifted to plot-local origin, unit: meters)
    col 3   : intensity            (min-max normalized to [0, 1])
    col 4   : return_number        (min-max normalized to [0, 1])
    col 5   : number_of_returns    (min-max normalized to [0, 1])
    col 6   : scan_angle_rank      (scaled by max abs angle to [-1, 1])
    col 7   : label                (int stored as float32; see LABEL_MAP below)

Label mapping (remapped from raw 'classification' field):
    raw 2  →  0  terrain
    raw 1  →  1  low_vegetation
    raw 4  →  2  stem
    raw 5  →  3  live_branches
    raw 6  →  4  woody_branches
    raw 0, 3  →  ignored (unclassified / out-of-boundary)

Output directory:
    datasets/FORInstance/scenes_<split>/data/<COLLECTION>_<plot_name>.npy

Usage:
    python preprocess/collect_forinstance_data.py \
        --data_path data/FORinstance_dataset \
        --collections NIBIO SCION \
        --split dev
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import laspy
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

# -----------------------------------------------------------------------
# Label mapping
# -----------------------------------------------------------------------
# Raw 'classification' field values in FOR-Instance .las files:
#   0 = Unclassified      → IGNORE
#   1 = Low-vegetation    → 1
#   2 = Terrain           → 0
#   3 = Out-points        → IGNORE  (boundary-outside, ~42 % of total pts)
#   4 = Stem              → 2
#   5 = Live branches     → 3
#   6 = Woody branches    → 4
#
# Live branches and Woody branches are kept SEPARATE (not merged) so that
# the dataloader can form a fold with them as the test classes (fold 1).
# -----------------------------------------------------------------------
LABEL_MAP = {
    2: 0,  # terrain
    1: 1,  # low_vegetation
    4: 2,  # stem
    5: 3,  # live_branches
    6: 4,  # woody_branches
}
IGNORE_LABELS = {0, 3}
NUM_CLASSES = 5
CLASS_NAMES = ["terrain", "low_vegetation", "stem", "live_branches", "woody_branches"]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def remap_labels(labels: np.ndarray) -> np.ndarray:
    """Map raw classification codes to contiguous 0-based indices.
    Points not in LABEL_MAP (i.e. ignore labels) are assigned -1.
    """
    mapped = np.full(labels.shape, -1, dtype=np.int64)
    for raw, new in LABEL_MAP.items():
        mapped[labels == raw] = new
    return mapped


def process_las_file(las_path: str, out_path: str) -> int:
    """
    Read one .las file and save a cleaned [N, 8] .npy array.

    Steps
    -----
    1. Load x, y, z, LiDAR attributes, classification from the .las file.
    2. Drop ignore labels (0 = unclassified, 3 = out-of-boundary points).
    3. Remap remaining labels to 0-4.
    4. Shift xyz so the per-plot minimum is at the origin.
    5. Normalize intensity with per-plot min-max scaling.
    6. Stack into [N, 5] and save.

    Returns the number of points saved (0 if the file is skipped).
    """
    las = laspy.read(las_path)

    # --- raw fields ---
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(
        np.float64
    )  # keep float64 for precision
    intensity = np.array(las.intensity, dtype=np.float32)
    return_number = np.array(las.return_number, dtype=np.float32)
    number_of_returns = np.array(las.number_of_returns, dtype=np.float32)
    scan_angle = np.array(las.scan_angle_rank, dtype=np.float32)
    labels_raw = np.array(las.classification, dtype=np.int64)

    # --- step 1: filter ignore labels ---
    valid = ~np.isin(labels_raw, list(IGNORE_LABELS))
    xyz = xyz[valid]
    intensity = intensity[valid]
    return_number = return_number[valid]
    number_of_returns = number_of_returns[valid]
    scan_angle = scan_angle[valid]
    labels_raw = labels_raw[valid]

    # --- step 2: remap labels, drop any residual unmapped points ---
    labels = remap_labels(labels_raw)
    valid2 = labels >= 0
    xyz = xyz[valid2]
    intensity = intensity[valid2]
    return_number = return_number[valid2]
    number_of_returns = number_of_returns[valid2]
    scan_angle = scan_angle[valid2]
    labels = labels[valid2]

    if len(xyz) == 0:
        print(f"  [WARN] No valid points in {os.path.basename(las_path)} — skipped.")
        return 0

    # --- step 3: shift xyz to plot-local origin (subtract per-plot min) ---
    xyz -= xyz.min(axis=0)
    xyz = xyz.astype(np.float32)

    # --- step 4: per-plot min-max intensity normalization ---
    i_min, i_max = float(intensity.min()), float(intensity.max())
    intensity = (intensity - i_min) / (i_max - i_min + 1e-8)

    rn_min, rn_max = float(return_number.min()), float(return_number.max())
    return_number = (return_number - rn_min) / (rn_max - rn_min + 1e-8)

    nr_min, nr_max = float(number_of_returns.min()), float(number_of_returns.max())
    number_of_returns = (number_of_returns - nr_min) / (nr_max - nr_min + 1e-8)

    angle_scale = max(float(np.max(np.abs(scan_angle))), 1.0)
    scan_angle = scan_angle / angle_scale

    # --- step 5: assemble [N, 8] ---
    # Label is stored as float32 to keep the array homogeneous;
    # cast back to int64 when loading in the dataloader.
    data = np.column_stack(
        [
            xyz,  # (N, 3)
            intensity.reshape(-1, 1),  # (N, 1)
            return_number.reshape(-1, 1),  # (N, 1)
            number_of_returns.reshape(-1, 1),  # (N, 1)
            scan_angle.reshape(-1, 1),  # (N, 1)
            labels.reshape(-1, 1).astype(np.float32),  # (N, 1)
        ]
    ).astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, data)
    return len(data)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: FOR-Instance .las → per-plot .npy"
    )
    parser.add_argument(
        "--data_path",
        default="data/FORinstance_dataset",
        help="Root directory of the raw FOR-Instance dataset "
        "(must contain data_split_metadata.csv)",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["NIBIO", "SCION"],
        help="Sub-collections to process. "
        "Options: NIBIO SCION CULS RMIT TUWIEN  (default: NIBIO SCION)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="dev",
        help="Official FOR-Instance split to process. Use 'dev' for training/"
        "validation data and 'test' only for final held-out evaluation. "
        "Default: dev.",
    )
    args = parser.parse_args()

    DATA_ROOT = args.data_path
    COLLECTIONS = set(args.collections)
    scenes_name = "scenes" if args.split == "all" else f"scenes_{args.split}"
    SAVE_PATH = os.path.join(ROOT_DIR, "datasets", "FORInstance", scenes_name, "data")
    META_CSV = os.path.join(DATA_ROOT, "data_split_metadata.csv")

    os.makedirs(SAVE_PATH, exist_ok=True)

    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"Metadata CSV not found: {META_CSV}")

    meta = pd.read_csv(META_CSV)
    meta = meta[meta["folder"].isin(COLLECTIONS)].reset_index(drop=True)
    if args.split != "all":
        meta = meta[meta["split"] == args.split].reset_index(drop=True)

    print(f"Collections : {COLLECTIONS}")
    print(f"Split       : {args.split}")
    print(f"Total plots : {len(meta)}")
    print(f"Output dir  : {SAVE_PATH}")
    print()

    class_counter = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_pts = 0
    skipped = 0

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing plots"):
        las_path = os.path.join(DATA_ROOT, row["path"])

        if not os.path.exists(las_path):
            print(f"  [WARN] Missing file: {las_path}")
            skipped += 1
            continue

        # Output filename: e.g. NIBIO_plot_1_annotated.npy
        collection = row["folder"]
        plot_stem = os.path.splitext(os.path.basename(row["path"]))[0]
        out_name = f"{collection}_{plot_stem}.npy"
        out_path = os.path.join(SAVE_PATH, out_name)

        n = process_las_file(las_path, out_path)
        if n == 0:
            skipped += 1
            continue

        # Accumulate per-class statistics from the saved file
        saved = np.load(out_path)
        lbls = saved[:, -1].astype(np.int64)
        for c in range(NUM_CLASSES):
            class_counter[c] += int(np.sum(lbls == c))
        total_pts += n

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 45)
    print("  Preprocessing complete")
    print("=" * 45)
    print(f"  Plots processed : {len(meta) - skipped} / {len(meta)}")
    print(f"  Plots skipped   : {skipped}")
    print(f"  Total points    : {total_pts:,}")
    print()
    print("  Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        ratio = class_counter[i] / total_pts if total_pts > 0 else 0.0
        print(f"    [{i}] {name:<20s}: {class_counter[i]:>10,}  ({ratio:.2%})")
    print("=" * 45)


if __name__ == "__main__":
    main()
