"""Data Preprocess and Loader for FOR-Instance Dataset

Mirrors the structure of s3dis.py / scannet.py exactly so that
loader.py can treat it as a drop-in replacement.

Data format expected (output of collect_forinstance_data.py +
forinstance2blocks.py):
    float32 [N, 7]
        col 0-2 : x, y, z              (block-local coordinates, meters)
        col 3   : intensity             (min-max normalized to [0, 1])
        col 4   : return_number         (normalized to [0, 1])
        col 5   : number_of_returns     (normalized to [0, 1])
        col 6   : label                 (0-4, stored as float32; cast to int64 on load)

This loader also accepts the legacy 5-column format (label in col 4) and
the earlier 8-column FOR-Instance blocks (label in col 7).

Label mapping (remapped from raw 'classification' field):
    0 → terrain
    1 → low_vegetation
    2 → stem
    3 → live_branches
    4 → woody_branches

Fold definitions (cvfold argument):
    fold 0  test : terrain (0), stem (2)
                   — largest geometric contrast, easy to distinguish
            train: low_vegetation (1), live_branches (3), woody_branches (4)

    fold 1  test : live_branches (3), woody_branches (4)
                   — high structural similarity, harder evaluation
            train: terrain (0), low_vegetation (1), stem (2)
"""

import os
import glob
import pickle
import numpy as np


class FORInstanceDataset(object):
    # Column index of the label field in the preprocessed .npy blocks.
    # S3DIS / ScanNet use col 6 ([x,y,z,r,g,b,label]).
    # FOR-Instance: auto-detected — col 6 for 7-col, col 4 for 5-col, col 7 for 8-col legacy.
    LABEL_COL = None
    CLASS_NAMES = ["terrain", "low_vegetation", "stem", "live_branches", "woody_branches"]

    def __init__(self, cvfold: int, data_path: str):
        """
        Args:
            cvfold    : cross-validation fold index (0 or 1).
            data_path : path to the blocks directory, e.g.
                        'datasets/FORInstance/blocks_bs10.0_s5.0'
        """
        self.data_path = data_path
        self.classes = 5  # number of valid semantic classes
        self.label_col = self._detect_label_col()
        print(f"[FORInstanceDataset] label_col={self.label_col}")

        # ------------------------------------------------------------------ #
        # Class name ↔ integer mapping (read from classnames.txt)
        # File is expected at  <parent_of_data_path>/meta/forinstance_classnames.txt
        # ------------------------------------------------------------------ #
        meta_dir = os.path.join(os.path.dirname(data_path), "meta")
        names_file = os.path.join(meta_dir, "forinstance_classnames.txt")
        if not os.path.exists(names_file):
            raise FileNotFoundError(
                f"Class-names file not found: {names_file}\n"
                f"Expected content (one name per line, 0-indexed):\n"
                f"  terrain\n  low_vegetation\n  stem\n"
                f"  live_branches\n  woody_branches"
            )

        class_names = [l.strip() for l in open(names_file).readlines()]
        if class_names != self.CLASS_NAMES:
            raise ValueError(
                "FOR-Instance class name order must match the preprocessing label map.\n"
                f"Expected: {self.CLASS_NAMES}\n"
                f"Found   : {class_names}\n"
                f"Please fix {names_file} and rebuild class2scans."
            )
        self.class2type = {i: name for i, name in enumerate(class_names)}
        self.type2class = {name: i for i, name in self.class2type.items()}
        self.types = self.type2class.keys()

        # ------------------------------------------------------------------ #
        # Fold definitions
        # ------------------------------------------------------------------ #
        self.fold_0 = ["terrain", "stem"]
        self.fold_1 = ["live_branches", "woody_branches"]

        if cvfold == 0:
            self.test_classes = [self.type2class[n] for n in self.fold_0]
        elif cvfold == 1:
            self.test_classes = [self.type2class[n] for n in self.fold_1]
        else:
            raise NotImplementedError(f"Unknown cvfold ({cvfold}). [Options: 0, 1]")

        all_classes = list(range(self.classes))
        self.train_classes = [c for c in all_classes if c not in self.test_classes]

        print(f"[FORInstanceDataset] cvfold={cvfold}")
        print(f"  train_classes : {[self.class2type[c] for c in self.train_classes]}")
        print(f"  test_classes  : {[self.class2type[c] for c in self.test_classes]}")

        # ------------------------------------------------------------------ #
        # Build (or load cached) class → block-name mapping
        # ------------------------------------------------------------------ #
        self.class2scans = self._get_class2scans()

    # ---------------------------------------------------------------------- #
    # class2scans construction
    # ---------------------------------------------------------------------- #
    def _get_class2scans(self) -> dict:
        """Return a dict mapping class_id → list of block names that
        contain a sufficient number of points for that class.

        The result is cached inside data_path so that subsequent runs skip the
        expensive glob + np.load loop. The cache name includes the label column
        and thresholds to avoid reusing stale mappings after preprocessing
        changes.
        """
        cache_file = os.path.join(
            self.data_path, f"class2scans_label{self.label_col}_min5pct_100pts.pkl"
        )

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                class2scans = pickle.load(f)
            print(f"[FORInstanceDataset] Loaded class2scans from cache: {cache_file}")
            return class2scans

        # ---- build from scratch ---- #
        min_ratio = 0.05  # a class needs ≥ 5 % of block points to qualify
        min_pts = 100  # … or at least 100 absolute points

        class2scans = {k: [] for k in range(self.classes)}

        block_files = sorted(glob.glob(os.path.join(self.data_path, "data", "*.npy")))
        if not block_files:
            raise FileNotFoundError(
                f"No .npy block files found under "
                f"{os.path.join(self.data_path, 'data')}/\n"
                f"Run forinstance2blocks.py first."
            )

        print(
            f"[FORInstanceDataset] Building class2scans from "
            f"{len(block_files)} blocks …"
        )

        for fpath in block_files:
            scan_name = os.path.basename(fpath)[:-4]  # strip .npy
            data = np.load(fpath)

            # label is stored as float32; cast to int for comparison
            labels = data[:, self.label_col].astype(np.int64)
            classes = np.unique(labels)

            print(f"  {scan_name} | shape: {data.shape} | classes: {list(classes)}")

            for class_id in classes:
                num_points = int(np.sum(labels == class_id))
                threshold = max(int(data.shape[0] * min_ratio), min_pts)
                if num_points > threshold:
                    class2scans[class_id].append(scan_name)

        print("==== class2scans mapping complete ====")
        for cid in range(self.classes):
            print(
                f"  class {cid} ({self.class2type[cid]:<20s}): "
                f"{len(class2scans[cid])} blocks"
            )

        with open(cache_file, "wb") as f:
            pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
        print(f"  → cached to {cache_file}")

        return class2scans

    def _detect_label_col(self) -> int:
        """Detect the semantic-label column for FOR-Instance block files.

        Current preprocessing writes 7 columns:
            [x, y, z, intensity, return_number, number_of_returns, label]
        with the label (0-4) in the last column (index 6).

        Legacy formats still supported: 5-col [x,y,z,intensity,label] and
        an older 8-col format with label in col 7.
        """
        block_files = sorted(glob.glob(os.path.join(self.data_path, "data", "*.npy")))
        if not block_files:
            return 4

        sample = np.load(block_files[0], mmap_mode="r")
        if sample.ndim != 2:
            raise ValueError(
                f"FOR-Instance block must be a 2-D array, got shape {sample.shape} "
                f"for {block_files[0]}"
            )

        ncols = sample.shape[1]

        # Check candidate label columns from rightmost to leftmost.
        # For each, verify the values are valid class indices (0-4).
        candidates = [
            (7, "8-col legacy"),  # legacy 8-column format
            (6, "7-col echo"),    # current format with return_number + number_of_returns
            (4, "5-col"),         # baseline format without echo attributes
        ]
        for candidate_col, desc in candidates:
            if ncols > candidate_col:
                labels = sample[:, candidate_col].astype(np.int64)
                if np.all((labels >= 0) & (labels < self.classes)):
                    return candidate_col

        raise ValueError(
            "Could not detect FOR-Instance label column. "
            f"Got shape {sample.shape} for {block_files[0]}. "
            "Expected one of: 7-col [x,y,z,I,R,N,label], "
            "5-col [x,y,z,I,label], or 8-col legacy."
        )
