"""Data Loader for Generating Tasks

Changes from original (PAP-FZS3D):
  1. Added `label_col` parameter chain through sample_pointcloud /
     sample_K_pointclouds / MyDataset / MyPretrainDataset so that
     datasets with different column layouts (e.g. FOR-Instance uses
     col 4 for labels vs. S3DIS / ScanNet col 6) work without
     changing downstream code.
  2. Added intensity channel ('I') support in sample_pointcloud so
     that pc_attribs='xyzIXYZ' works for FOR-Instance.
  3. Added 'forinstance' branch in MyDataset.__init__.
  4. Fixed deprecated np.int → np.int64.
"""

import os
import random
import math
import glob
import numpy as np
from itertools import combinations

import torch
from torch.utils.data import Dataset


# =========================================================================== #
#  Point-cloud sampling helpers
# =========================================================================== #


def sample_K_pointclouds(
    data_path,
    num_point,
    pc_attribs,
    pc_augm,
    pc_augm_config,
    scan_names,
    sampled_class,
    sampled_classes,
    is_support=False,
    label_col=6,
):
    """Sample K point clouds and the corresponding labels for one class (one way).

    Args:
        label_col : column index of the label field in the .npy block files.
                    S3DIS / ScanNet → 6,  FOR-Instance → 4.
    """
    ptclouds = []
    labels = []
    for scan_name in scan_names:
        ptcloud, label = sample_pointcloud(
            data_path,
            num_point,
            pc_attribs,
            pc_augm,
            pc_augm_config,
            scan_name,
            sampled_classes,
            sampled_class,
            support=is_support,
            label_col=label_col,
        )
        ptclouds.append(ptcloud)
        labels.append(label)

    ptclouds = np.stack(ptclouds, axis=0)
    labels = np.stack(labels, axis=0)
    return ptclouds, labels


def sample_pointcloud(
    data_path,
    num_point,
    pc_attribs,
    pc_augm,
    pc_augm_config,
    scan_name,
    sampled_classes,
    sampled_class=0,
    support=False,
    random_sample=False,
    label_col=6,
):
    """Load one block .npy file and build a fixed-size point cloud tensor.

    Args:
        label_col : column index of the label field.
                    S3DIS / ScanNet → 6 ([x,y,z,r,g,b,label]).
                    FOR-Instance    → 4 ([x,y,z,intensity,label]).
        pc_attribs: string of attribute codes.
                    Supported codes: 'xyz', 'rgb', 'XYZ', 'I'
                    'xyz' → raw 3-D coordinates (cols 0-2)
                    'rgb' → colour  (cols 3-5, scaled to [0,1])
                    'I'   → intensity (col 3, already in [0,1])
                    'XYZ' → block-normalised coordinates (computed)
    """
    sampled_classes = list(sampled_classes)
    data = np.load(os.path.join(data_path, "data", "%s.npy" % scan_name))
    N = data.shape[0]

    if "rgb" in pc_attribs and data.shape[1] < 7:
        raise ValueError(
            f"pc_attribs='{pc_attribs}' requests rgb, but {scan_name}.npy has "
            f"only {data.shape[1]} columns. FOR-Instance blocks use "
            "[x,y,z,intensity,label]; pass --pc_attribs xyzIXYZ."
        )
    # ------------------------------------------------------------------ #
    # Point sampling
    # ------------------------------------------------------------------ #
    if random_sample:
        sampled_point_inds = np.random.choice(
            np.arange(N), num_point, replace=(N < num_point)
        )
    else:
        # Ensure the sampled points contain a sufficient proportion of the
        # target class (same strategy as the original loader).
        valid_point_inds = np.nonzero(data[:, label_col] == sampled_class)[0]

        if N < num_point:
            sampled_valid_point_num = len(valid_point_inds)
        else:
            valid_ratio = len(valid_point_inds) / float(N)
            sampled_valid_point_num = int(valid_ratio * num_point)

        sampled_valid_point_inds = np.random.choice(
            valid_point_inds,
            sampled_valid_point_num,
            replace=(len(valid_point_inds) < sampled_valid_point_num),
        )
        sampled_other_point_inds = np.random.choice(
            np.arange(N), num_point - sampled_valid_point_num, replace=(N < num_point)
        )
        sampled_point_inds = np.concatenate(
            [sampled_valid_point_inds, sampled_other_point_inds]
        )

    data = data[sampled_point_inds]

    # ------------------------------------------------------------------ #
    # Extract raw fields
    # ------------------------------------------------------------------ #
    xyz = data[:, 0:3]  # always available
    rgb = data[:, 3:6]  # S3DIS / ScanNet
    intensity = data[:, 3:4]  # FOR-Instance (col 3)
    labels = data[:, label_col].astype(np.int64)  # use label_col

    # ------------------------------------------------------------------ #
    # Coordinate processing
    # ------------------------------------------------------------------ #
    xyz_min = np.amin(xyz, axis=0)
    xyz -= xyz_min  # shift block to local origin

    if pc_augm:
        xyz = augment_pointcloud(xyz, pc_augm_config)

    if "XYZ" in pc_attribs:
        xyz_min = np.amin(xyz, axis=0)
        XYZ = xyz - xyz_min
        xyz_max = np.amax(XYZ, axis=0)
        XYZ = XYZ / (xyz_max + 1e-8)  # guard against zero-range blocks

    # ------------------------------------------------------------------ #
    # Feature assembly  (order matches pc_attribs string for dim counting)
    # ------------------------------------------------------------------ #
    ptcloud = []
    if "xyz" in pc_attribs:
        ptcloud.append(xyz)
    if "rgb" in pc_attribs:
        ptcloud.append(rgb / 255.0)
    if "I" in pc_attribs:
        ptcloud.append(intensity)  # already normalised to [0, 1]
    if "XYZ" in pc_attribs:
        ptcloud.append(XYZ)
    ptcloud = np.concatenate(ptcloud, axis=1)

    # ------------------------------------------------------------------ #
    # Ground-truth labels
    # ------------------------------------------------------------------ #
    if support:
        groundtruth = (labels == sampled_class).astype(np.int64)
    else:
        groundtruth = np.zeros_like(labels)
        for i, label in enumerate(labels):
            if label in sampled_classes:
                groundtruth[i] = sampled_classes.index(label) + 1

    return ptcloud, groundtruth


def augment_pointcloud(P, pc_augm_config):
    """ "Augmentation on XYZ and jittering of everything"""
    M = np.eye(3, dtype=np.float32)
    if pc_augm_config["scale"] > 1:
        s = random.uniform(1 / pc_augm_config["scale"], pc_augm_config["scale"])
        M = np.dot(np.eye(3, dtype=np.float32) * s, M)
    if pc_augm_config["rot"] == 1:
        angle = random.uniform(0, 2 * math.pi)
        cosval = math.cos(angle)
        sinval = math.sin(angle)
        R = np.array(
            [[cosval, -sinval, 0.0], [sinval, cosval, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        M = np.dot(R, M)
    if pc_augm_config["mirror_prob"] > 0:
        if random.random() < pc_augm_config["mirror_prob"] / 2:
            M = np.dot(np.diag([-1.0, 1.0, 1.0]).astype(np.float32), M)
        if random.random() < pc_augm_config["mirror_prob"] / 2:
            M = np.dot(np.diag([1.0, -1.0, 1.0]).astype(np.float32), M)
    P[:, :3] = np.dot(P[:, :3], M.T)
    if pc_augm_config["shift"] > 0:
        shift = np.random.uniform(-pc_augm_config["shift"], pc_augm_config["shift"], 3)
        P[:, :3] += shift
    if pc_augm_config["jitter"]:
        sigma, clip = 0.01, 0.05
        P = P + np.clip(sigma * np.random.randn(*P.shape), -clip, clip).astype(
            np.float32
        )
    return P


# =========================================================================== #
#  Few-shot episode dataset
# =========================================================================== #


class MyDataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset_name,
        cvfold=0,
        num_episode=50000,
        n_way=3,
        k_shot=5,
        n_queries=1,
        phase=None,
        mode="train",
        num_point=4096,
        pc_attribs="xyz",
        pc_augm=False,
        pc_augm_config=None,
    ):
        super(MyDataset).__init__()
        self.data_path = data_path
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_queries = n_queries
        self.num_episode = num_episode
        self.phase = phase
        self.mode = mode
        self.num_point = num_point
        self.pc_attribs = pc_attribs
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config

        # ------------------------------------------------------------------ #
        # Dataset instantiation
        # ------------------------------------------------------------------ #
        if dataset_name == "s3dis":
            from dataloaders.s3dis import S3DISDataset

            self.dataset = S3DISDataset(cvfold, data_path)
        elif dataset_name == "scannet":
            from dataloaders.scannet import ScanNetDataset

            self.dataset = ScanNetDataset(cvfold, data_path)
        elif dataset_name == "forinstance":
            from dataloaders.forinstance import FORInstanceDataset

            if "rgb" in pc_attribs:
                raise ValueError(
                    "FOR-Instance does not provide RGB in this preprocessing pipeline. "
                    "Use --pc_attribs xyzIXYZ or another combination without 'rgb'."
                )
            self.dataset = FORInstanceDataset(cvfold, data_path)
        else:
            raise NotImplementedError("Unknown dataset %s!" % dataset_name)

        # label_col is read from the dataset object so that the correct column
        # is used in sample_pointcloud regardless of dataset type.
        self.label_col = self.dataset.label_col  # type: ignore

        # ------------------------------------------------------------------ #
        # Train / test class split
        # ------------------------------------------------------------------ #
        if mode == "train":
            self.classes = np.array(self.dataset.train_classes)
        elif mode == "test":
            self.classes = np.array(self.dataset.test_classes)
        else:
            raise NotImplementedError("Unknown mode %s! [Options: train/test]" % mode)

        print("MODE: {0} | Classes: {1}".format(mode, self.classes))
        self.class2scans = self.dataset.class2scans

    def __len__(self):
        return self.num_episode

    def __getitem__(self, index, n_way_classes=None):
        if n_way_classes is not None:
            sampled_classes = np.array(n_way_classes)
        else:
            sampled_classes = np.random.choice(self.classes, self.n_way, replace=False)

        support_ptclouds, support_masks, query_ptclouds, query_labels = (
            self.generate_one_episode(sampled_classes)
        )

        if self.mode == "train" and self.phase == "metatrain":
            remain_classes = list(set(self.classes) - set(sampled_classes))
            try:
                sampled_valid_classes = np.random.choice(
                    np.array(remain_classes), self.n_way, replace=False
                )
            except Exception:
                raise NotImplementedError(
                    "The number of remaining classes is less than %d_way" % self.n_way
                )

            (
                valid_support_ptclouds,
                valid_support_masks,
                valid_query_ptclouds,
                valid_query_labels,
            ) = self.generate_one_episode(sampled_valid_classes)

            return (
                support_ptclouds.astype(np.float32),
                support_masks.astype(np.int32),
                query_ptclouds.astype(np.float32),
                query_labels.astype(np.int64),
                valid_support_ptclouds.astype(np.float32),
                valid_support_masks.astype(np.int32),
                valid_query_ptclouds.astype(np.float32),
                valid_query_labels.astype(np.int64),
            )
        else:
            return (
                support_ptclouds.astype(np.float32),
                support_masks.astype(np.int32),
                query_ptclouds.astype(np.float32),
                query_labels.astype(np.int64),
                sampled_classes.astype(np.int32),
            )

    def generate_one_episode(self, sampled_classes):
        support_ptclouds = []
        support_masks = []
        query_ptclouds = []
        query_labels = []

        black_list = []  # prevent sampling the same block twice in one episode
        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            if black_list:
                all_scannames = [x for x in all_scannames if x not in black_list]
            if len(all_scannames) < self.k_shot + self.n_queries:
                raise RuntimeError(
                    f"Class {sampled_class} has only {len(all_scannames)} available "
                    f"blocks after episode de-duplication, but "
                    f"{self.k_shot + self.n_queries} are required. "
                    "Regenerate blocks with a smaller min_npts/larger data split, "
                    "or reduce k_shot/n_queries/n_way."
                )
            selected_scannames = np.random.choice(
                all_scannames, self.k_shot + self.n_queries, replace=False
            )
            black_list.extend(selected_scannames)

            query_scannames = selected_scannames[: self.n_queries]
            support_scannames = selected_scannames[self.n_queries :]

            query_ptclouds_one_way, query_labels_one_way = sample_K_pointclouds(
                self.data_path,
                self.num_point,
                self.pc_attribs,
                self.pc_augm,
                self.pc_augm_config,
                query_scannames,
                sampled_class,
                sampled_classes,
                is_support=False,
                label_col=self.label_col,
            )

            support_ptclouds_one_way, support_masks_one_way = sample_K_pointclouds(
                self.data_path,
                self.num_point,
                self.pc_attribs,
                self.pc_augm,
                self.pc_augm_config,
                support_scannames,
                sampled_class,
                sampled_classes,
                is_support=True,
                label_col=self.label_col,
            )

            query_ptclouds.append(query_ptclouds_one_way)
            query_labels.append(query_labels_one_way)
            support_ptclouds.append(support_ptclouds_one_way)
            support_masks.append(support_masks_one_way)

        support_ptclouds = np.stack(support_ptclouds, axis=0)
        support_masks = np.stack(support_masks, axis=0)
        query_ptclouds = np.concatenate(query_ptclouds, axis=0)
        query_labels = np.concatenate(query_labels, axis=0)

        return support_ptclouds, support_masks, query_ptclouds, query_labels


def batch_train_task_collate(batch):
    (
        task_train_support_ptclouds,
        task_train_support_masks,
        task_train_query_ptclouds,
        task_train_query_labels,
        task_valid_support_ptclouds,
        task_valid_support_masks,
        task_valid_query_ptclouds,
        task_valid_query_labels,
    ) = list(zip(*batch))

    task_train_support_ptclouds = np.stack(task_train_support_ptclouds)
    task_train_support_masks = np.stack(task_train_support_masks)
    task_train_query_ptclouds = np.stack(task_train_query_ptclouds)
    task_train_query_labels = np.stack(task_train_query_labels)
    task_valid_support_ptclouds = np.stack(task_valid_support_ptclouds)
    task_valid_support_masks = np.stack(task_valid_support_masks)
    task_valid_query_ptclouds = np.array(task_valid_query_ptclouds)
    task_valid_query_labels = np.stack(task_valid_query_labels)

    data = [
        torch.from_numpy(task_train_support_ptclouds).transpose(3, 4),
        torch.from_numpy(task_train_support_masks),
        torch.from_numpy(task_train_query_ptclouds).transpose(2, 3),
        torch.from_numpy(task_train_query_labels),
        torch.from_numpy(task_valid_support_ptclouds).transpose(3, 4),
        torch.from_numpy(task_valid_support_masks),
        torch.from_numpy(task_valid_query_ptclouds).transpose(2, 3),
        torch.from_numpy(task_valid_query_labels),
    ]
    return data


# =========================================================================== #
#  Static testing dataset
# =========================================================================== #


class MyTestDataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset_name,
        cvfold=0,
        num_episode_per_comb=100,
        n_way=3,
        k_shot=5,
        n_queries=1,
        num_point=4096,
        pc_attribs="xyz",
        mode="valid",
    ):
        super(MyTestDataset).__init__()

        dataset = MyDataset(
            data_path,
            dataset_name,
            cvfold=cvfold,
            n_way=n_way,
            k_shot=k_shot,
            n_queries=n_queries,
            mode="test",
            num_point=num_point,
            pc_attribs=pc_attribs,
            pc_augm=False,
        )
        self.classes = dataset.classes

        attrib_tag = (
            "%s_label%d_" % (pc_attribs, dataset.label_col)
            if dataset_name == "forinstance"
            else ""
        )
        if mode == "valid":
            test_data_path = os.path.join(
                data_path,
                "S_%d_N_%d_K_%d_%sepisodes_%d_pts_%d"
                % (cvfold, n_way, k_shot, attrib_tag, num_episode_per_comb, num_point),
            )
        elif mode == "test":
            test_data_path = os.path.join(
                data_path,
                "S_%d_N_%d_K_%d_%stest_episodes_%d_pts_%d"
                % (cvfold, n_way, k_shot, attrib_tag, num_episode_per_comb, num_point),
            )
        else:
            raise NotImplementedError("Mode (%s) is unknown!" % mode)

        if os.path.exists(test_data_path):
            self.file_names = glob.glob(os.path.join(test_data_path, "*.h5"))
            self.num_episode = len(self.file_names)
        else:
            print(
                "Test dataset (%s) does not exist...\nConstructing..." % test_data_path
            )
            os.mkdir(test_data_path)

            class_comb = list(combinations(self.classes, n_way))
            self.num_episode = len(class_comb) * num_episode_per_comb

            episode_ind = 0
            self.file_names = []
            for sampled_classes in class_comb:
                sampled_classes = list(sampled_classes)
                for _ in range(num_episode_per_comb):
                    data = dataset.__getitem__(episode_ind, sampled_classes)
                    out_filename = os.path.join(test_data_path, "%d.h5" % episode_ind)
                    write_episode(out_filename, data)
                    self.file_names.append(out_filename)
                    episode_ind += 1

    def __len__(self):
        return self.num_episode

    def __getitem__(self, index):
        return read_episode(self.file_names[index])


def batch_test_task_collate(batch):
    (
        batch_support_ptclouds,
        batch_support_masks,
        batch_query_ptclouds,
        batch_query_labels,
        batch_sampled_classes,
    ) = batch[0]

    data = [
        torch.from_numpy(batch_support_ptclouds).transpose(2, 3),
        torch.from_numpy(batch_support_masks),
        torch.from_numpy(batch_query_ptclouds).transpose(1, 2),
        torch.from_numpy(batch_query_labels.astype(np.int64)),
    ]
    return data, batch_sampled_classes


def write_episode(out_filename, data):
    support_ptclouds, support_masks, query_ptclouds, query_labels, sampled_classes = (
        data
    )
    try:
        import h5py as h5

        data_file = h5.File(out_filename, "w")
        data_file.create_dataset(
            "support_ptclouds", data=support_ptclouds, dtype="float32"
        )
        data_file.create_dataset("support_masks", data=support_masks, dtype="int32")
        data_file.create_dataset("query_ptclouds", data=query_ptclouds, dtype="float32")
        data_file.create_dataset("query_labels", data=query_labels, dtype="int64")
        data_file.create_dataset("sampled_classes", data=sampled_classes, dtype="int32")
        data_file.close()
    except ImportError:
        with open(out_filename, "wb") as f:
            np.savez_compressed(
                f,
                support_ptclouds=support_ptclouds.astype(np.float32),
                support_masks=support_masks.astype(np.int32),
                query_ptclouds=query_ptclouds.astype(np.float32),
                query_labels=query_labels.astype(np.int64),
                sampled_classes=sampled_classes.astype(np.int32),
            )
    print("\t {0} saved! | classes: {1}".format(out_filename, sampled_classes))


def read_episode(file_name):
    try:
        import h5py as h5

        data_file = h5.File(file_name, "r")
        support_ptclouds = data_file["support_ptclouds"][:]
        support_masks = data_file["support_masks"][:]
        query_ptclouds = data_file["query_ptclouds"][:]
        query_labels = data_file["query_labels"][:]
        sampled_classes = data_file["sampled_classes"][:]
        data_file.close()
    except ImportError:
        data_file = np.load(file_name)
        support_ptclouds = data_file["support_ptclouds"]
        support_masks = data_file["support_masks"]
        query_ptclouds = data_file["query_ptclouds"]
        query_labels = data_file["query_labels"]
        sampled_classes = data_file["sampled_classes"]
    return (
        support_ptclouds,
        support_masks,
        query_ptclouds,
        query_labels,
        sampled_classes,
    )


# =========================================================================== #
#  Pre-train dataset
# =========================================================================== #


class MyPretrainDataset(Dataset):
    def __init__(
        self,
        data_path,
        classes,
        class2scans,
        mode="train",
        num_point=4096,
        pc_attribs="xyz",
        pc_augm=False,
        pc_augm_config=None,
        label_col=6,
    ):  # ← NEW: default keeps S3DIS / ScanNet working
        super(MyPretrainDataset).__init__()
        self.data_path = data_path
        self.classes = classes
        self.num_point = num_point
        self.pc_attribs = pc_attribs
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config
        self.label_col = label_col  # ← stored for use in __getitem__

        train_block_names = []
        all_block_names = []
        for k, v in sorted(class2scans.items()):
            all_block_names.extend(v)
            n_blocks = len(v)
            n_test_blocks = int(n_blocks * 0.1)
            n_train_blocks = n_blocks - n_test_blocks
            train_block_names.extend(v[:n_train_blocks])

        if mode == "train":
            self.block_names = list(set(train_block_names))
        elif mode == "test":
            self.block_names = list(set(all_block_names) - set(train_block_names))
        else:
            raise NotImplementedError("Mode is unknown!")

        print(
            "[Pretrain Dataset] Mode: {0} | Num_blocks: {1}".format(
                mode, len(self.block_names)
            )
        )

    def __len__(self):
        return len(self.block_names)

    def __getitem__(self, index):
        block_name = self.block_names[index]
        ptcloud, label = sample_pointcloud(
            self.data_path,
            self.num_point,
            self.pc_attribs,
            self.pc_augm,
            self.pc_augm_config,
            block_name,
            self.classes,
            random_sample=True,
            label_col=self.label_col,
        )
        return (
            torch.from_numpy(ptcloud.transpose().astype(np.float32)),
            torch.from_numpy(label.astype(np.int64)),
        )
