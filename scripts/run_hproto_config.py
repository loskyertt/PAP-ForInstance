"""Build PAP FOR-Instance hproto commands from a small YAML config.

This helper keeps lambda ablations reproducible without starting a run during
validation. Use --dry-run to print the command that would be executed.

# 1. 数据预处理，若已完成可跳过
bash scripts/preprocess_forinstance.sh

# 2. 预训练，若已有 ./logs/log_forinstance/log_pretrain_forinstance_S0 可跳过
bash scripts/pretrain_forinstance.sh

# 3. 运行某个 λ 的原型训练，例如 λ=0.05
python scripts/run_hproto_config.py configs/hproto/lambda_005.yaml --phase train

# 4. 训练完成后评估同一个 λ
python scripts/run_hproto_config.py configs/hproto/lambda_005.yaml --phase eval
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULTS = {
    "dataset": "forinstance",
    "cvfold": 0,
    "data_path": "./datasets/FORInstance/blocks_dev_bs10.0_s5.0",
    "save_path": "./logs/log_forinstance_hproto/",
    "pretrain_checkpoint_path": "./logs/log_forinstance/log_pretrain_forinstance_S0",
    "model_checkpoint_path": "",
    "pc_npts": 2048,
    "pc_attribs": "xyzIXYZ",
    "edgeconv_widths": "[[64,64], [64, 64], [64, 64]]",
    "dgcnn_mlp_widths": "[512, 256]",
    "dgcnn_k": 20,
    "base_widths": "[128, 64]",
    "n_iters": 40000,
    "eval_interval": 2000,
    "batch_size": 1,
    "lr": 0.001,
    "step_size": 5000,
    "gamma": 0.5,
    "n_way": 2,
    "k_shot": 1,
    "n_queries": 1,
    "n_episode_test": 100,
    "trans_lr": 0.0001,
    "pc_augm_shift": 0.1,
    "use_transformer": True,
    "use_supervise_prototype": True,
    "use_attention": True,
    "use_align": True,
    "use_high_dgcnn": True,
    "use_linear_proj": True,
    "use_height_proto": True,
    "height_proto_bins": 3,
    "height_proto_weight": 0.05,
}

BOOL_FLAGS = {
    "use_transformer",
    "use_supervise_prototype",
    "use_attention",
    "use_align",
    "use_high_dgcnn",
    "use_linear_proj",
    "use_height_proto",
}

TRAIN_ONLY_FLAGS = {"pc_augm": True}

ORDERED_ARGS = [
    "dataset",
    "cvfold",
    "data_path",
    "save_path",
    "model_checkpoint_path",
    "pretrain_checkpoint_path",
    "pc_augm_shift",
    "pc_npts",
    "pc_attribs",
    "edgeconv_widths",
    "dgcnn_k",
    "dgcnn_mlp_widths",
    "base_widths",
    "n_iters",
    "eval_interval",
    "batch_size",
    "lr",
    "step_size",
    "gamma",
    "n_way",
    "k_shot",
    "n_queries",
    "n_episode_test",
    "trans_lr",
    "height_proto_bins",
    "height_proto_weight",
]


def parse_scalar(value: str):
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return ""
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_flat_yaml(path: Path) -> dict:
    config = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_no} is not a simple 'key: value' entry")
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_no} has an empty key")
        config[key] = parse_scalar(value)
    return config


def proto_log_name(cfg: dict) -> str:
    return (
        "log_proto_%s_S%d_N%d_K%d_Att%d_HProto_B%d_W%.2f"
        % (
            cfg["dataset"],
            int(cfg["cvfold"]),
            int(cfg["n_way"]),
            int(cfg["k_shot"]),
            bool(cfg["use_attention"]),
            int(cfg["height_proto_bins"]),
            float(cfg["height_proto_weight"]),
        )
    )


def build_command(cfg: dict, phase: str) -> list[str]:
    run_phase = "prototrain" if phase == "train" else "protoeval"
    merged = dict(DEFAULTS)
    merged.update(cfg)
    merged["phase"] = run_phase

    save_path = str(merged["save_path"])
    if not save_path.startswith("./logs/"):
        raise ValueError("save_path must stay under ./logs for project log hygiene")
    if not save_path.endswith("/"):
        save_path += "/"
    merged["save_path"] = save_path

    if phase == "eval" and not merged.get("model_checkpoint_path"):
        merged["model_checkpoint_path"] = save_path + proto_log_name(merged)

    cmd = [sys.executable, "main.py", "--phase", run_phase]
    for flag in sorted(BOOL_FLAGS):
        if merged.get(flag):
            cmd.append(f"--{flag}")
    if phase == "train":
        for flag, enabled in TRAIN_ONLY_FLAGS.items():
            if enabled:
                cmd.append(f"--{flag}")
    for name in ORDERED_ARGS:
        value = merged.get(name)
        if value in ("", None):
            continue
        cmd.extend([f"--{name}", str(value)])
    return cmd


def quote_cmd(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run/dry-run hproto lambda config.")
    parser.add_argument("config", type=Path, help="Path to configs/hproto/*.yaml")
    parser.add_argument("--phase", choices=["train", "eval"], default="train")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing it")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value when executing")
    args = parser.parse_args()

    cfg = read_flat_yaml(args.config)
    cmd = build_command(cfg, args.phase)
    print(quote_cmd(cmd))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
