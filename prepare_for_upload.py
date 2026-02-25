"""
prepare_for_upload.py

对 fine-tuned ckpt 目录自动补全 openvla-7b base model 文件 + OFT 架构代码。
运行一次即可修复所有权重问题（自动清理旧的错误文件）。

用法:
    python3 prepare_for_upload.py --ckpt_root /home/guian/checkpoints --force

脚本会：
  1. 清理旧的错误文件（4-shard LIBERO 权重、过期 config 备份等）
  2. 从 HuggingFace 下载 openvla/openvla-7b 的 base model 权重（3 shards）
  3. 复制 deploy/ 目录下自带的 OFT 架构文件（modeling_prismatic.py, configuration_prismatic.py）
  4. 在每个 ckpt 目录内创建软链接

IMPORTANT: 训练时的 base model 是 openvla/openvla-7b (3 shards)，
           NOT moojink/openvla-7b-oft-finetuned-libero-spatial (4 shards)！
           LoRA 权重必须叠加在正确的 base 上才能工作。
"""

import argparse
import glob
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

# ── 配置 ─────────────────────────────────────────────────────────────────────

# Base model: 必须与训练时一致！
BASE_REPO = "openvla/openvla-7b"

# 从 HuggingFace Hub 下载的 base model 文件（权重 + 配置）
# openvla/openvla-7b 有 3 个 safetensors 分片
BASE_HF_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]

# OFT 架构文件：从 deploy/ 目录自身复制（自包含，无需外部依赖）
OFT_CODE_FILES = [
    "modeling_prismatic.py",
    "configuration_prismatic.py",
]

# 所有需要在 ckpt 目录中存在的文件
ALL_BASE_FILES = BASE_HF_FILES + OFT_CODE_FILES

# 需要清理的旧错误文件（来自之前错误使用 LIBERO 4-shard 权重）
STALE_FILES = [
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
]

# ckpt 目录的识别标志（至少满足一个）
CKPT_MARKERS = [
    "lora_adapter",
    "dataset_statistics.json",
]
# ─────────────────────────────────────────────────────────────────────────────


DEPLOY_DIR = Path(__file__).parent.resolve()


def find_ckpt_dirs(ckpt_root: Path) -> list[Path]:
    """扫描 ckpt_root 的直接子目录，返回看起来像 checkpoint 的目录列表。"""
    found = []
    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir() or d.name == "base_model_files":
            continue
        if any((d / marker).exists() for marker in CKPT_MARKERS):
            found.append(d)
    return found


def download_and_prepare_base(base_dir: Path, force: bool = False) -> None:
    """下载 base model 权重 + 复制 OFT 架构文件到 base_dir/。"""
    base_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 下载 HF 权重 ──
    print(f"[1/4] Downloading base model weights from {BASE_REPO}")
    print(f"      → {base_dir}\n")
    for fname in BASE_HF_FILES:
        dest = base_dir / fname
        if dest.exists() or dest.is_symlink():
            if not force:
                print(f"  skip (exists): {fname}")
                continue
            dest.unlink()
        print(f"  downloading : {fname} ...", end=" ", flush=True)
        hf_hub_download(
            repo_id=BASE_REPO,
            filename=fname,
            local_dir=str(base_dir),
            local_dir_use_symlinks=False,
        )
        print("done")
    print()

    # ── Step 2: 从 deploy/ 目录复制 OFT 架构文件（自包含） ──
    print(f"[2/4] Copying OFT architecture files from {DEPLOY_DIR}\n")
    for fname in OFT_CODE_FILES:
        src = DEPLOY_DIR / fname
        dest = base_dir / fname
        if not src.exists():
            print(f"  ERROR: {src} not found! Make sure {fname} is in the deploy/ directory.")
            raise SystemExit(1)
        if dest.exists() or dest.is_symlink():
            if not force:
                print(f"  skip (exists): {fname}")
                continue
            dest.unlink()
        shutil.copy2(src, dest)
        print(f"  copied: {fname}")
    print()


def clean_and_link(ckpt_dirs: list[Path], base_dir: Path, force: bool = False) -> None:
    """清理旧文件 + 创建新软链接。"""
    print(f"[3/4] Cleaning stale files & creating symlinks in {len(ckpt_dirs)} checkpoint(s)")
    for ckpt_dir in ckpt_dirs:
        print(f"\n  {ckpt_dir.name}/")

        # ── 清理旧的 4-shard 文件 ──
        for old_shard in STALE_FILES:
            old_path = ckpt_dir / old_shard
            if old_path.is_symlink() or old_path.exists():
                old_path.unlink()
                print(f"    removed stale: {old_shard}")

        # ── 清理旧的 config.json 备份（update_auto_map 生成的） ──
        for backup in glob.glob(str(ckpt_dir / "config.json.back.*")):
            os.remove(backup)
            print(f"    removed stale backup: {Path(backup).name}")

        # ── 为所有 base 文件创建软链接 ──
        for fname in ALL_BASE_FILES:
            src = base_dir / fname
            link = ckpt_dir / fname
            # 强制模式或文件不存在时都重建
            if link.exists() or link.is_symlink():
                if not force:
                    print(f"    skip (exists): {fname}")
                    continue
                link.unlink()

            if not src.exists():
                print(f"    WARN: base file not found, skip: {fname}")
                continue
            rel_src = os.path.relpath(src, ckpt_dir)
            os.symlink(rel_src, link)
            print(f"    linked: {fname} -> {rel_src}")
    print()


def verify(ckpt_dirs: list[Path]) -> None:
    """打印每个 ckpt 目录的完整性报告。"""
    print("[4/4] Verifying checkpoint directories")
    all_ok = True
    for ckpt_dir in ckpt_dirs:
        files = set()
        for f in ckpt_dir.rglob("*"):
            if f.is_file() or f.is_symlink():
                files.add(str(f.relative_to(ckpt_dir)))

        missing_base = [f for f in ALL_BASE_FILES if f not in files]
        has_adapter  = (ckpt_dir / "lora_adapter" / "adapter_model.safetensors").exists()
        action_heads = sorted(f for f in files if "action_head" in f)
        has_stats    = "dataset_statistics.json" in files
        has_old_shards = any((ckpt_dir / s).exists() for s in STALE_FILES)

        ok = not missing_base and has_adapter and action_heads and has_stats and not has_old_shards
        if not ok:
            all_ok = False

        status = "✓" if ok else "✗"
        print(f"\n  [{status}] {ckpt_dir.name}")
        print(f"      base files missing : {missing_base if missing_base else 'none ✓'}")
        print(f"      lora_adapter       : {'✓' if has_adapter else '✗ MISSING'}")
        print(f"      action_head        : {action_heads if action_heads else '✗ MISSING'}")
        print(f"      dataset_statistics : {'✓' if has_stats else '✗ MISSING'}")
        if has_old_shards:
            print(f"      WARNING: stale 4-shard files still present!")

    print()
    if all_ok:
        print("All checkpoints look complete. Ready for inference.")
    else:
        print("Some checkpoints have issues — check the report above.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix checkpoint directories: download correct base model + link OFT architecture files."
    )
    parser.add_argument(
        "--ckpt_root",
        type=Path,
        default=Path.home() / "checkpoints",
        help="Directory containing the downloaded checkpoint subdirs (default: ~/checkpoints)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove all existing base files and rebuild from scratch (recommended for fixing broken checkpoints)",
    )
    args = parser.parse_args()

    ckpt_root: Path = args.ckpt_root.expanduser().resolve()
    if not ckpt_root.exists():
        print(f"ERROR: ckpt_root does not exist: {ckpt_root}")
        raise SystemExit(1)

    base_dir = ckpt_root / "base_model_files"

    ckpt_dirs = find_ckpt_dirs(ckpt_root)
    if not ckpt_dirs:
        print(f"No checkpoint directories found under {ckpt_root}")
        print("Make sure each ckpt folder contains 'lora_adapter/' or 'dataset_statistics.json'.")
        raise SystemExit(1)

    print(f"Found {len(ckpt_dirs)} checkpoint(s) under {ckpt_root}:")
    for d in ckpt_dirs:
        print(f"  - {d.name}")
    print()

    download_and_prepare_base(base_dir, force=args.force)
    clean_and_link(ckpt_dirs, base_dir, force=args.force)
    verify(ckpt_dirs)


if __name__ == "__main__":
    main()
