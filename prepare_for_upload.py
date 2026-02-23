"""
prepare_for_upload.py

在部署机器上，对已下载的 fine-tuned ckpt 目录自动补全 openvla-7b base model 文件。

用法:
    python3 prepare_for_upload.py --ckpt_root /home/guian/checkpoints

脚本会：
  1. 扫描 ckpt_root 下的所有 checkpoint 目录（识别标志：含 lora_adapter/ 或 dataset_statistics.json）
  2. 从 HuggingFace 下载 openvla/openvla-7b 的共享文件到 ckpt_root/base_model_files/（只下载一次）
  3. 在每个 ckpt 目录内为缺失的 base model 文件创建相对路径软链接

下载三个 ckpt 的参考命令:
    huggingface-cli download Enderfga/vla franka_pick_place_baseline_step8000 \\
        --local-dir /home/guian/checkpoints/franka_pick_place_baseline_step8000
    huggingface-cli download Enderfga/vla franka_pick_place_mix_idm_step6000 \\
        --local-dir /home/guian/checkpoints/franka_pick_place_mix_idm_step6000
    huggingface-cli download Enderfga/vla franka_pick_place_auxloss_step9000 \\
        --local-dir /home/guian/checkpoints/franka_pick_place_auxloss_step9000
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# ── 配置 ─────────────────────────────────────────────────────────────────────
BASE_REPO = "moojink/openvla-7b-oft-finetuned-libero-spatial"

# 需要从 base model 补充的文件（fine-tuned ckpt 目录里没有的）
BASE_FILES = [
    "config.json",
    "configuration_prismatic.py",
    "modeling_prismatic.py",
    "generation_config.json",
    "model.safetensors.index.json",
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


def find_ckpt_dirs(ckpt_root: Path) -> list[Path]:
    """扫描 ckpt_root 的直接子目录，返回看起来像 checkpoint 的目录列表。"""
    found = []
    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir() or d.name == "base_model_files":
            continue
        if any((d / marker).exists() for marker in CKPT_MARKERS):
            found.append(d)
    return found


def download_base_files(base_dir: Path, force: bool = False) -> None:
    """下载 base model 的共享文件到 base_dir/。force=True 时删除已有文件强制重下。"""
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] Downloading base model files from {BASE_REPO}")
    print(f"      → {base_dir}  (force={force})\n")
    for fname in BASE_FILES:
        dest = base_dir / fname
        if dest.exists() or dest.is_symlink():
            if not force:
                print(f"  skip (exists): {fname}")
                continue
            dest.unlink()
            print(f"  removed (force): {fname}")
        print(f"  downloading : {fname} ...", end=" ", flush=True)
        hf_hub_download(
            repo_id=BASE_REPO,
            filename=fname,
            local_dir=str(base_dir),
            local_dir_use_symlinks=False,
        )
        print("done")
    print()


def symlink_into_ckpts(ckpt_dirs: list[Path], base_dir: Path, force: bool = False) -> None:
    """对每个 ckpt 目录，为缺失的 base model 文件创建相对路径软链接。force=True 时重建所有软链接。"""
    print(f"[2/3] Creating symlinks in {len(ckpt_dirs)} checkpoint director(ies)")
    for ckpt_dir in ckpt_dirs:
        print(f"\n  {ckpt_dir.name}/")
        for fname in BASE_FILES:
            src  = base_dir / fname
            link = ckpt_dir / fname
            if link.exists() or link.is_symlink():
                if not force:
                    print(f"    skip (exists): {fname}")
                    continue
                link.unlink()
                print(f"    removed (force): {fname}")

            if not src.exists():
                print(f"    WARN: base file not found, skip: {fname}")
                continue
            rel_src = os.path.relpath(src, ckpt_dir)
            os.symlink(rel_src, link)
            print(f"    linked: {fname} -> {rel_src}")
    print()


def verify(ckpt_dirs: list[Path]) -> None:
    """打印每个 ckpt 目录的完整性报告。"""
    print("[3/3] Verifying checkpoint directories")
    all_ok = True
    for ckpt_dir in ckpt_dirs:
        files = set()
        for f in ckpt_dir.rglob("*"):
            if f.is_file() or f.is_symlink():
                files.add(str(f.relative_to(ckpt_dir)))

        missing_base = [f for f in BASE_FILES if f not in files]
        has_adapter  = (ckpt_dir / "lora_adapter" / "adapter_model.safetensors").exists()
        action_heads = sorted(f for f in files if "action_head" in f)
        has_stats    = "dataset_statistics.json" in files

        ok = not missing_base and has_adapter and action_heads and has_stats
        if not ok:
            all_ok = False

        status = "✓" if ok else "✗"
        print(f"\n  [{status}] {ckpt_dir.name}")
        print(f"      base files missing : {missing_base if missing_base else 'none ✓'}")
        print(f"      lora_adapter       : {'✓' if has_adapter else '✗ MISSING'}")
        print(f"      action_head        : {action_heads if action_heads else '✗ MISSING'}")
        print(f"      dataset_statistics : {'✓' if has_stats else '✗ MISSING'}")

    print()
    if all_ok:
        print("All checkpoints look complete. Ready for inference.")
    else:
        print("Some checkpoints have issues — check the report above.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supplement fine-tuned ckpt dirs with openvla-7b base model files via symlinks."
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
        help="Delete and re-download base model files; rebuild all symlinks from scratch",
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

    print(f"Found {len(ckpt_dirs)} checkpoint director(ies) under {ckpt_root}:")
    for d in ckpt_dirs:
        print(f"  - {d.name}")
    print()

    download_base_files(base_dir, force=args.force)
    symlink_into_ckpts(ckpt_dirs, base_dir, force=args.force)
    verify(ckpt_dirs)


if __name__ == "__main__":
    main()
