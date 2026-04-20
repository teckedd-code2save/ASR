#!/usr/bin/env python3
"""
Pre-flight Environment Validation Script
Run this BEFORE starting training to catch configuration issues early.
Usage: python check_env.py
"""

import os
import sys
import platform
import torch
from datetime import datetime


def check(section, checks):
    """Run a list of checks and report results."""
    print(f"\n{'='*60}")
    print(f"  {section}")
    print(f"{'='*60}")
    errors = []
    warnings_list = []

    for name, func, level in checks:
        try:
            result = func()
            status = "PASS" if result[0] else ("WARN" if level == "warn" else "FAIL")
            icon = "✅" if status == "PASS" else ("⚠️" if status == "WARN" else "❌")
            print(f"  {icon} {name}: {result[1]}")
            if not result[0]:
                if level == "warn":
                    warnings_list.append((name, result[1]))
                else:
                    errors.append((name, result[1]))
        except Exception as e:
            print(f"  ❌ {name}: Exception - {e}")
            errors.append((name, str(e)))

    return errors, warnings_list


def main():
    print("\n" + "=" * 60)
    print("  WHISPER ASR TRAINING - ENVIRONMENT CHECK")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_errors = []
    all_warnings = []

    # ------------------------------------------------------------------
    # 1. Python Version
    # ------------------------------------------------------------------
    def py_version():
        v = sys.version_info
        ok = v.major == 3 and v.minor >= 10
        return ok, f"Python {v.major}.{v.minor}.{v.micro}"

    errs, warns = check("Python Environment", [
        ("Python 3.10+", py_version, "error"),
    ])
    all_errors.extend(errs)
    all_warnings.extend(warns)

    # ------------------------------------------------------------------
    # 2. GPU & CUDA
    # ------------------------------------------------------------------
    def cuda_available():
        return torch.cuda.is_available(), f"CUDA available: {torch.cuda.is_available()}"

    def cuda_version():
        return True, f"CUDA {torch.version.cuda}"

    def cudnn_version():
        return True, f"cuDNN {torch.backends.cudnn.version()}"

    def gpu_name():
        if not torch.cuda.is_available():
            return False, "No GPU detected"
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return True, f"{name} ({vram:.1f} GB VRAM)"

    def vram_check():
        if not torch.cuda.is_available():
            return False, "No GPU"
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok = vram >= 16  # Minimum for whisper-medium
        return ok, f"{vram:.1f} GB (need >=16GB for medium, >=8GB for small)"

    def bf16_support():
        if not torch.cuda.is_available():
            return False, "No GPU"
        return torch.cuda.is_bf16_supported(), f"bf16: {torch.cuda.is_bf16_supported()}"

    errs, warns = check("GPU & CUDA", [
        ("CUDA Available", cuda_available, "error"),
        ("CUDA Version", cuda_version, "info"),
        ("cuDNN Version", cudnn_version, "info"),
        ("GPU Name", gpu_name, "info"),
        ("VRAM Check", vram_check, "error"),
        ("bf16 Support", bf16_support, "warn"),
    ])
    all_errors.extend(errs)
    all_warnings.extend(warns)

    # ------------------------------------------------------------------
    # 3. Python Packages
    # ------------------------------------------------------------------
    packages = [
        ("torch", "2.5", True),
        ("transformers", "4.47", True),
        ("datasets", "3.2", True),
        ("accelerate", "1.2", False),
        ("evaluate", "0.4", False),
        ("jiwer", "3.1", False),
        ("peft", "0.14", False),
        ("wandb", "0.19", False),
        ("librosa", "0.10", False),
        ("soundfile", "0.13", False),
        ("requests", "2.32", False),
        ("pandas", "2.2", False),
    ]

    def make_pkg_check(pkg_name, min_ver, required):
        def _check():
            try:
                mod = __import__(pkg_name)
                ver = getattr(mod, "__version__", "unknown")
                return True, f"{pkg_name} {ver}"
            except ImportError:
                return not required, f"{pkg_name} NOT FOUND {'(required)' if required else '(optional)'}"
        return _check

    pkg_checks = [(pkg, make_pkg_check(pkg, min_v, req), "error" if req else "warn")
                  for pkg, min_v, req in packages]

    errs, warns = check("Python Packages", pkg_checks)
    all_errors.extend(errs)
    all_warnings.extend(warns)

    # ------------------------------------------------------------------
    # 4. System Dependencies
    # ------------------------------------------------------------------
    def ffmpeg():
        import shutil
        path = shutil.which("ffmpeg")
        return path is not None, f"{'Found at ' + path if path else 'NOT FOUND'}"

    def libsndfile():
        try:
            import soundfile
            return True, "Available via soundfile"
        except Exception as e:
            return False, str(e)

    errs, warns = check("System Dependencies", [
        ("ffmpeg", ffmpeg, "error"),
        ("libsndfile", libsndfile, "error"),
    ])
    all_errors.extend(errs)
    all_warnings.extend(warns)

    # ------------------------------------------------------------------
    # 5. Environment Variables
    # ------------------------------------------------------------------
    def hf_token():
        tok = os.environ.get("HF_TOKEN")
        return tok is not None and len(tok) > 10, "SET" if tok else "NOT SET"

    def wandb_key():
        key = os.environ.get("WANDB_API_KEY")
        return key is not None, "SET" if key else "NOT SET (optional)"

    def mozilla_key():
        key = os.environ.get("MOZILLA_APIKEY")
        return key is not None, "SET" if key else "NOT SET (Common Voice will be skipped)"

    def output_dir():
        d = os.environ.get("OUTPUT_DIR", "/workspace/outputs")
        exists = os.path.exists(d)
        return True, f"{d} ({'exists' if exists else 'will be created'})"

    errs, warns = check("Environment Variables", [
        ("HF_TOKEN", hf_token, "error"),
        ("WANDB_API_KEY", wandb_key, "warn"),
        ("MOZILLA_APIKEY", mozilla_key, "warn"),
        ("OUTPUT_DIR", output_dir, "info"),
    ])
    all_errors.extend(errs)
    all_warnings.extend(warns)

    # ------------------------------------------------------------------
    # 6. Disk Space
    # ------------------------------------------------------------------
    def disk_space():
        import shutil
        d = os.environ.get("OUTPUT_DIR", "/workspace")
        os.makedirs(d, exist_ok=True)
        stat = shutil.disk_usage(d)
        free_gb = stat.free / 1e9
        ok = free_gb >= 50  # Need ~50GB for model + data + checkpoints
        return ok, f"{free_gb:.1f} GB free at {d} (need >=50GB)"

    errs, warns = check("Disk Space", [
        ("Free Space", disk_space, "error"),
    ])
    all_errors.extend(errs)
    all_warnings.extend(warns)

    # ------------------------------------------------------------------
    # 7. HuggingFace Hub Connectivity
    # ------------------------------------------------------------------
    def hf_connectivity():
        try:
            import urllib.request
            urllib.request.urlopen("https://huggingface.co", timeout=10)
            return True, "huggingface.co reachable"
        except Exception as e:
            return False, str(e)

    errs, warns = check("Network Connectivity", [
        ("HuggingFace Hub", hf_connectivity, "error"),
    ])
    all_errors.extend(errs)
    all_warnings.extend(warns)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    if all_errors:
        print(f"\n  ❌ ERRORS: {len(all_errors)}")
        for name, msg in all_errors:
            print(f"     - {name}: {msg}")

    if all_warnings:
        print(f"\n  ⚠️  WARNINGS: {len(all_warnings)}")
        for name, msg in all_warnings:
            print(f"     - {name}: {msg}")

    if not all_errors:
        print("\n  ✅ ALL CHECKS PASSED - Ready for training!")
        return 0
    else:
        print(f"\n  ❌ {len(all_errors)} error(s) must be fixed before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
