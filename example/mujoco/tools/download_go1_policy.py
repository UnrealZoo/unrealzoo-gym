#!/usr/bin/env python3
"""Download an open-source Go1 ONNX locomotion policy for the MuJoCo demo."""
import argparse
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MUJOCO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = MUJOCO_DIR / "policies"
HF_REPO = "robomotic/mjlab-policies"
POLICIES = {
    "go1_velocity": "go1_velocity/2026-04-27_13-40-37/policy.onnx",
    "go1_rough_velocity": "go1_rough_velocity/2026-04-29_08-00-37/policy.onnx",
}
OUTPUT_PATHS = {
    "go1_velocity": Path("go1") / "velocity" / "policy.onnx",
    "go1_rough_velocity": Path("go1") / "rough_velocity" / "policy.onnx",
}


def build_hf_url(filename):
    return f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}"


def download(url, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"DOWNLOAD|{url}")
    print(f"OUTPUT|{output_path}")
    with urllib.request.urlopen(url, timeout=120) as response:
        with tmp_path.open("wb") as dst:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    tmp_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser(description="Download Go1 MJLab ONNX policy")
    parser.add_argument("--name", choices=sorted(POLICIES), default="go1_velocity")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    filename = POLICIES[args.name]
    output_path = args.output_dir / OUTPUT_PATHS[args.name]
    if output_path.exists() and not args.force:
        print(f"EXISTS|{output_path}")
        return 0

    try:
        download(build_hf_url(filename), output_path)
    except Exception as exc:
        print(f"ERROR|failed_to_download|{exc}", file=sys.stderr)
        return 1

    print(f"POLICY_PATH|{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
