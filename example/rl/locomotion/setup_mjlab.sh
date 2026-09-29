#!/usr/bin/env bash
# Install the keyboard policy's original MJLab revision in an isolated environment.
# Linux x86_64 + NVIDIA only. Does not change conda, system Python, or uv.lock.
set -euo pipefail
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

readonly MJLAB_COMMIT=e710cead240b4c0f6f52afaa4f4b2a22c734082c
readonly UV_VERSION=0.10.12
readonly UV_SHA256=ec72570c9d1f33021aa80b176d7baba390de2cfeb1abcbefca346d563bf17484
readonly PYTHON_VERSION=3.11.15
readonly RUNTIME_ROOT="${UNREALZOO_RUNTIME_ROOT:-$HOME/.local/share/unrealzoo}"
readonly SOURCE_DIR="${UNREALZOO_MJLAB_SOURCE:-$RUNTIME_ROOT/mjlab-e710cead}"
readonly VENV_DIR="${UNREALZOO_MJLAB_VENV:-$HOME/.venvs/unrealzoo-go1-mjlab}"
readonly UV_BIN="$RUNTIME_ROOT/tools/uv-x86_64-unknown-linux-gnu/uv"
readonly TORCH_WHEEL="${UNREALZOO_TORCH_WHEEL:-$RUNTIME_ROOT/torch-wheel/torch-2.9.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl}"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This installer targets the Ubuntu x86_64 NVIDIA training host." >&2
  exit 1
fi
for tool in curl git tar sha256sum nvidia-smi; do
  command -v "$tool" >/dev/null || { echo "Required command missing: $tool" >&2; exit 1; }
done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
mkdir -p "$RUNTIME_ROOT/tools" "$(dirname "$VENV_DIR")"

if [[ ! -x "$UV_BIN" ]]; then
  uv_archive="$RUNTIME_ROOT/tools/uv-$UV_VERSION.tar.gz"
  curl --fail --location --retry 3 \
    "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
    --output "$uv_archive"
  printf '%s  %s\n' "$UV_SHA256" "$uv_archive" | sha256sum --check
  tar -xzf "$uv_archive" -C "$RUNTIME_ROOT/tools"
fi
[[ "$("$UV_BIN" --version)" == "uv $UV_VERSION "* ]] || {
  echo "Expected uv $UV_VERSION at $UV_BIN" >&2; exit 1;
}

if [[ ! -e "$SOURCE_DIR" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/mujocolab/mjlab.git "$SOURCE_DIR"
  git -C "$SOURCE_DIR" checkout --detach "$MJLAB_COMMIT"
fi
[[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$MJLAB_COMMIT" ]] || {
  echo "Existing MJLab checkout is not the expected revision: $SOURCE_DIR" >&2; exit 1;
}
[[ -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ]] || {
  echo "Existing MJLab checkout has tracked changes; refusing to overwrite them." >&2; exit 1;
}

"$UV_BIN" python install "$PYTHON_VERSION"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -c \
  'import sys; assert sys.version_info[:3] == (3, 11, 15), sys.version'

cd "$SOURCE_DIR"
"$UV_BIN" export --quiet --locked --no-dev --extra cu128 --no-emit-project \
  --no-hashes --python "$PYTHON_VERSION" \
  --output-file "$RUNTIME_ROOT/mjlab-requirements-original.txt"

# The original nightly wheel was removed upstream (HTTP 404 on 2026-09-16).
# Preserve uv.lock and every other locked version; use the stable release of
# the same MuJoCo 3.7 series. This is an explicit runtime compatibility deviation.
"$VENV_DIR/bin/python" - "$RUNTIME_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
original = (root / "mjlab-requirements-original.txt").read_text()
old = "mujoco==3.7.0.dev892927987"
assert original.count(old) == 1, "Unexpected upstream lock; inspect before changing dependencies"
runtime = original.replace(old, "mujoco==3.7.0")
# Keep PyPI's exact torchvision build instead of accepting a local-version
# variant from the extra CUDA index under PEP 440's normal == matching.
runtime = runtime.replace("torchvision==0.24.0", "torchvision===0.24.0")
(root / "mjlab-requirements-runtime.txt").write_text(runtime)
PY

# A predownloaded official wheel avoids slow CDN downloads on repeated setup.
# This checksum is copied from the unmodified upstream uv.lock cu128 entry.
if [[ -f "$TORCH_WHEEL" ]]; then
  printf '%s  %s\n' e97c264478c9fc48f91832749d960f1e349aeb214224ebe65fb09435dd64c59a "$TORCH_WHEEL" | sha256sum --check
  "$UV_BIN" pip install --no-config --python "$VENV_DIR/bin/python" --no-deps "$TORCH_WHEEL"
fi

# Ignore project uv overrides during pip installation: the original project
# loosens the MuJoCo constraint, which would otherwise replace the explicit pin.
"$UV_BIN" pip install --no-config --python "$VENV_DIR/bin/python" --no-deps \
  --index-url https://pypi.org/simple \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  --requirement "$RUNTIME_ROOT/mjlab-requirements-runtime.txt"
"$UV_BIN" pip install --no-config --python "$VENV_DIR/bin/python" --no-deps --editable "$SOURCE_DIR"
"$UV_BIN" pip check --python "$VENV_DIR/bin/python"
"$UV_BIN" pip freeze --python "$VENV_DIR/bin/python" > "$RUNTIME_ROOT/mjlab-runtime-freeze.txt"

"$VENV_DIR/bin/python" - "$RUNTIME_ROOT" "$SOURCE_DIR" <<'PY'
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import mujoco
import torch
import warp

root, source = map(Path, sys.argv[1:])
assert mujoco.__version__ == "3.7.0", mujoco.__version__
assert version("torch") == "2.9.0+cu128", version("torch")
assert version("torchvision") == "0.24.0", version("torchvision")
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
warp.init()
assert warp.is_cuda_available(), "Warp cannot access CUDA"
result = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source": str(source),
    "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
    "original_uv_lock_sha256": hashlib.sha256((source / "uv.lock").read_bytes()).hexdigest(),
    "python": sys.version,
    "python_executable": sys.executable,
    "packages": {name: version(name) for name in ("mjlab", "torch", "torchvision", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib")},
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
    "runtime_deviation": {
        "mujoco": {
            "locked": "3.7.0.dev892927987",
            "installed": mujoco.__version__,
            "reason": "Original nightly cp311 Linux wheel returns HTTP 404; same-series stable 3.7.0 used. Source and uv.lock unchanged.",
        }
    },
}
(root / "mjlab-runtime-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

printf '\nTraining Python: %s\nMJLab source: %s\n' "$VENV_DIR/bin/python" "$SOURCE_DIR"
