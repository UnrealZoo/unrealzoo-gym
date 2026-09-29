#!/usr/bin/env bash
# Isolated CPU PPO/client runtime. UE supplies physics; MJLab is source data only.
set -euo pipefail

readonly MJLAB_COMMIT=e710cead240b4c0f6f52afaa4f4b2a22c734082c
readonly RUNTIME_ROOT="${UNREALZOO_RUNTIME_ROOT:-$HOME/.local/share/unrealzoo}"
readonly SOURCE_DIR="${UNREALZOO_MJLAB_SOURCE:-$RUNTIME_ROOT/mjlab-e710cead}"
readonly VENV_DIR="${UNREALZOO_UE_VENV:-$HOME/.venvs/unrealzoo-go1-ue-macos}"
readonly PYTHON_BIN="${UNREALZOO_PYTHON:-python3}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "This installer requires macOS on Apple Silicon (arm64)." >&2
  exit 1
fi
command -v git >/dev/null || { echo "Required command missing: git" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || {
  echo "Set UNREALZOO_PYTHON to an existing arm64 Python 3.10–3.12 executable." >&2
  exit 1
}
"$PYTHON_BIN" -I - <<'PY'
import platform
import sys

if sys.platform != "darwin" or platform.machine() != "arm64":
    raise SystemExit("Use a native arm64 macOS Python, not a Rosetta interpreter")
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit("This runtime supports Python 3.10–3.12; set UNREALZOO_PYTHON")
PY

# Never convert a conda/system environment or an unrelated directory into this
# venv. Reuse is allowed only for an isolated, compatible lightweight runtime.
if [[ -e "$VENV_DIR" && (! -f "$VENV_DIR/pyvenv.cfg" || ! -x "$VENV_DIR/bin/python") ]]; then
  echo "Existing UNREALZOO_UE_VENV is not a complete Python venv: $VENV_DIR" >&2
  exit 1
fi
if [[ -e "$SOURCE_DIR" ]]; then
  [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$MJLAB_COMMIT" ]] || {
    echo "Existing MJLab checkout is not the expected revision: $SOURCE_DIR" >&2; exit 1;
  }
  [[ -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ]] || {
    echo "Existing MJLab checkout has tracked changes; refusing to overwrite them." >&2; exit 1;
  }
else
  mkdir -p "$(dirname "$SOURCE_DIR")"
  git clone --filter=blob:none --no-checkout https://github.com/mujocolab/mjlab.git "$SOURCE_DIR"
  git -C "$SOURCE_DIR" checkout --detach "$MJLAB_COMMIT"
fi

mkdir -p "$RUNTIME_ROOT" "$(dirname "$VENV_DIR")"
if [[ ! -e "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -I -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -I - "$VENV_DIR" <<'PY'
from importlib import metadata
from pathlib import Path
import platform
import sys

target = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != target or sys.prefix == sys.base_prefix:
    raise SystemExit("Refusing to install outside the selected isolated venv")
if not (3, 10) <= sys.version_info[:2] <= (3, 12) or platform.machine() != "arm64":
    raise SystemExit("Existing venv needs arm64 Python 3.10–3.12")
config = (target / "pyvenv.cfg").read_text().lower().replace(" ", "")
if "include-system-site-packages=true" in config:
    raise SystemExit("The venv must not inherit system site packages")
for package in ("mjlab", "mujoco", "mujoco-warp", "warp-lang"):
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        continue
    raise SystemExit(f"Use a separate lightweight UE venv; {package} is already installed")
PY

# Top-level versions used for the Mac UE training experiment. Transitive
# dependencies are resolved by pip and recorded in the complete freeze below;
# this is deliberately not the CUDA/native-simulator lock from MJLab.
cat > "$RUNTIME_ROOT/ue-macos-requirements.txt" <<'REQ'
torch==2.9.0
torchvision==0.24.0
tensordict==0.10.0
rsl-rl-lib==5.0.1
tensorboard==2.20.0
numpy==2.2.6
PyYAML==6.0.2
unrealcv==1.3.2
onnxruntime==1.23.2
psutil==7.0.0
REQ
"$VENV_DIR/bin/python" -I -m pip --isolated --require-virtualenv install \
  --index-url https://pypi.org/simple \
  --requirement "$RUNTIME_ROOT/ue-macos-requirements.txt"
"$VENV_DIR/bin/python" -I -m pip --isolated --require-virtualenv check
"$VENV_DIR/bin/python" -I -m pip --isolated --require-virtualenv freeze --all \
  > "$RUNTIME_ROOT/ue-macos-runtime-freeze.txt"

"$VENV_DIR/bin/python" -I - "$RUNTIME_ROOT" "$SOURCE_DIR" "$script_dir" <<'PY'
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import hashlib
import json
import platform
import sys

root, source, scripts = (Path(arg).resolve() for arg in sys.argv[1:])
sys.path.insert(0, str(scripts))
from ue_ppo_config import inspect_ppo_source, load_ppo_runner_cfg

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict
from unrealcv import Client
import onnxruntime
import psutil
import yaml

requirements = (root / "ue-macos-requirements.txt").read_text().splitlines()
packages = {}
for requirement in requirements:
    name, expected = requirement.split("==")
    actual = metadata.version(name)
    if actual != expected:
        raise RuntimeError(f"Expected {name} {expected}, found {actual}")
    packages[name] = actual
source_info = inspect_ppo_source(source)
recipe = asdict(load_ppo_runner_cfg(source_info))
for module in ("mjlab", "warp", "mujoco", "mujoco_warp"):
    if module in sys.modules:
        raise RuntimeError(f"Unexpected native simulator import: {module}")
assert torch.ones(1, device="cpu").item() == 1.0
result = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python": sys.version,
    "python_executable": sys.executable,
    "packages": packages,
    "source": source_info,
    "original_uv_lock_sha256": hashlib.sha256((source / "uv.lock").read_bytes()).hexdigest(),
    "original_ppo_recipe": recipe,
    "torch_cuda": torch.version.cuda,
    "validated_training_device": "cpu",
    "full_freeze": str(root / "ue-macos-runtime-freeze.txt"),
    "runtime_scope": "UE physics with original PPO config; native MJLab/Warp is not installed",
    "runtime_deviations": [
        "Mac CPU wheels and Python 3.10–3.12 instead of the Linux CUDA runtime",
        "NumPy 2.2.6 supports Python 3.10; transitive versions are recorded, not upstream-lock-equivalent",
        "UE ue_v310_velocity is a reduced task and is not the full native MJLab environment",
    ],
}
(root / "ue-macos-runtime-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

printf '\nTraining Python: %s\nMJLab source (configuration only): %s\n' "$VENV_DIR/bin/python" "$SOURCE_DIR"
