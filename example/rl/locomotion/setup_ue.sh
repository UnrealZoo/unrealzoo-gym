#!/usr/bin/env bash
# Use the pinned keyboard-source PPO runtime with the real UE RPC sampler.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(uname -s)" == Darwin ]]; then
  exec bash "$script_dir/setup_ue_macos.sh"
fi
bash "$script_dir/setup_mjlab.sh"

runtime_root="${UNREALZOO_RUNTIME_ROOT:-$HOME/.local/share/unrealzoo}"
venv_dir="${UNREALZOO_MJLAB_VENV:-$HOME/.venvs/unrealzoo-go1-mjlab}"
uv_bin="$runtime_root/tools/uv-x86_64-unknown-linux-gnu/uv"

# Keep the original PPO/CUDA versions; their lock already supplies the remaining
# client dependencies. pip check fails explicitly if that assumption changes.
"$uv_bin" pip install --no-config --python "$venv_dir/bin/python" --no-deps \
  unrealcv==1.3.2 opencv-python==4.10.0.84 docker==7.1.0 \
  gym==0.10.9 pyglet==1.5.21
"$uv_bin" pip check --python "$venv_dir/bin/python"
"$uv_bin" pip freeze --python "$venv_dir/bin/python" \
  > "$runtime_root/ue-go1-runtime-freeze.txt"
