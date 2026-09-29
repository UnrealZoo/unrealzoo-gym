"""Read the pinned keyboard-source PPO recipe without importing its simulator.

MJLab's package initializer configures Warp and discovers environment plugins.
UE supplies the simulator here, so only the original pure configuration files
and requested utility modules are executed in independent module namespaces.
"""
from __future__ import annotations

import ast
import hashlib
from importlib import metadata, util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from train_go1 import SOURCE_COMMIT, SOURCE_REPOSITORY

CONFIG_FILE = "src/mjlab/rl/config.py"
GO1_FILE = "src/mjlab/tasks/velocity/config/go1/rl_cfg.py"
UTILITY_FILES = ("src/mjlab/utils/os.py", "src/mjlab/utils/torch.py")
CONFIG_CLASSES = ("RslRlModelCfg", "RslRlOnPolicyRunnerCfg", "RslRlPpoAlgorithmCfg")


def inspect_ppo_source(source_dir: Path | None = None) -> dict:
    """Verify the real Git checkout without executing mjlab.__init__."""
    if source_dir is None:
        configured = os.environ.get("UNREALZOO_MJLAB_SOURCE")
        if configured:
            source_dir = Path(configured)
        else:
            default = Path.home() / ".local/share/unrealzoo/mjlab-e710cead"
            if default.is_dir():
                source_dir = default
            else:
                spec = util.find_spec("mjlab")
                if spec is not None and spec.origin:
                    source_dir = Path(spec.origin).resolve().parents[2]
    if source_dir is None:
        raise RuntimeError("Pass --mjlab-source or UNREALZOO_MJLAB_SOURCE pointing to the pinned MJLab Git checkout")
    root = Path(source_dir).expanduser().resolve(strict=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if commit.returncode or commit.stdout.strip() != SOURCE_COMMIT:
        raise RuntimeError(f"Expected MJLab {SOURCE_COMMIT} at {root}, found {commit.stdout.strip() or 'no Git commit'}")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"Pinned MJLab has tracked modifications: {dirty}")
    hashes = {}
    for relative in (CONFIG_FILE, GO1_FILE, *UTILITY_FILES):
        data = (root / relative).read_bytes()
        # Also reject an untracked/shadow file where the pinned checkout should
        # contain source, without relying solely on Git's dirty-file cache.
        original = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{relative}"],
            capture_output=True, check=True,
        ).stdout
        if data != original:
            raise RuntimeError(f"Source file differs from the pinned Git object: {relative}")
        hashes[relative] = hashlib.sha256(data).hexdigest()
    versions = {}
    for name in ("mjlab", "rsl-rl-lib", "torch", "torchvision", "tensordict", "numpy", "tensorboard", "onnx"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "repository": SOURCE_REPOSITORY, "commit": SOURCE_COMMIT,
        "source_root": str(root), "tracked_files_clean": True,
        "configuration_files_sha256": hashes, "versions": versions,
        "source_loading": "isolated original configuration modules; mjlab package and native simulator are not imported",
    }


def _source_bytes(source: dict, relative: str) -> tuple[Path, bytes]:
    path = Path(source["source_root"]) / relative
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != source["configuration_files_sha256"][relative]:
        raise RuntimeError(f"Source changed after verification: {relative}")
    return path, data


def load_source_utility(source: dict, relative: str) -> ModuleType:
    """Load a verified upstream utility by file, without MJLab package imports."""
    if relative not in (CONFIG_FILE, *UTILITY_FILES):
        raise ValueError("Only the pinned pure configuration and utility files may be loaded")
    path, data = _source_bytes(source, relative)
    name = "_unrealzoo_source_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module  # Required by the upstream dataclass decorators.
    try:
        exec(compile(data, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_ppo_runner_cfg(source: dict):
    """Execute the original Go1 factory with its original config dataclasses."""
    config = load_source_utility(source, CONFIG_FILE)
    path, data = _source_bytes(source, GO1_FILE)
    tree = ast.parse(data, filename=str(path))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    if (len(imports) != 1 or not isinstance(imports[0], ast.ImportFrom)
            or imports[0].module != "mjlab.rl" or imports[0].level != 0
            or {alias.name for alias in imports[0].names} != set(CONFIG_CLASSES)
            or any(alias.asname for alias in imports[0].names)):
        raise RuntimeError("Pinned Go1 PPO config import structure changed; inspect before loading")
    # Replace only the package import with the exact classes loaded above.
    # The source factory, including every default and override, is unchanged.
    tree.body.remove(imports[0])
    namespace = {name: getattr(config, name) for name in CONFIG_CLASSES}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["unitree_go1_ppo_runner_cfg"]()
