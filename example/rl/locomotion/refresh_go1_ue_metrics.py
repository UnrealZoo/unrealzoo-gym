#!/usr/bin/env python3
"""Add ground/contact diagnostics to saved UE evaluation states, entirely offline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from evaluate_go1_ue import METRICS_SCHEMA_VERSION, ground_contact_metrics


def refresh_metrics(result):
    """Preserve existing rollout evidence and change only derived diagnostics."""
    if result.get("evaluation_profile") != "ue_go1_fixed_commands_v1":
        raise ValueError("Expected a fixed-command UE Go1 evaluation artifact")
    cases = result.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation contains no cases")
    updates = []
    for case in cases:
        samples = case.get("samples")
        if not isinstance(samples, list) or not isinstance(case.get("metrics"), dict):
            raise ValueError("Each case must contain saved samples and metrics")
        if any(not isinstance(sample.get("state"), dict) for sample in samples):
            raise ValueError("Full UE states are required; cannot reconstruct missing evidence")
        measured = [sample for sample in samples if sample.get("phase") == "command"]
        updates.append(ground_contact_metrics(measured))
    for case, diagnostics in zip(cases, updates):
        case["metrics"]["metrics_schema_version"] = METRICS_SCHEMA_VERSION
        case["metrics"]["ground_contact_after_warmup"] = diagnostics
    return len(cases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="New JSON artifact; keep the input unchanged")
    mode.add_argument("--in-place", action="store_true", help="Update input, retaining an exact .before-ground-contact.json backup")
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve(strict=True)
    target = source if args.in_place else args.output.expanduser().resolve()
    backup = source.with_name(source.stem + ".before-ground-contact.json") if args.in_place else None
    if (not args.in_place and target.exists()) or (backup is not None and backup.exists()):
        parser.error("Output or backup already exists; choose a new output path")
    original = source.read_bytes()
    result = json.loads(original)
    cases = refresh_metrics(result)
    result.setdefault("metrics_postprocessing", []).append({
        "operation": "ground_contact_diagnostics_from_saved_states",
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "source_sha256": hashlib.sha256(original).hexdigest(),
        "raw_samples_unchanged": True, "physics_steps_executed": 0,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
    })
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if backup is not None:
        with backup.open("xb") as stream:
            stream.write(original)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"METRICS|{target}|cases={cases}|physics_steps=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
