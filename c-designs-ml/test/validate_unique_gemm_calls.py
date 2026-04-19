#!/usr/bin/env python3

"""Validate unique call naming and manifest port roles for GEMM helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
TOOL = REPO_ROOT / "tools" / "scalehls-c-to-cpp.py"
ARTIFACT_ROOT = (
    REPO_ROOT
    / "conductor"
    / "tracks"
    / "gemm_only_accel_stack_revision_20260414"
    / "artifacts"
    / "c_unique_gemm_calls"
)


def run_design(source: Path, artifact_dir: Path) -> dict:
    cmd = [
        "python3",
        str(TOOL),
        str(source),
        "--pipeline-mode",
        "accel",
        "--artifact-dir",
        str(artifact_dir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    result_path = artifact_dir / "result.json"
    if proc.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"pipeline failed for {source.name}:\n{proc.stdout}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def main() -> int:
    duplicate = run_design(
        THIS_DIR / "gemm_duplicate_fp32.cpp", ARTIFACT_ROOT / "gemm_duplicate_fp32"
    )
    duplicate_symbols = duplicate.get("mapped_symbols", [])
    if duplicate_symbols != ["gemm_8x8x8_call0", "gemm_8x8x8_call1"]:
        raise RuntimeError(f"unexpected duplicate symbols: {duplicate_symbols}")
    duplicate_entries = duplicate.get("candidate_entries", [])
    if [entry.get("candidate_index") for entry in duplicate_entries] != [0, 1]:
        raise RuntimeError(f"unexpected candidate indices: {duplicate_entries}")
    duplicate_manifests = [Path(path) for path in duplicate.get("manifest_paths", [])]
    if len(duplicate_manifests) != 1 or duplicate_manifests[0].name != "manifest.json":
        raise RuntimeError(f"unexpected manifest files: {duplicate_manifests}")
    duplicate_manifest = json.loads(duplicate_manifests[0].read_text(encoding="utf-8"))
    helper_symbols = [helper.get("symbol") for helper in duplicate_manifest.get("helpers", [])]
    if helper_symbols != duplicate_symbols:
        raise RuntimeError(f"unexpected helper symbols in manifest: {helper_symbols}")
    helper_indices = [helper.get("call_index") for helper in duplicate_manifest.get("helpers", [])]
    if helper_indices != [0, 1]:
        raise RuntimeError(f"unexpected helper indices in manifest: {helper_indices}")
    emitted_cpp = Path(duplicate["emitted_cpp"]).read_text(encoding="utf-8")
    for symbol in duplicate_symbols:
        if emitted_cpp.count(f"void {symbol}(") != 1:
            raise RuntimeError(f"missing helper declaration for {symbol}")
        if emitted_cpp.count(f"{symbol}(") != 2:
            raise RuntimeError(f"expected one declaration and one call for {symbol}")

    existing_input = run_design(
        THIS_DIR / "gemmv_fp32_ptr.cpp",
        ARTIFACT_ROOT / "gemmv_fp32_ptr",
    )
    manifest_paths = existing_input.get("manifest_paths", [])
    if len(manifest_paths) != 1:
        raise RuntimeError(f"expected one manifest, got {manifest_paths}")
    manifest = json.loads(Path(manifest_paths[0]).read_text(encoding="utf-8"))
    helpers = manifest.get("helpers", [])
    if len(helpers) != 1:
        raise RuntimeError(f"expected one helper entry, got {helpers}")
    sizes = helpers[0].get("sizes", {})
    expected_roles = {"A", "B", "ExistingInput", "C"}
    if set(sizes.keys()) != expected_roles:
        raise RuntimeError(f"unexpected manifest size keys: {list(sizes.keys())}")
    if helpers[0].get("precision") != {"element": "f32", "bits": 32, "bytes": 4}:
        raise RuntimeError(f"unexpected manifest precision: {helpers[0].get('precision')}")

    print("Unique GEMM call naming validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
