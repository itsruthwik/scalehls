#!/usr/bin/env python3

"""Validate GEMM IP tiling across representative C-side configurations."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
TOOL = REPO_ROOT / "tools" / "scalehls-c-to-cpp.py"
TRACK_ROOT = (
    REPO_ROOT / "conductor" / "tracks" / "accel_pointer_tiled_abi_20260412"
)
ARTIFACT_ROOT = TRACK_ROOT / "artifacts" / "c_gemm_ip_tiling"
SUMMARY_JSON = ARTIFACT_ROOT / "summary.json"
SUMMARY_MD = ARTIFACT_ROOT / "summary.md"


@dataclass(frozen=True)
class Case:
    name: str
    source: Path
    ip_size: str
    serial: bool
    expected_symbols: int
    expected_family: str = "GEMM"


CASES: tuple[Case, ...] = (
    Case(
        name="gemm_fp32_ip4_parallel",
        source=THIS_DIR.parent / "layers" / "gemm_fp32.cpp",
        ip_size="4x4x4",
        serial=False,
        expected_symbols=4,
    ),
    Case(
        name="gemm_fp32_ip4_serial",
        source=THIS_DIR.parent / "layers" / "gemm_fp32.cpp",
        ip_size="4x4x4",
        serial=True,
        expected_symbols=1,
    ),
    Case(
        name="gemm_tiling_fp32_ip8_parallel",
        source=THIS_DIR / "gemm_tiling_fp32.cpp",
        ip_size="8x8x8",
        serial=False,
        expected_symbols=8,
    ),
    Case(
        name="gemm_tiling_fp32_ip8_serial",
        source=THIS_DIR / "gemm_tiling_fp32.cpp",
        ip_size="8x8x8",
        serial=True,
        expected_symbols=1,
    ),
    Case(
        name="gemm_tiling_fp32_ip16_parallel",
        source=THIS_DIR / "gemm_tiling_fp32.cpp",
        ip_size="16x16x16",
        serial=False,
        expected_symbols=1,
    ),
)


def run_case(case: Case) -> dict:
    artifact_dir = ARTIFACT_ROOT / case.name
    cmd = [
        "python3",
        str(TOOL),
        str(case.source),
        "--pipeline-mode",
        "accel",
        "--ip-size",
        case.ip_size,
        "--artifact-dir",
        str(artifact_dir),
    ]
    if case.serial:
        cmd.append("--serial")
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    result_path = artifact_dir / "result.json"
    case_result = {
        "name": case.name,
        "source": str(case.source.relative_to(REPO_ROOT)),
        "ip_size": case.ip_size,
        "serial": case.serial,
        "ok": False,
        "stdout": proc.stdout,
    }
    if proc.returncode != 0 or not result_path.exists():
        case_result["failure"] = "pipeline failed"
        return case_result

    result = json.loads(result_path.read_text(encoding="utf-8"))
    case_result["result"] = result

    if not result.get("frontend_succeeded"):
        case_result["failure"] = "frontend failed"
        return case_result
    if not result.get("mapper_succeeded"):
        case_result["failure"] = "mapper failed"
        return case_result
    if not result.get("mapped"):
        case_result["failure"] = "not mapped"
        return case_result
    if not result.get("emit_succeeded"):
        case_result["failure"] = "emit failed"
        return case_result

    candidate_entries = result.get("candidate_entries", [])
    if len(candidate_entries) != case.expected_symbols:
        case_result["failure"] = (
            f"expected {case.expected_symbols} mapped candidate(s), got {len(candidate_entries)}"
        )
        return case_result

    if not candidate_entries or candidate_entries[0].get("family") != case.expected_family:
        case_result["failure"] = f"expected family {case.expected_family}"
        return case_result

    case_result["ok"] = True
    return case_result


def write_summary(results: list[dict]) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(
        json.dumps({"cases": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# GEMM IP Tiling Validation Summary",
        "",
        "| Case | IP Size | Mode | Status | Mapped Symbols |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in results:
        result = item.get("result", {})
        mapped_symbols = len(result.get("candidate_entries", []))
        status = "ok" if item.get("ok") else item.get("failure", "failed")
        mode = "serial" if item.get("serial") else "parallel"
        lines.append(
            f"| {item['name']} | {item['ip_size']} | {mode} | {status} | {mapped_symbols} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = [run_case(case) for case in CASES]
    write_summary(results)

    failures = [item for item in results if not item.get("ok")]
    if failures:
        print("GEMM IP tiling validation failures:")
        for item in failures:
            print(f" - {item['name']}: {item.get('failure', 'failed')}")
        return 1

    print("GEMM IP tiling validation passed.")
    print(f"summary: {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
