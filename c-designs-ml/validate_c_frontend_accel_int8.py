#!/usr/bin/env python3
"""Validate the canonical INT8 C accelerator-family samples through the C frontend."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from validate_c_frontend_gemm_mapper import main as shared_main


def main(argv: list[str] | None = None) -> int:
  default_output = (
      THIS_DIR.parent
      / "conductor"
      / "tracks"
      / "shared_accelerator_family_ir_20260411"
      / "artifacts"
      / "c_int8_validation"
  )

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo-root", type=Path, default=THIS_DIR.parent)
  parser.add_argument("--output-root", type=Path, default=default_output)
  parser.add_argument(
      "--designs",
      nargs="+",
      default=[
          "gemmv_int8",
          "gemm_int8",
          "conv2d_int8",
          "tiny_lenet_i8",
      ],
  )
  parser.add_argument("--print-commands", action="store_true")
  args = parser.parse_args(argv)

  forwarded = [
      "--repo-root",
      str(args.repo_root),
      "--output-root",
      str(args.output_root),
      "--designs",
      *args.designs,
  ]
  if args.print_commands:
    forwarded.append("--print-commands")
  return shared_main(forwarded)


if __name__ == "__main__":
  raise SystemExit(main())
