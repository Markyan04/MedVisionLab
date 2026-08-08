#!/usr/bin/env python
"""Run the fixed KL-match ordinal soft-CE experiments on ResNet50."""

from __future__ import annotations

import sys

from run_resnet50_dasl_experiments import main


if __name__ == "__main__":
    raise SystemExit(main(["--loss", "kl_match_ce", *sys.argv[1:]]))
