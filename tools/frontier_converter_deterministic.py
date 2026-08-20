#!/usr/bin/env python3
"""Seed every converter RNG before entering the pinned EXL3 conversion CLI."""
from __future__ import annotations

import random

import torch

from exllamav3.conversion.convert_model import main, parser, prepare  # type: ignore[import-not-found]

SEED = 1376380199


def run() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    arguments = parser.parse_args()
    inputs, state, ok, error = prepare(arguments)
    if not ok:
        raise RuntimeError(error)
    main(inputs, state)


if __name__ == "__main__":
    run()
