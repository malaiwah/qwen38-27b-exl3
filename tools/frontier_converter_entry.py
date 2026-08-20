#!/usr/bin/env python3
"""Run the offline EXL3 converter without optional attention-serving kernels."""
from __future__ import annotations

import runpy
import sys
from types import ModuleType


def _serving_only(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError(
        "flash_attn is unavailable in the offline converter environment; "
        "attention execution is forbidden during conversion"
    )


def main() -> None:
    flash_attn = ModuleType("flash_attn")
    flash_attn.flash_attn_func = _serving_only  # type: ignore[attr-defined]
    flash_attn.flash_attn_with_kvcache = _serving_only  # type: ignore[attr-defined]
    flash_attn.flash_attn_varlen_func = _serving_only  # type: ignore[attr-defined]
    sys.modules["flash_attn"] = flash_attn
    runpy.run_path("/opt/frontier/convert.py", run_name="__main__")


if __name__ == "__main__":
    main()
