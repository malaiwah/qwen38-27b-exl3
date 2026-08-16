#!/usr/bin/env python3
"""Local CPU gate for the MP-adapter fail-closed patch.

Upstream CI runs tests/v1/test_vllm_mp_adapter.py with the full native
build; this box has no CUDA toolchain, so native-only modules that are
irrelevant to the unit under test (device kernels, native bitmap/enum)
are stubbed before import. The unit under test
(LMCacheMPWorkerAdapter.get_finished*) is pure Python and runs unmodified.
"""

import sys
import types

import lmcache


class _Stub:
    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> "_Stub":
        return _Stub(f"{self._name}.{attr}")

    def __call__(self, *args: object, **kwargs: object) -> "_Stub":
        return _Stub(f"{self._name}()")

    def __hash__(self) -> int:
        return hash(self._name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Stub) and other._name == self._name


class _AutoIntMembers(type):
    """Metaclass minting distinct int members on attribute access, standing
    in for the native EngineKVFormat enum at import time."""

    _counter = 0

    def __getattr__(cls, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        _AutoIntMembers._counter += 1
        member = int.__new__(cls, _AutoIntMembers._counter)
        member._member_name = name
        setattr(cls, name, member)
        return member


class _EngineKVFormat(int, metaclass=_AutoIntMembers):
    _member_name = "stub"

    @property
    def name(self) -> str:
        return self._member_name

    @property
    def value(self) -> int:
        return int(self)


class _Bitmap:
    def __init__(self, *args: object) -> None:
        raise NotImplementedError("native Bitmap stub must not be constructed")


for name in ("device_ops", "c_ops", "lmcache_native"):
    if not hasattr(lmcache, name):
        mod = types.ModuleType(f"lmcache.{name}")
        mod.EngineKVFormat = _EngineKVFormat
        mod.Bitmap = _Bitmap
        mod.__getattr__ = lambda attr, _n=name: _Stub(f"{_n}.{attr}")  # noqa: E731
        sys.modules[f"lmcache.{name}"] = mod
        setattr(lmcache, name, mod)

import pytest  # noqa: E402

sys.exit(
    pytest.main(
        [
            "tests/v1/test_vllm_mp_adapter.py",
            "--confcutdir=tests/v1",
            "-q",
            *sys.argv[1:],
        ]
    )
)
