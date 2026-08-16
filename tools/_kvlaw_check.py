#!/usr/bin/env python3
"""Scratch: recompute every 24 GB-class KV number from the measured affine law.

Inputs are read from receipts/qualification-24gib-capped.json, never hardcoded
from prose. Not part of the shipped tool set; used to verify docs/34 numbers.
"""
import json

GIB = 2 ** 30
Q = json.load(open("receipts/qualification-24gib-capped.json"))
law = Q["kv_law_measured"]["law"]
a3 = float(law["mtp3"]["a_bytes_per_token"])
M3 = float(law["mtp3"]["M_gib"]) * GIB
a0 = float(law["mtpoff"]["a_bytes_per_token"])
M0 = float(law["mtpoff"]["M_gib"]) * GIB

T = Q["memory_component_comparison"]["table"]
b3 = T["board_arm_22_49_gib_mtp3"]
b0 = T["board_arm_22_49_gib_mtpoff"]
ov3 = b3["peak_activation"] + b3["non_torch"] + b3["cudagraph"]
ov0 = b0["peak_activation"] + b0["non_torch"] + b0["cudagraph"]
budget = 24 * 0.98125 * 0.955


def need(L, a, M):
    return (a * L + M) / GIB


def lmax(pool, a, M):
    return int((pool * GIB - M) / a)


def rule15(pool, a, M):
    L = int((pool / 1.15 * GIB - M) / a)
    return L, (L // 4096) * 4096


def hmem(pool, L, a, M):
    return (pool / need(L, a, M) - 1) * 100


def htok(pool, L, a, M):
    return (lmax(pool, a, M) / L - 1) * 100


print("law from receipt: a3=%.1f M3=%.2f GiB | a0=%.1f M0=%.2f GiB"
      % (a3, M3 / GIB, a0, M0 / GIB))
print("a3/floor = %.6f (17/16 = %.6f)" % (a3 / 32768, 17 / 16))
print("per-token premium of MTP-3 over MTP-off: %.2f %%" % ((a3 / a0 - 1) * 100))
print("premium as fraction of the fp8 floor:    %.2f %%" % ((a3 - a0) / 32768 * 100))
print("overheads measured: MTP-3 %.2f GiB, MTP-off %.2f GiB" % (ov3, ov0))
print("24 GiB board: free %.4f -> budget %.4f GiB" % (24 * 0.98125, budget))
print()

print("== reproduce the receipt's own board-arm numbers ==")
for L, pool, a, M, lbl in ((32768, b3["kv_pool"], a3, M3, "MTP-3"),
                           (45056, b0["kv_pool"], a0, M0, "MTP-off")):
    print("%-8s L=%-6d pool=%.2f need=%.4f GiB headroom=%.1f %% engine_max=%d"
          % (lbl, L, pool, need(L, a, M), hmem(pool, L, a, M), lmax(pool, a, M)))
print()

print("== pools from the card model, both resident measurements ==")
cases = [
    ("MTP-3  conservative 18.41", budget - ov3 - 18.41, a3, M3),
    ("MTP-3  card-measured 18.19", budget - ov3 - 18.19, a3, M3),
    ("MTP-off conservative 18.158", budget - ov0 - (18.41 - 0.252), a0, M0),
    ("MTP-off card-measured 17.94", budget - ov0 - 17.94, a0, M0),
]
for lbl, pool, a, M in cases:
    L15, pub = rule15(pool, a, M)
    print("%-28s pool=%.4f Lmax=%6d  15%%-rule<=%6d -> publish %6d"
          % (lbl, pool, lmax(pool, a, M), L15, pub))
print()

print("== candidate windows ==")
for L in (20480, 24576, 28672, 32768):
    print("MTP-3 L=%-6d need=%.4f GiB" % (L, need(L, a3, M3)))
    for lbl, pool, a, M in cases[:2]:
        print("    %-28s fits=%-5s mem-headroom=%6.2f %%  token-headroom=%6.2f %%"
              % (lbl, need(L, a, M) <= pool, hmem(pool, L, a, M), htok(pool, L, a, M)))
for L in (45056, 65536):
    print("MTP-off L=%-6d need=%.4f GiB" % (L, need(L, a0, M0)))
    for lbl, pool, a, M in cases[2:]:
        print("    %-28s fits=%-5s mem-headroom=%6.2f %%  token-headroom=%6.2f %%"
              % (lbl, need(L, a, M) <= pool, hmem(pool, L, a, M), htok(pool, L, a, M)))
print()

print("== MTP-3 line items ==")
dw = b3["weight"] - b0["weight"]
gp = b3["cudagraph"] - b0["cudagraph"]
fk = (M3 - M0) / GIB
ac = b3["peak_activation"] - b0["peak_activation"]
print("draft weights %.2f + graph pool %.2f + fixed per-request KV %.2f = %.2f GiB"
      % (dw, gp, fk, dw + gp + fk))
print("(+ draft activation %.2f -> %.2f GiB of total device cost)" % (ac, dw + gp + fk + ac))
print("fixed KV term as a share of the board-arm MTP-3 pool: %.1f %%"
      % (M3 / GIB / b3["kv_pool"] * 100))
print()

print("== 5.3 profile table, re-derived under the measured law at 0.955 ==")
for name, res3, draft in (("M24-Q (hydrated body)", 19.27, 0.264),
                          ("M24 (context body)", 18.41, 0.252),
                          ("M24-L", 17.55, 0.198)):
    kv3 = budget - ov3 - res3
    res0 = res3 - draft
    kv0 = budget - ov0 - res0
    p3 = rule15(kv3, a3, M3)[1]
    p0 = rule15(kv0, a0, M0)[1]
    print("%-22s res3=%5.2f KV3=%4.2f Lmax3=%6d pub3=%6d | res0=%6.3f KV0=%4.2f Lmax0=%6d pub0=%6d"
          % (name, res3, kv3, lmax(kv3, a3, M3), p3, res0, kv0, lmax(kv0, a0, M0), p0))

print()
print("== specific windows quoted in prose ==")
for L in (16384, 20480, 24576, 28672, 32768, 40960, 45056, 49152, 53248, 61440):
    print("  MTP-3  need(%6d) = %.4f GiB   |  MTP-off need = %.4f GiB"
          % (L, need(L, a3, M3), need(L, a0, M0)))
print()
print("== 4-bit KV column (a halved, M held: never measured) ==")
for name, res3 in (("M24-Q", 19.27), ("M24", 18.41), ("M24-L", 17.55)):
    kv3 = budget - ov3 - res3
    print("  %-6s KV3=%.2f  Lmax(4bit)=%d" % (name, kv3, lmax(kv3, a3 / 2, M3)))
print()
print("== 16 GB sanity: does MTP-3 fit any window? ==")
print("  fixed MTP-3 KV term alone = %.2f GiB; s6 pool is 0.55 GiB -> fits=%s"
      % (M3 / GIB, M3 / GIB <= 0.55))
print()
print("== board-arm pool vs card model, the consistency check ==")
print("  22.49 - %.2f - 17.94 = %.4f  (board arm measured MTP-off pool %.2f)"
      % (ov0, budget - ov0 - 17.94, b0["kv_pool"]))
print("  22.49 - %.2f - 18.19 = %.4f  (board arm measured MTP-3   pool %.2f)"
      % (ov3, budget - ov3 - 18.19, b3["kv_pool"]))
