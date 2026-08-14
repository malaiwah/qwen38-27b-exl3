from __future__ import annotations
import math
import os
import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING
import re
if TYPE_CHECKING:
    from ..model import Model, Config

@dataclass
class QTarget:
    numel: int
    target_bpw: int
    min_bpw: int
    priority: int

    def total_bits(self):
        return self.numel * self.target_bpw

    def delta_1(self):
        delta = (min(8, self.target_bpw + 1) - self.target_bpw) * self.numel
        return delta

    def increase_1(self):
        self.target_bpw = min(8, self.target_bpw + 1)

    def clamp_min(self):
        self.target_bpw = max(self.target_bpw, self.min_bpw)


def create_q_strategy(
    model: Model,
    mtp_model: Model,
    config: Config,
    bpw: float,
    head_bpw: int,
    mtp_bpw: int,
    hq: bool,
    vision_model: Model = None,
    vision_bpw: int = None,
) -> (dict, float):
    """
    Build the per-module quantization bitrate strategy for a converted model.

    Quantizable modules declare their quantization role in the model architecture, primarily through their qmap,
    qbits_key, qgroup, q_priority and select_hq_bits attributes. This function walks the module tree, aggregates
    all eligible Linear layers into qgroups, assigns an initial integer bitrate from the requested average bpw and
    then spends the remaining bit budget one bit at a time according to group priority. Auxiliary targets, such as
    output heads using head_bits, are collected alongside the main budgeted weights and merged into the returned
    strategy.
    """
    from ..modules.module import Module
    from ..modules.linear import Linear

    base_bpw = int(math.floor(bpw))
    sum_numel = 0
    sum_bits = 0
    targets = {}
    aux_targets = {}
    target_groups = {}

    # Collect and initialize targets
    def _add(module: Module, priority: int, fixed_bpw: int = None):
        priority = max(priority, module.q_priority)
        nonlocal sum_numel, sum_bits, targets, aux_targets
        if isinstance(module, Linear) and module.qmap is not None:
            if fixed_bpw is not None:
                # Uncalibrated side model (vision tower): every eligible Linear gets the same
                # fixed bitrate outside the main budget
                numel = module.weights_numel()
                aux_targets[module.key] = QTarget(
                    numel = numel,
                    target_bpw = fixed_bpw,
                    min_bpw = fixed_bpw,
                    priority = priority
                )

            elif module.qbits_key == "bits":
                numel = module.weights_numel()
                sum_numel += numel
                sum_bits += numel * base_bpw
                qt = QTarget(
                    numel = numel,
                    target_bpw = base_bpw,
                    min_bpw = min(base_bpw + module.select_hq_bits, 8),
                    priority = priority
                )
                targets[module.key] = qt
                if module.qgroup not in target_groups:
                    target_groups[module.qgroup] = []
                target_groups[module.qgroup].append(qt)

            elif module.qbits_key == "head_bits":
                numel = module.weights_numel()
                aux_targets[module.key] = QTarget(
                    numel = numel,
                    target_bpw = head_bpw,
                    min_bpw = head_bpw,
                    priority = priority
                )

            elif module.qbits_key == "mtp_bits":
                numel = module.weights_numel()
                aux_targets[module.key] = QTarget(
                    numel = numel,
                    target_bpw = mtp_bpw,
                    min_bpw = mtp_bpw,
                    priority = priority
                )

            else:
                raise ValueError("Logic error in create_q_strategy")
        for sm in module.modules:
            _add(sm, priority, fixed_bpw)

    modules = model.modules
    if mtp_model:
        modules = modules + mtp_model.modules
    for m in modules:
        _add(m, 0)
    if vision_model:
        for m in vision_model.modules:
            _add(m, 0, fixed_bpw = vision_bpw)

    # Target
    max_bits = int(bpw * float(sum_numel))

    # Promotion order: group priority first, then distance to the nearer end of the forward pass.
    # End layers contribute disproportionately to end-to-end error
    stack_max = {}
    group_meta = []
    for idx, (gkey, tlist) in enumerate(target_groups.items()):
        m = re.search(r"\.(\d+)\.", gkey)
        stack, layer = (gkey[:m.start()], int(m.group(1))) if m else (None, -1)
        if stack is not None:
            stack_max[stack] = max(stack_max.get(stack, -1), layer)
        group_meta.append((tlist, stack, layer, idx))

    def group_order_key(meta):
        tlist, stack, layer, idx = meta
        dist = 0 if stack is None else min(layer, stack_max[stack] - layer)
        return (-max(t.priority for t in tlist), dist, layer, idx)

    order = [meta[0] for meta in sorted(group_meta, key = group_order_key)]

    # Bump with priorities target met
    while sum_bits < max_bits:
        updates = False
        for target_list in order:
            cost = sum(t.delta_1() for t in target_list)
            if cost > 0 and sum_bits + cost <= max_bits:
                for t in target_list:
                    t.increase_1()
                updates = True
                sum_bits += cost
        if not updates:
            break

    # Apply constraint from --hq:
    final_bits = 0
    for t in targets.values():
        if hq:
            t.clamp_min()
        final_bits += t.total_bits()

    # Combined with head layer
    targets.update(aux_targets)

    # Keep only bitrate
    f_targets = {k: v.target_bpw for k, v in targets.items()}

    # Per-module bitrate overrides, applied after budget allocation.
    #
    # Upstream exposes only --bits/--head_bits/--mtp_bits/--vision_bits, and the
    # remaining budget is spent by a static priority order rather than by measured
    # error. This hook lets a caller pin specific modules, which is what a
    # mixed-precision recipe needs (e.g. down_proj at K6 because it carries the
    # largest per-tensor proxy error in every layer).
    #
    #   EXL3_BITS_OVERRIDE='{"^.*mlp\\.down_proj$": 6, "^.*mlp\\.(gate|up)_proj$": 5}'
    #
    # Keys are regexes matched against the module key; values are integer bpw 1-8.
    # The override is applied to f_targets only, so the reported average bpw still
    # reflects the budgeted allocation and the caller can see the difference.
    spec = os.environ.get("EXL3_BITS_OVERRIDE")
    if spec:
        import json as _json
        if os.path.isfile(spec):
            spec = open(spec).read()
        rules = [(re.compile(k), int(v)) for k, v in _json.loads(spec).items()]
        hits = {}
        for key in list(f_targets):
            for pat, bpw in rules:
                if pat.match(key):
                    if not 1 <= bpw <= 8:
                        raise ValueError(f"EXL3_BITS_OVERRIDE: {bpw} bpw out of range for {key}")
                    f_targets[key] = bpw
                    hits[pat.pattern] = hits.get(pat.pattern, 0) + 1
                    break
        print(" -- Bitrate overrides applied: " +
              ", ".join(f"{k} -> {dict(rules)[re.compile(k)] if False else ''}{n} modules"
                        for k, n in hits.items()), flush = True)
        for pat, bpw in rules:
            if pat.pattern not in hits:
                raise ValueError(f"EXL3_BITS_OVERRIDE: pattern {pat.pattern!r} matched nothing")

    return f_targets, float(final_bits) / sum_numel if sum_numel else 0


def print_strategy(
    strategy: dict
) -> str:
    pattern = re.compile(r'(?<=\.)\d+(?=\.)')
    output = {}
    max_length = 0
    for target_key, target_bpw in strategy.items():
        if target_bpw not in output:
            output[target_bpw] = {}
        mkey = pattern.sub("*", target_key)
        max_length = max(len(mkey), max_length)
        if mkey not in output[target_bpw]:
            output[target_bpw][mkey] = 0
        output[target_bpw][mkey] += 1

    r = ""
    for bpw, tensors in sorted(output.items()):
        r += f"     - {bpw} bpw:\n"
        for key, count in sorted(tensors.items()):
            r += f"        {key:{max_length}}   {count:8,} " + ("tensors\n" if count > 1 else "tensor\n")
    return r.rstrip()
