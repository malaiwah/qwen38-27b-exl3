# R37 — SLQ-style exact fidelity frontier

**Status:** final Phase-B scientific negative; zero eligible measured action rows.

## Result

No numerical model frontier or heterogeneous assignment is reported. Every R32–R36
lane explicitly completed or falsified without a row that simultaneously supplied a
complete current-R30 action, exact serialized bytes, and direct validation
full-vocabulary KLD/EAR/p99/CVaR/top1 evidence.

Proxy and local-only values remain excluded. This is a promotion-gated evidence
negative, not a claim that the underlying local methods can never improve fidelity.

The method-of-record receipt is
`receipts/wave5/r37-slq-frontier.json`.

## Exact allocator

`tools/research/wave5/r37_slq_frontier.py` implements:

- one complete stock-format action per legal topology/fused unit;
- exact action rate as R30 serialized buffer bytes plus action sidecars;
- a separately itemized fixed checkpoint byte sum for headers, alignment, and
  non-action components;
- direct single-group validation full-vocabulary marginals only;
- targeted interactions only when they are high-sensitivity/fused evidence and
  enter allocation as a directly remeasured complete grouped action;
- an exact multiple-choice dynamic program at every reachable integer byte budget;
- lexicographic mean KLD, p99, CVaR1%, negative EAR, negative top1, runtime, and
  startup ranking with canonical assignment-SHA tie breaking;
- the frozen mean-KLD, p99, CVaR1%, EAR, and top1 gates plus codec-exact route,
  runtime, startup, graph, context, and no-fallback gates;
- stock hydrated/current recipe, local EDA negative, uniform K6, and best
  whole-module K5/K6 mix as mandatory controls;
- role/depth/K frequencies for a selected assignment;
- a fail-closed prohibition on per-tile selectors, entropy coding, and new payload
  buffers.

The DP's composed points use the separate
`wave5/r37-screening-frontier/1` schema and are non-promotable. Only a fresh
sequential whole-checkpoint rebuild with verified manifests and a direct validation
measurement may satisfy the primal, because later activations depend on earlier choices.

## Solver verification

The targeted command was:

```text
/Users/mbelleau/Projects/qwen38-research-venv/bin/python tools/research/wave5/r37_slq_frontier.py self-test
```

It passed an independent Cartesian oracle over two units and six assignments,
including a true equal-metric tie. The DP and oracle produced the same reachable
exact budgets: 200, 220, and 240 bytes, preserved the full tie equivalence class,
and selected its canonical SHA. The fixture also checks exact component summation,
fresh sequential selection, screen-only non-promotion, strict incomplete-action
rejection, CI/mean consistency, new-format and untouched-test rejection, and that
screening cannot open R38.

This qualifies only the targeted additive DP/oracle behavior, not production
normalization or a Qwen3.8-27B frontier. Two adversarial review rounds ended
REJECT for production use: full R30 semantic validation, semantic report/proof
validation, frozen hashes for three named controls, and a complete schema-valid
production fixture remain unresolved. Therefore this interim artifact cannot emit
a measured registry selection, promotable frontier, exact checkpoint claim, or R38
proceed decision.

## Action evidence gate

### Historical R30 K4/K5/K6 panel

`receipts/wave5/stock-control.json` contains actual stock payloads and exact buffer
bytes for a 128×128 K4/K5/K6 codec panel. Its reported distortion is a reconstruction
proxy, not validation model KLD/EAR, and it is not a whole-model action registry. It
was excluded from selection.

The historical A0 receipt remains useful for codec identity at its recorded
historical pins. Current intake binds harness
`d4dfd35cd7b85beab11d33de110eb240ca87162e4a01ec434cb19e5b6a82605d`,
schema `275644ed86017f54953d7eecd2f843e6b6f6c14ae52df163ef5827179edf7af8`,
and qualified extension
`e2e26e0dcfa6eb637215c673a30522076c9d530140cd0d5c727ca549f2d8801e`.
Historical runs are not relabeled.

### Final R32–R36 intake

R32 was local-positive but validation-blocked: all ten K5 actions were 55,750,660
bytes, yet the approved launcher produced zero validation KLD rows. R33 selected no
action; stock rho=1 won local HWE, and its ~4.86 values were SNR-equivalent bits,
not measured QMM headroom. R34 failed the block-output gate after 7.80%–30.98%
MSE regressions. R35 remained local-only. R36 failed its hard-path stability gate
at 11/20 versus the required 16/20. Thus the final registry has zero eligible rows.

### R35 Schur refinement

R35 explicitly supplied a no-eligible-row outcome in
`receipts/wave5/r35-schur-refine.json`
(`ac2071d026d1fec26e829302ba2bafda319c7e19118174790303de6dc407959c`).
Full L0-gate same-byte calibration dense-H changes were −0.018% at K4, −0.060% at
K5, and −0.122% at K6. R35 had neither a direct validation full-vocabulary KLD/EAR
row nor an R34 reconstructed-target factorial. Its receipt remains historical
execution evidence under its recorded old extension/harness/schema; it is not
relabeled to the current operational pins. All R35 values were excluded.

## Controls and requested reports

None of the four required controls had a complete direct validation row in an R37
registry. Consequently:

- exact checkpoint byte sum: not measured;
- selected assignment file: no assignment exists;
- validation mean KLD/EAR/p99/CVaR/top1 constraints: not evaluated;
- role/depth action frequencies: not applicable;
- best measured screened menu: none;
- global optimum claim: none.

The untouched test was not opened.

## R38 residual gate

R38 remains stopped. R37 has no selected best measured whole-module K5/K6
assignment, no exact target byte budget, and no eligible QMM effective-bit residual
bound to such an assignment. A local or additive screening signal cannot open the
custom-format lane.

## Foundation identity

R37 binds the R29 data file/content hashes
`68bcc5ddce1d34f71d696265d908eccd1b75f48444cb0a3aaffe86fea02bff37` /
`51957ac986dc44bc06f937ae74b005e090883348c947ef65ac331ed5a91057c2`,
the combined split file/content hashes
`a7eab6e2d8ee78e8d27655f8e9caf4c7813c43539ba24b31c4941d3d38ee09cc` /
`151c41151142060619e6a7957f36daa4849e53276435df54b74bdfc223596a2e`,
the selected Fisher manifest file/content hashes
`4541b2ed392d518eaec24cb4ac2936757cb21cb1148857b08e3e4840fbca8b9a` /
`28d3b59353e9b8aab2be47bace1086431b59e55e62cc52d47317729541d26237`,
and validation projection
`4c5cf19acc18835ee6d36da91b2b93135c5d33655ca410c079d4d4be83c5a5de`.
R37 never accessed the untouched-test contents.
