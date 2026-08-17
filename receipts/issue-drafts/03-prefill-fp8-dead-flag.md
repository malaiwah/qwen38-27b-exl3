# Issue draft 03 — VLLM_EXL3_PREFILL_FP8=1 silently no-ops when the extension lacks reconstruct_fp8_slice

**Repo:** local-inference-lab/vllm · **Cites:** docs/47 F5.6, plan P1.3

## Summary
`_reconstruct_fp8_mm_into` probes `hasattr(ext, "reconstruct_fp8_slice")` and returns False when
absent (exl3.py:877-878 at 4d006a4). The exllamav3 0.0.43 extension shipped in the r34 image does
NOT export that symbol (`bindings.cpp:95-101`; `nm`-verified on the shipped .so — it exports
`reconstruct_fp8dg_nt` but not `reconstruct_fp8_slice`). Result: setting `VLLM_EXL3_PREFILL_FP8=1`
silently runs the fp16 path — an operator tuning this flag measures noise. The documented
"+31 % prefill" required a rebuilt extension.

## Proposed fix
One-time `logger.warning` when the flag is set but the symbol is missing (in
`_prefill_fp8_enabled` or at the hasattr probe). Optionally: evaluate the shipped-but-unused
`reconstruct_fp8dg_nt` (DeepGEMM NT fp8, reconstruct.cu:144-326) as the binding-complete
implementation — noting the fork's own fidelity verdict (+0.0141 mean KLD) keeps FP8 prefill
default-off regardless.

## What would falsify this
An r34 image whose `VLLM_EXL3_EXT_PATH` .so exports `reconstruct_fp8_slice` (then the flag is live
and the docs/41 state-(e) entry is wrong for that image). Check: `python -c "import exllamav3_ext as
e; print(hasattr(e,'reconstruct_fp8_slice'))"`.
