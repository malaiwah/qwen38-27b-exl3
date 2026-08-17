#!/usr/bin/env bash
# Build, verify and receipt the immutable production runtime (P0 / rank 2).
#
# Two images, deliberately separated:
#
#   VARIANT=release (default)  localhost/vllm:gg-r34-patched
#       The release unit: the pinned r34 digest plus exactly the three published modules.
#       This is the digest that gets hardware-qualified and the one the model cards pin.
#
#   VARIANT=apc                localhost/vllm:gg-r34-patched-apc
#       A declared superset of the release image adding only vllm/v1/core/sched/scheduler.py
#       from upstream PR #51113. Not qualified. Only needed to enable prefix caching, which
#       is default-off for this hybrid model, so no published measurement depends on it.
#
# Every stage writes JSON fragments under a per-variant staging directory; `receipt`
# composes both variants into receipts/production-image.json atomically. Stages are
# separate because the GPU stage (`smoke`) contends with qualification runs while
# `build`/`verify`/`sbom` do not touch the GPU at all.
#
#   docker/build-image.sh build                 verify sources, podman build, record digest
#   docker/build-image.sh verify                re-verify the modules INSIDE the image
#   docker/build-image.sh sbom                  syft if present, else pip freeze + dpkg -l
#   docker/build-image.sh plan k4 context       print the exact podman argv, run nothing
#   docker/build-image.sh smoke k4 k5k6 ...     start a model-card recipe, text+image request
#   docker/build-image.sh receipt               compose receipts/production-image.json
#
# Prefix any stage with VARIANT=apc to operate on the superset image instead.
#
# Requires: podman, python3, sha256sum. Runs rootless as the invoking user; no stage ever
# needs host root, and no stage touches an already-running container.
set -euo pipefail

# Preflight the documented requirements explicitly. `set -euo pipefail` does abort on a missing
# binary, but only once some stage happens to invoke it, which surfaces as a confusing failure
# deep in a build rather than a one-line "you are missing podman". A peer review flagged the
# header's "Requires:" line as documentation with no enforcement behind it; this is the enforcement.
missing=()
for _bin in podman python3 sha256sum; do
  command -v "$_bin" >/dev/null 2>&1 || missing+=("$_bin")
done
if (( ${#missing[@]} )); then
  printf 'build-image.sh: missing required binaries: %s\n' "${missing[*]}" >&2
  printf 'build-image.sh: this script runs rootless and needs all of: podman python3 sha256sum\n' >&2
  exit 127
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
STAGE=${STAGE:-$REPO_ROOT/receipts/.production-image-stage}
RECEIPT=${RECEIPT:-$REPO_ROOT/receipts/production-image.json}
BASE_REF=docker.io/voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b
BASE_TAG=docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34
VLLM_DIR=/opt/venv/lib/python3.12/site-packages/vllm
MODELS_ROOT=${MODELS_ROOT:-/mnt/vault/llm/huggingface}
SMOKE_PORT=${SMOKE_PORT:-8137}
CACHE_VOLUME=${CACHE_VOLUME:-gg-r34-patched-jit}
# The base image bakes every JIT/compile cache path under a fingerprint
# (LOCAL_INFERENCE_CACHE_FINGERPRINT); a three-file Python overlay leaves it unchanged, so
# an existing warm cache from the same base digest is reusable. Point these at warm host
# directories to avoid a cold FlashInfer/Triton/CUDA-graph compile; leave them unset to
# use a private named volume mounted at /cache instead.
CACHE_JIT_DIR=${CACHE_JIT_DIR:-}
CACHE_EXL3_DIR=${CACHE_EXL3_DIR:-}

RELEASE_TAG=${RELEASE_TAG:-localhost/vllm:gg-r34-patched}
APC_TAG=${APC_TAG:-localhost/vllm:gg-r34-patched-apc}
CONVERT_TAG=${CONVERT_TAG:-localhost/vllm:gg-r34-convert}
VARIANT=${VARIANT:-release}

# published source map: source file -> destination inside the image -> published sha256
EXL3_MODULE="tools/vllm-exl3-prefill-dispatch.py|$VLLM_DIR/model_executor/layers/quantization/exl3.py|2df9d0799fd323798cead1edb773cab556c94798eec263ee03ded35408c6e4ee"
QWEN_MODULE="tools/vllm-qwen3_5-embed-quant-config.py|$VLLM_DIR/model_executor/models/qwen3_5.py|04d2bd587b37142f4f55a8d00b9f8c907309490168cb7fcdfde450531df2c9e7"
MTP_MODULE="tools/vllm-qwen3_5_mtp-embed-quant-config.py|$VLLM_DIR/model_executor/models/qwen3_5_mtp.py|0090dc131f0eaf439b24d50baf4def9f10b052864c76e695053d64f66b274bab"
SCHED_MODULE="tools/vllm-mamba-align-scheduler.py|$VLLM_DIR/v1/core/sched/scheduler.py|b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf"

case $VARIANT in
  release)
    IMAGE_TAG=${IMAGE_TAG:-$RELEASE_TAG}
    DOCKERFILE=${DOCKERFILE:-$REPO_ROOT/docker/Dockerfile.gg-r34-patched}
    # PATCH_MAP: verified inside the image. LAYER_ADDS: what this layer adds over PARENT_REF.
    PATCH_MAP=("$EXL3_MODULE" "$QWEN_MODULE" "$MTP_MODULE")
    LAYER_ADDS=("$EXL3_MODULE" "$QWEN_MODULE" "$MTP_MODULE")
    PARENT_REF=$BASE_REF
    PARENT_DESC="pinned public r34 digest"
    SBOM_INFIX=""
    QUALIFIABLE=true
    DECLARATION_JSON='null'
    ;;
  apc)
    IMAGE_TAG=${IMAGE_TAG:-$APC_TAG}
    DOCKERFILE=${DOCKERFILE:-$REPO_ROOT/docker/Dockerfile.gg-r34-patched-apc}
    PATCH_MAP=("$EXL3_MODULE" "$QWEN_MODULE" "$MTP_MODULE" "$SCHED_MODULE")
    LAYER_ADDS=("$SCHED_MODULE")
    PARENT_REF=$RELEASE_TAG
    PARENT_DESC="the release image localhost/vllm:gg-r34-patched"
    SBOM_INFIX="-apc"
    QUALIFIABLE=true
    DECLARATION_JSON=$(cat <<'JSON'
{
  "role": "the promoted release unit: a declared superset of localhost/vllm:gg-r34-patched, and the image every published recipe now names",
  "qualified": true,
  "qualified_by": {
    "with_prefix_caching_off": "receipts/qualification-5090-context.json transfers to this digest unchanged. This image is its parent plus one module whose changed function, Scheduler._mamba_block_aligned_split, is only reachable when mamba_cache_mode is 'align', which only happens when prefix caching is enabled. With the cache off the promoted image is behaviourally identical to the qualified parent, not merely similar, and the four-recipe serving smoke in this receipt was run on the promoted digest.",
    "with_prefix_caching_on_at_8192": "receipts/qualification-5090-apc.json plus this receipt's own smoke: the k4, k5k6 and hydrated recipes all start healthy on this digest with --enable-prefix-caching --mamba-cache-mode align, answer a text and an image request exactly, and report enable_prefix_caching True in the engine banner. Pool is roughly 32x the window at 8,192, so the near-ceiling failures the context profile hit cannot occur.",
    "with_prefix_caching_on_at_262144": "NOT QUALIFIED, and not for want of trying. receipts/qualification-5090-apc.json records a startup refusal at utilisation 0.955, a deadlock at 0.9555 and a livelock at 0.9585, with the engine's own arithmetic. The context edition's native-window recipe therefore keeps prefix caching OFF. This is a limit of the window-and-image-ceiling combination on a 32,607 MiB card, not a defect of this image."
  },
  "what_promotion_did_and_did_not_change": "promotion moved the release unit from the three-module digest to this four-module one, which is what allows any recipe to enable prefix caching at all. It did not, by itself, enable prefix caching everywhere: three of the four published recipes turn it on, the context edition's native-window recipe does not.",
  "promoted_utc_date": "2026-08-16",
  "promoted_by": "Main's call, recorded in promotion_status",
  "supersedes_as_release_unit": {
    "tag": "localhost/vllm:gg-r34-patched",
    "manifest_digest": "sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27",
    "still_valid_as": "this image's parent, the top-level build record of this receipt, and the digest receipts/qualification-5090-context.json was measured against. It is superseded as the recommended runtime, not invalidated as evidence."
  },
  "delta_vs_release_image": ["/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"],
  "why_promoted": "prefix caching is the largest untapped win for repeated long prompts on this model, and it cannot be enabled safely on the parent: unpatched, upstream #43559 returns wrong tokens rather than failing. Promotion is what lets the published recipes turn it on.",
  "not_rebuilt_for_promotion": {
    "claim": "no layer was added, no label edited and no tag re-pointed as part of the promotion. The manifest digest promoted is byte-identical to the one built on 2026-08-16T03:16:01Z.",
    "why": "this digest is the one ApcPoisonRepro's arms C and E measured with no bind mounts. Any relabelling is a new layer and a new digest, which would throw away the only prior evidence on the identical bytes and leave the qualification standing alone.",
    "stale_label_in_image": {
      "label": "io.malaiwah.image.qualified=\"false\"",
      "also_stale": "io.malaiwah.image.role=\"prefix-caching prerequisite; requires its own qualification run before --enable-prefix-caching is used\", and org.opencontainers.image.description, which calls this image \"Not the hardware-qualified digest\"",
      "status": "accurate as of build time, wrong as of promotion, and deliberately left uncorrected",
      "precedence": "receipts/production-image.json is authoritative for qualification state; the in-image labels are a build-time statement only. A reader who trusts the label over the receipt reaches the exact inversion of the truth, which is why this is recorded here rather than only in prose."
    }
  },
  "prefix_caching_by_recipe": {
    "on": ["k4", "k5k6", "hydrated"],
    "off": ["context, at the native 262,144 window: it does not fit, see qualified_by"],
    "flags": ["--enable-prefix-caching", "--mamba-cache-mode align"],
    "mamba_cache_mode_is_derivable": "the engine sets mamba_cache_mode to align on its own once prefix caching is enabled (vllm/model_executor/models/config.py:558-562); both flags exist in this build (vllm serve --help=all lists --enable-prefix-caching/--no-enable-prefix-caching and --mamba-cache-mode {align,all,none}) and the recipe prints both so a reader has something to check the engine banner against",
    "was_off_before": "prefix caching is default-off for this hybrid model (vllm/engine/arg_utils.py:2532-2534 excludes hybrid models), so every archived smoke row on the parent image reads enable_prefix_caching=False. Those rows are the superseded three-module image, not a failure.",
    "patched_function_now_reached": "Scheduler._mamba_block_aligned_split, guarded by has_mamba_layers and mamba_cache_mode == 'align' (vllm/v1/core/sched/scheduler.py:316-318). Before promotion it was never called anywhere; from promotion onwards it is on the hot path of every request served by the three recipes that enable the cache, and still unreachable in the context edition's native-window recipe, which does not."
  },
  "upstream": {
    "pr": "https://github.com/vllm-project/vllm/pull/51113",
    "title": "Keep mamba align prefill chunks block-aligned past last_cache_position",
    "merge_commit": "c56f169d9ae46ca420617e2cf5f0c9135da0f651",
    "merged_utc_date": "2026-08-06",
    "fixes_issue": "https://github.com/vllm-project/vllm/issues/43559"
  },
  "provenance": {
    "vendored_scheduler_sha256_in_r34": "1ea341f4cc28d282452597c25d97eea84be8b5f984d2e1a6b548356c8417fdce",
    "patch": "patches/vllm-51113-mamba-align.patch",
    "patch_reproduces_module_byte_identically": true,
    "evidence_receipt": "receipts/mamba-align-defect.json",
    "evidence_receipt_sha256": "a654f3df5776466575a6706a73f3ea369b73587e6fb832a6ea5917f617c31e82",
    "evidence_receipt_owner": "MambaAlignProof; that receipt is authoritative for the defect and the CPU-only test result, this one for the image identities"
  },
  "cpu_only_evidence": {
    "test_file": "upstream tests/v1/core/test_mamba_align_chunk_split.py, run unmodified",
    "vendored_r34_scheduler": "14 failed, 6 passed",
    "this_module": "20 passed",
    "gpu_used": false,
    "why_it_matters_for_the_promotion": "the patch is carried as insurance backed by upstream's own regression file, not by a reproduction of our own: receipts/apc-poison-repro.json failed to reproduce the reported corruption on the unpatched parent across 266 scored requests"
  }
}
JSON
)
    ;;
  convert)
    IMAGE_TAG=${IMAGE_TAG:-$CONVERT_TAG}
    DOCKERFILE=${DOCKERFILE:-$REPO_ROOT/docker/Dockerfile.gg-r34-convert}
    PATCH_MAP=("$EXL3_MODULE" "$QWEN_MODULE" "$MTP_MODULE")
    LAYER_ADDS=()
    PARENT_REF=$RELEASE_TAG
    PARENT_DESC="the release image localhost/vllm:gg-r34-patched"
    SBOM_INFIX="-convert"
    QUALIFIABLE=false
    DECLARATION_JSON=$(cat <<'JSON'
{
  "role": "conversion and quantization path only; not a serving runtime",
  "qualified": false,
  "qualification_note": "the serving qualification and receipts/production-image.json are pinned to localhost/vllm:gg-r34-patched, whose digest this tag does not change. Adding a layer to the serving image would have invalidated that pin to fix a build-tool import, which is the wrong trade.",
  "adds_no_python_modules_under_vllm": true,
  "added_package": {
    "name": "marisa_trie",
    "version": "1.4.1",
    "wheel": "marisa_trie-1.4.1-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
    "wheel_sha256": "4d51bdd22a7238ef4d681effd7c224a267ddae054b64b1cec9ce95bbcd2b6a88",
    "installed_with": "pip install --no-deps --no-cache-dir on a locally fetched, digest-verified wheel, so no dependency of the inventoried venv can be resolved or upgraded",
    "why": "imported by exllamav3/loader/safetensors.py, which both conversion entry points reach: compile_model and util/add_quant_config.py",
    "cost_of_the_gap": "one 2h05 ladder pass died at compile_model with all 409 modules already quantized, and one conversion died at add_quant_config with the checkpoint already written; the import is at the end of a long job, so the gap is expensive every time",
    "never_imported_at_serve_time": true
  },
  "supersedes_workaround": "the local shim at /work/kld6/run_shimmed.py; with this tag no agent needs to shim around the missing import"
}
JSON
)
    ;;
  *) printf 'FAIL: unknown VARIANT %s (expected release, apc or convert)\n' "$VARIANT" >&2
     exit 1 ;;
esac
VARIANT_STAGE=$STAGE/$VARIANT

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
digest_of() { sha256sum "$1" | cut -d' ' -f1; }

json_fragment() {  # json_fragment <name> ; body on stdin
  mkdir -p "$VARIANT_STAGE"
  cat > "$VARIANT_STAGE/$1.json"
  python3 -c 'import json,sys;json.load(open(sys.argv[1]))' "$VARIANT_STAGE/$1.json" \
    || die "fragment $1 is not valid JSON"
  log "wrote $VARIANT fragment $1"
}
# --------------------------------------------------------------------------- build
stage_build() {
  local context=$VARIANT_STAGE/context
  rm -rf "$context"
  mkdir -p "$context/tools"
  local entry source dest want got sources_json="" adds_json=""
  # Every module the image must contain is digest-checked in the repo first, even the ones
  # this layer inherits rather than copies.
  for entry in "${PATCH_MAP[@]}"; do
    IFS='|' read -r source dest want <<<"$entry"
    [[ -f $REPO_ROOT/$source ]] || die "missing source $source"
    got=$(digest_of "$REPO_ROOT/$source")
    [[ $got == "$want" ]] || die "$source digest $got != published $want"
    sources_json+="{\"source\":\"$source\",\"dest\":\"$dest\",\"sha256\":\"$got\",\"bytes\":$(stat -c%s "$REPO_ROOT/$source")},"
    log "source verified $source $got"
  done
  # Only what this Dockerfile actually COPYs goes into the build context.
  for entry in "${LAYER_ADDS[@]}"; do
    IFS='|' read -r source dest want <<<"$entry"
    cp -f -- "$REPO_ROOT/$source" "$context/$source"
    adds_json+="\"$dest\","
  done
  cp -f -- "$DOCKERFILE" "$context/Dockerfile"

  podman image inspect "$PARENT_REF" >/dev/null 2>&1 \
    || die "parent image $PARENT_REF is not present locally; build or pull it first"

  local build_started build_command
  build_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local build_argv=(podman build --pull=never -f "$context/Dockerfile" -t "$IMAGE_TAG")
  [[ $PARENT_REF != "$BASE_REF" ]] && build_argv+=(--build-arg "PARENT=$PARENT_REF")
  build_argv+=("$context")
  build_command=$(python3 -c 'import json,sys;print(json.dumps(" ".join(sys.argv[1:]))[1:-1])' \
    "${build_argv[@]}")
  log "building $IMAGE_TAG from $PARENT_REF"
  "${build_argv[@]}" >"$VARIANT_STAGE/build.log" 2>&1 \
    || { tail -40 "$VARIANT_STAGE/build.log" >&2
         die "podman build failed (see $VARIANT_STAGE/build.log)"; }
  tail -5 "$VARIANT_STAGE/build.log" >&2

  local image_id image_digest created size
  image_id=$(podman image inspect "$IMAGE_TAG" --format '{{.Id}}')
  image_digest=$(podman image inspect "$IMAGE_TAG" --format '{{.Digest}}')
  created=$(podman image inspect "$IMAGE_TAG" --format '{{.Created}}')
  size=$(podman image inspect "$IMAGE_TAG" --format '{{.Size}}')

  json_fragment build <<JSON
{
  "variant": "$VARIANT",
  "declaration": $DECLARATION_JSON,
  "base_image": {
    "reference": "$BASE_REF",
    "tag": "$BASE_TAG",
    "config_id": "$(podman image inspect "$BASE_REF" --format '{{.Id}}')",
    "present_locally_before_build": true,
    "pull_during_build": false
  },
  "parent_image": {
    "reference": "$PARENT_REF",
    "description": "$PARENT_DESC",
    "config_id": "$(podman image inspect "$PARENT_REF" --format '{{.Id}}')",
    "manifest_digest": "$(podman image inspect "$PARENT_REF" --format '{{.Digest}}')"
  },
  "image": {
    "tag": "$IMAGE_TAG",
    "config_id": "sha256:$image_id",
    "manifest_digest": "$image_digest",
    "created_utc": "$created",
    "size_bytes": $size,
    "repo_digests": $(podman image inspect "$IMAGE_TAG" --format '{{json .RepoDigests}}'),
    "pushed_to_registry": false
  },
  "build": {
    "started_utc": "$build_started",
    "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "host": "$(hostname)",
    "podman_version": "$(podman --version | awk '{print $3}')",
    "command": "$build_command",
    "context": "minimal: Dockerfile plus only the module(s) this layer copies",
    "context_files": $(cd "$context" && find . -type f | sort | python3 -c 'import json,sys;print(json.dumps([l.strip()[2:] for l in sys.stdin]))'),
    "dockerfile_sha256": "$(digest_of "$DOCKERFILE")",
    "build_script_sha256": "$(digest_of "${BASH_SOURCE[0]}")",
    "smoke_client_sha256": "$(digest_of "$REPO_ROOT/docker/smoke_client.py")",
    "build_log_sha256": "$(digest_of "$VARIANT_STAGE/build.log")"
  },
  "source_map": [${sources_json%,}],
  "added_by_this_layer": [${adds_json%,}]
}
JSON
}

# -------------------------------------------------------------------------- verify
# Everything here is measured inside the built image, not asserted from the Dockerfile.
stage_verify() {
  local entry source dest want script="set -eu"
  for entry in "${PATCH_MAP[@]}"; do
    IFS='|' read -r source dest want <<<"$entry"
    script+="; sha256sum '$dest'"
  done
  local inside
  inside=$(podman run --rm --network none --entrypoint /bin/bash "$IMAGE_TAG" -lc "$script")

  local modules="" ok=true
  for entry in "${PATCH_MAP[@]}"; do
    IFS='|' read -r source dest want <<<"$entry"
    local got base_got parent_got
    got=$(awk -v d="$dest" '$2==d{print $1}' <<<"$inside")
    base_got=$(podman run --rm --network none --entrypoint /bin/bash "$BASE_REF" \
      -lc "sha256sum '$dest' | cut -d' ' -f1")
    if [[ $PARENT_REF == "$BASE_REF" ]]; then
      parent_got=$base_got
    else
      parent_got=$(podman run --rm --network none --entrypoint /bin/bash "$PARENT_REF" \
        -lc "sha256sum '$dest' | cut -d' ' -f1")
    fi
    [[ $got == "$want" ]] || ok=false
    modules+="{\"source\":\"$source\",\"dest\":\"$dest\",\"published_sha256\":\"$want\",\"in_image_sha256\":\"$got\",\"base_image_sha256\":\"$base_got\",\"parent_image_sha256\":\"$parent_got\",\"matches_published_map\":$([[ $got == "$want" ]] && echo true || echo false),\"differs_from_base\":$([[ $got != "$base_got" ]] && echo true || echo false),\"added_by_this_layer\":$([[ $got != "$parent_got" ]] && echo true || echo false)},"
    log "in-image $dest -> $got"
  done
  $ok || die "in-image module digests do not match the published map"

  local expected_adds=""
  for entry in "${LAYER_ADDS[@]}"; do
    IFS='|' read -r source dest want <<<"$entry"
    expected_adds+="$dest "
  done

  # Nothing under site-packages/vllm may differ from the parent except this layer's adds,
  # and nothing may differ from the pinned public base except the whole verified set.
  manifest_of() {
    podman run --rm --network none --entrypoint /bin/bash "$1" \
      -lc "cd $VLLM_DIR && find . -name '*.py' -type f -exec sha256sum {} + | sort -k2"
  }
  local manifest_new manifest_parent manifest_base changed_parent changed_base
  manifest_new=$(manifest_of "$IMAGE_TAG")
  manifest_parent=$(manifest_of "$PARENT_REF")
  if [[ $PARENT_REF == "$BASE_REF" ]]; then
    manifest_base=$manifest_parent
  else
    manifest_base=$(manifest_of "$BASE_REF")
  fi
  changed_parent=$(diff <(echo "$manifest_parent") <(echo "$manifest_new") \
    | grep '^>' | awk '{print $3}' | sort || true)
  changed_base=$(diff <(echo "$manifest_base") <(echo "$manifest_new") \
    | grep '^>' | awk '{print $3}' | sort || true)

  # Image config must be inherited unchanged: no new entrypoint, command, user or env.
  local cfg_new cfg_parent env_added
  cfg_new=$(podman image inspect "$IMAGE_TAG" --format '{{json .Config.Entrypoint}}|{{json .Config.Cmd}}|{{.Config.User}}|{{.Config.WorkingDir}}')
  cfg_parent=$(podman image inspect "$PARENT_REF" --format '{{json .Config.Entrypoint}}|{{json .Config.Cmd}}|{{.Config.User}}|{{.Config.WorkingDir}}')
  env_added=$(diff \
      <(podman image inspect "$BASE_REF" --format '{{range .Config.Env}}{{println .}}{{end}}' | sort) \
      <(podman image inspect "$IMAGE_TAG" --format '{{range .Config.Env}}{{println .}}{{end}}' | sort) \
    | grep '^>' | sed 's/^> //' | sort || true)

  # Added history entries must be COPY/LABEL plus the one verifying RUN, and that RUN
  # must contain no package manager invocation.
  local added_history installer_hits
  added_history=$(diff \
      <(podman image history "$BASE_REF" --no-trunc --format '{{.CreatedBy}}') \
      <(podman image history "$IMAGE_TAG" --no-trunc --format '{{.CreatedBy}}') \
    | grep '^>' | sed 's/^> //' || true)
  installer_hits=$(grep -Eic 'pip[[:space:]]+install|apt-get|apt install|dnf |yum |uv pip|conda install|curl .*\|.*sh' \
    <<<"$added_history" || true)

  python3 - "$VARIANT_STAGE" "$modules" <<'PY' \
    "$cfg_new" "$cfg_parent" "$env_added" "$added_history" "$installer_hits" \
    "$changed_parent" "$changed_base" "$expected_adds" "$PARENT_REF" \
    "$(podman image inspect "$IMAGE_TAG" --format '{{.Config.WorkingDir}}')"
import json, sys
(stage, modules, cfg_new, cfg_parent, env_added, history, installer_hits,
 changed_parent, changed_base, expected_adds, parent_ref, working_dir) = sys.argv[1:13]
PREFIX = "/opt/venv/lib/python3.12/site-packages/vllm"
patch_modules = json.loads("[" + modules.rstrip(",") + "]")

def absolute(listing):
    # find prints ./relative paths under the vllm package
    return {PREFIX + path[1:] for path in listing.split() if path}

changed_from_parent = absolute(changed_parent)
changed_from_base = absolute(changed_base)
expected_from_parent = {path for path in expected_adds.split() if path}
expected_from_base = {module["dest"] for module in patch_modules}
payload = {
    "patch_modules": patch_modules,
    "immutability": {
        "parent_image": parent_ref,
        "python_files_changed_vs_parent": sorted(changed_from_parent),
        "python_files_changed_vs_parent_are_exactly_this_layers_adds":
            changed_from_parent == expected_from_parent,
        "python_files_changed_vs_base": sorted(changed_from_base),
        "python_files_changed_are_exactly_the_patch_set":
            changed_from_base == expected_from_base,
        "image_config_entrypoint_cmd_user_workdir_unchanged": cfg_new == cfg_parent,
        "image_config_inherited": cfg_new,
        "image_config_working_dir": working_dir or "/",
        "image_config_working_dir_is_the_duplicate_source_tree":
            (working_dir or "/").rstrip("/") == "/opt/vllm",
        "env_vars_added_vs_base_image": [e for e in env_added.splitlines() if e.strip()],
        "added_history_entries_vs_base": [h for h in history.splitlines() if h.strip()],
        "package_manager_invocations_in_added_layers": int(installer_hits or 0),
        "runtime_package_installs": False if int(installer_hits or 0) == 0 else True,
        "source_bind_mounts_required_at_runtime": False,
        "startup_file_copies_required": False,
    },
}
with open(f"{stage}/verify.json", "w") as handle:
    json.dump(payload, handle, indent=2)
print(json.dumps(payload["immutability"], indent=2))
PY
  log "wrote fragment verify"

  # Strongest form of the claim: Python's own import machinery, inside the image, resolves
  # each module name to the patched file, and no stale bytecode shadows it. A digest at a
  # path only proves the path; this proves what would actually be imported.
  local module_names=() find_names=() relative dotted
  for entry in "${PATCH_MAP[@]}"; do
    IFS='|' read -r source dest want <<<"$entry"
    relative=${dest#"$VLLM_DIR"/}
    dotted=vllm.${relative%.py}
    module_names+=("${dotted//\//.}")
    find_names+=(-o -name "$(basename "$dest")")
  done
  # The image also carries an unimported second copy of the vLLM sources at /opt/vllm.
  # An auditor will find it, so measure it: record its digests and prove it is not on
  # sys.path, rather than leaving "only the imported tree is patched" as an assertion.
  local resolution_json duplicates
  resolution_json=$(podman run --rm -i --network none \
    --entrypoint /opt/venv/bin/python "$IMAGE_TAG" - "${module_names[@]}" <<'PY'
import hashlib, importlib.util, json, os, sys


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


SECOND_TREE = "/opt/vllm"
rows = {}
for module in sys.argv[1:]:
    try:
        spec = importlib.util.find_spec(module)
        relative = module.replace(".", "/") + ".py"
        duplicate = os.path.join(SECOND_TREE, relative)
        rows[module] = {
            "resolved_origin": spec.origin,
            "resolved_origin_sha256": digest(spec.origin),
            "cached_bytecode_path": spec.cached,
            "stale_bytecode_present": os.path.exists(spec.cached or ""),
            "unimported_duplicate_path": duplicate if os.path.exists(duplicate) else None,
            "unimported_duplicate_sha256":
                digest(duplicate) if os.path.exists(duplicate) else None,
        }
    except Exception as exc:
        rows[module] = {"error": f"{type(exc).__name__}: {exc}"}
# A console-script launch (/opt/venv/bin/vllm, what every recipe uses) puts the script
# directory at sys.path[0]. `python -m` and `python -c` instead prepend the working
# directory, so if the inherited WORKDIR were /opt/vllm, `import vllm` would silently
# resolve to the unpatched duplicate tree and defeat every patched module. This probe is
# deliberately run as `python -` with no -w, i.e. with cwd-prepending semantics and the
# image's own WorkingDir, so it exercises exactly that case.
site = os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm").origin))
path_files = {}
for name in sorted(os.listdir(site)):
    if name.endswith(".pth"):
        with open(os.path.join(site, name), encoding="utf-8", errors="replace") as handle:
            path_files[name] = [line.strip() for line in handle if line.strip()]
direct_url = None
for name in os.listdir(site):
    if name.startswith("vllm-") and name.endswith(".dist-info"):
        candidate = os.path.join(site, name, "direct_url.json")
        if os.path.exists(candidate):
            direct_url = json.load(open(candidate))
print(json.dumps({
    "probe_invocation": "podman run --entrypoint /opt/venv/bin/python IMAGE - "
                        "(script on stdin, no -w), so sys.path[0] is derived from the "
                        "working directory exactly as `python -m` or `python -c` would be",
    "cwd": os.getcwd(),
    "sys_path_0": sys.path[0],
    "sys_path": sys.path,
    "site_packages": site,
    "path_configuration_files": path_files,
    "editable_install_pth_present":
        any(name.startswith("__editable__") for name in path_files),
    "vllm_dist_info_direct_url": direct_url,
    "vllm_installed_editable": bool((direct_url or {}).get("dir_info", {}).get("editable")),
    "second_source_tree": SECOND_TREE,
    "second_source_tree_on_sys_path":
        any(os.path.realpath(entry) == os.path.realpath(SECOND_TREE)
            for entry in sys.path if entry),
    "second_source_tree_reachable_from_cwd":
        os.path.realpath(os.path.join(os.getcwd(), "vllm"))
        == os.path.realpath(os.path.join(SECOND_TREE, "vllm")),
    "modules": rows,
}))
PY
)
  duplicates=$(podman run --rm --network none --entrypoint /bin/bash "$IMAGE_TAG" -lc \
    "find / -xdev \( ${find_names[*]:1} \) -not -path '$VLLM_DIR/*' 2>/dev/null | sort \
       | python3 -c \
       'import json,sys;print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))'" \
    || echo '[]')
  json_fragment resolution <<JSON
{
  "import_resolution": {
    "measured_in": "$IMAGE_TAG",
    "measured": $resolution_json,
    "other_copies_of_these_filenames_on_the_filesystem": $duplicates
  }
}
JSON

  # toolchain tuple, measured inside the image
  local tool_json
  tool_json=$(podman run --rm -i --network none \
    --entrypoint /opt/venv/bin/python "$IMAGE_TAG" - <<'PY'
import json, os, platform, sys
row = {"python": sys.version.split()[0], "cuda_runtime_env": os.environ.get("CUDA_VERSION"),
       "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
       "local_inference_cache_fingerprint":
           os.environ.get("LOCAL_INFERENCE_CACHE_FINGERPRINT"),
       "libc_platform": platform.platform()}
try:
    import torch
    row["torch"] = torch.__version__
    row["torch_cuda"] = torch.version.cuda
    row["torch_cudnn"] = str(torch.backends.cudnn.version())
except Exception as exc:
    row["torch_error"] = f"{type(exc).__name__}: {exc}"
try:
    from importlib.metadata import version
    row["vllm"] = version("vllm")
except Exception as exc:
    row["vllm_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(row))
PY
)
  local host_driver host_cuda host_gpu
  host_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unavailable)
  host_cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA UMD Version: *\([0-9.]*\).*/\1/p' | head -1)
  host_cuda=${host_cuda:-unavailable}
  host_gpu=$(nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo unavailable)
  json_fragment toolchain <<JSON
{
  "toolchain": {
    "in_image": $tool_json,
    "host_driver_version": "$host_driver",
    "host_cuda_umd_version": "$host_cuda",
    "host_gpu": "$host_gpu",
    "os_release": "$(podman run --rm --network none --entrypoint /bin/bash "$IMAGE_TAG" -lc '. /etc/os-release && echo "$PRETTY_NAME"')"
  },
  "privilege": {
    "podman_rootless": $(podman info --format '{{.Host.Security.Rootless}}'),
    "invoking_host_uid": $(id -u),
    "image_config_user": "$(podman image inspect "$IMAGE_TAG" --format '{{.Config.User}}')",
    "note": "the image declares no USER, so processes are uid 0 inside the container's user namespace; rootless podman maps that to host uid $(id -u), so no host-root privilege is held"
  }
}
JSON
}

# ---------------------------------------------------------------------------- sbom
stage_sbom() {
  local sbom_source tool_version pip_file dpkg_file
  pip_file=$REPO_ROOT/receipts/production-image$SBOM_INFIX-sbom-pip.txt
  dpkg_file=$REPO_ROOT/receipts/production-image$SBOM_INFIX-sbom-dpkg.txt
  if command -v syft >/dev/null 2>&1; then
    sbom_source=syft
    tool_version=$(syft version | awk '/Version:/{print $2}' | head -1)
    syft "podman:$IMAGE_TAG" -o spdx-json \
      > "$REPO_ROOT/receipts/production-image$SBOM_INFIX-sbom.spdx.json"
  else
    sbom_source="pip-freeze+dpkg (syft not installed on the build host)"
    tool_version="none"
    podman run --rm --network none --entrypoint /opt/venv/bin/python "$IMAGE_TAG" \
      -m pip freeze --all > "$pip_file"
    podman run --rm --network none --entrypoint /bin/bash "$IMAGE_TAG" \
      -lc "dpkg-query -W -f='\${Package}\t\${Version}\t\${Architecture}\n' | sort" > "$dpkg_file"
  fi
  # A source-only overlay cannot change the package inventory. Prove it rather than assert
  # it: if a variant's lists are byte-identical to the release image's, drop the duplicate
  # files and point at the release ones.
  local identical=null release_pip release_dpkg
  release_pip=$REPO_ROOT/receipts/production-image-sbom-pip.txt
  release_dpkg=$REPO_ROOT/receipts/production-image-sbom-dpkg.txt
  if [[ -n $SBOM_INFIX && -f $release_pip && -f $release_dpkg ]]; then
    if cmp -s "$pip_file" "$release_pip" && cmp -s "$dpkg_file" "$release_dpkg"; then
      identical=true
      rm -f "$pip_file" "$dpkg_file"
      pip_file=$release_pip
      dpkg_file=$release_dpkg
    else
      identical=false
    fi
  fi
  json_fragment sbom <<JSON
{
  "sbom": {
    "source": "$sbom_source",
    "tool_version": "$tool_version",
    "captured_from": "$IMAGE_TAG",
    "python_packages": $( [[ -f $pip_file ]] && wc -l < "$pip_file" || echo null ),
    "python_packages_file": "receipts/$(basename "$pip_file")",
    "python_packages_sha256": "$( [[ -f $pip_file ]] && digest_of "$pip_file" || echo null )",
    "debian_packages": $( [[ -f $dpkg_file ]] && wc -l < "$dpkg_file" || echo null ),
    "debian_packages_file": "receipts/$(basename "$dpkg_file")",
    "debian_packages_sha256": "$( [[ -f $dpkg_file ]] && digest_of "$dpkg_file" || echo null )",
    "package_lists_identical_to_release_image": $identical
  }
}
JSON
}

# --------------------------------------------------------------------------- smoke
# Card recipe arguments, verbatim from the model cards with the three -v source mounts
# deleted (they are inside the image now). `--host 0.0.0.0 --port 8000` is the cards'
# wording: the socket is namespace-internal and the *published* endpoint is loopback only.
recipe_args() {
  local model_path=$2
  local ignore='["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*in_proj_ba$","re:.*mtp\\..*","lm_head"]'
  local k4_ignore='["re:.*visual\\..*","re:.*in_proj_a$","re:.*in_proj_b$","re:.*mtp\\..*","lm_head"]'
  # Prefix caching is on in the published bounded profiles. The native context profile is
  # deliberately prefix-cache-off: the qualification receipt records refusal/deadlock/livelock
  # at this window when align mode is enabled. RECIPE_PREFIX_CACHING=0 keeps every profile off.
  local apc=(--enable-prefix-caching --mamba-cache-mode align)
  if [[ $1 == context || ${RECIPE_PREFIX_CACHING:-1} == 0 ]]; then
    apc=(--no-enable-prefix-caching)
  fi
  case $1 in
    k4) printf '%s\n' serve "$model_path" \
        --served-model-name qwen38-k4 --quantization exl3 --enforce-eager \
        --quantization-config "{\"linear\":{\"weight\":\"mxfp8\"},\"ignore\":$k4_ignore}" \
        --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 4 \
        "${apc[@]}" \
        --host 0.0.0.0 --port 8000 ;;
    k5k6) printf '%s\n' serve "$model_path" \
        --served-model-name qwen38 --quantization exl3 \
        --quantization-config "{\"linear\":{\"weight\":\"mxfp8\"},\"ignore\":$ignore}" \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
        --mm-processor-kwargs '{"truncation":false}' \
        --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 8 \
        "${apc[@]}" \
        --host 0.0.0.0 --port 8000 ;;
    hydrated) printf '%s\n' serve "$model_path" \
        --served-model-name qwen38 --quantization exl3 --enforce-eager \
        --quantization-config "{\"linear\":{\"weight\":\"mxfp8\"},\"ignore\":$ignore}" \
        --mm-processor-kwargs '{"truncation":false}' \
        --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        --max-model-len 8192 --gpu-memory-utilization 0.95 --max-num-seqs 8 \
        "${apc[@]}" \
        --host 0.0.0.0 --port 8000 ;;
    context) printf '%s\n' serve "$model_path" \
        --served-model-name qwen38 --quantization exl3 \
        --quantization-config "{\"linear\":{\"weight\":\"mxfp8\"},\"ignore\":$ignore}" \
        --max-model-len "${CONTEXT_MAX_MODEL_LEN:-262144}" \
        --gpu-memory-utilization "${CONTEXT_GPU_UTIL:-0.955}" --max-num-seqs 1 \
        --kv-cache-dtype fp8 --max-num-batched-tokens 2048 \
        --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
        --mm-processor-kwargs '{"truncation":false,"max_pixels":8388608}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}' \
        --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        "${apc[@]}" \
        --host 0.0.0.0 --port 8000 ;;
    *) die "unknown recipe $1" ;;
  esac
}

recipe_repo() {
  case $1 in
    k4) echo models--malaiwah--Qwen3.8-27B-K4 ;;
    k5k6) echo models--malaiwah--Qwen3.8-27B-EXL3-K5K6 ;;
    hydrated) echo models--malaiwah--Qwen3.8-27B-EXL3-K5K6-hydrated ;;
    context) echo models--malaiwah--Qwen3.8-27B-EXL3-K5K6-context ;;
  esac
}

recipe_env() {
  case $1 in
    k4) printf '%s\n' -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \
          -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online \
          -e VLLM_EXL3_ONLINE_CACHE_MODE=readwrite ;;
    k5k6) printf '%s\n' -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \
          -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online \
          -e VLLM_EXL3_GRAPH_DECODE=1 -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 ;;
    hydrated) : ;;
    # expandable_segments is NOT baked into the image env; the live launcher passes it, and
    # the native-context profile needs it (plus the qualified 0.955 utilisation)
    # to survive a multi-megapixel image on a physical 5090. Override with CONTEXT_GPU_UTIL
    # and CONTEXT_ALLOC_CONF; the exact argv used is recorded verbatim in the receipt.
    context) printf '%s\n' -e VLLM_EXL3_EMBED_BITS=8 -e VLLM_EXL3_GRAPH_DECODE=1 \
          -e VLLM_EXL3_PREFILL_RECONSTRUCT_M=128 \
          -e "PYTORCH_CUDA_ALLOC_CONF=${CONTEXT_ALLOC_CONF:-expandable_segments:True}" ;;
  esac
}

# Resolve a recipe into the exact podman argv. RUN_COMMAND, MODEL_PATH, SNAPSHOT and
# MODEL_REVISION_SOURCE are globals so `plan` (dry run, no GPU) and `smoke` cannot drift
# apart. Revision comes from the Hugging Face cache's own refs/main when the cache has
# one; a bare snapshot directory falls back to the most recently written revision, which
# is recorded as such rather than silently presented as "the" revision.
#
# This host's cache holds some repositories at the root and some under hub/, so both are
# searched. The *repository* directory is mounted read-only, never the snapshot directory:
# snapshot entries are relative symlinks into ../../blobs and would dangle.
build_run_command() {
  local recipe=$1 name=$2 repo ref candidate
  repo=$(recipe_repo "$recipe")
  REPO_DIR=""
  for candidate in "$MODELS_ROOT/$repo" "$MODELS_ROOT/hub/$repo"; do
    [[ -d $candidate/snapshots ]] && { REPO_DIR=$candidate; break; }
  done
  [[ -n $REPO_DIR ]] \
    || { RUN_COMMAND=(); MODEL_PATH=""; MODEL_REVISION=""; SNAPSHOT=""; return 1; }
  ref=$REPO_DIR/refs/main
  if [[ -f $ref ]]; then
    SNAPSHOT=$REPO_DIR/snapshots/$(cat "$ref")
    MODEL_REVISION_SOURCE=refs/main
  else
    SNAPSHOT=$(find "$REPO_DIR/snapshots" -maxdepth 1 -mindepth 1 -type d \
      -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2- || true)
    MODEL_REVISION_SOURCE="newest snapshot directory (cache has no refs/main)"
  fi
  [[ -n $SNAPSHOT && -d $SNAPSHOT ]] \
    || { RUN_COMMAND=(); MODEL_PATH=""; MODEL_REVISION=""; return 1; }
  MODEL_REVISION=$(basename "$SNAPSHOT")
  MODEL_PATH=/models/$repo/${SNAPSHOT#"$REPO_DIR/"}
  local serve_args env_args
  mapfile -t serve_args < <(recipe_args "$recipe" "$MODEL_PATH")
  mapfile -t env_args < <(recipe_env "$recipe")
  RUN_COMMAND=(podman run --rm --name "$name" --detach
    --device nvidia.com/gpu=all --ipc host --security-opt label=disable
    --publish "127.0.0.1:$SMOKE_PORT:8000"
    --volume "$REPO_DIR:/models/$repo:ro")
  if [[ -d ${CACHE_JIT_DIR:-} && -d ${CACHE_EXL3_DIR:-} ]]; then
    RUN_COMMAND+=(--volume "$CACHE_JIT_DIR:/cache/jit:rw"
                  --volume "$CACHE_EXL3_DIR:/cache/exl3-online:rw")
    CACHE_MOUNT_DESC="warm host caches $CACHE_JIT_DIR -> /cache/jit and $CACHE_EXL3_DIR -> /cache/exl3-online, the only writable mounts added"
  else
    RUN_COMMAND+=(--volume "$CACHE_VOLUME:/cache:rw")
    CACHE_MOUNT_DESC="named podman volume $CACHE_VOLUME -> /cache, the only writable mount added"
  fi
  RUN_COMMAND+=(-e HF_HUB_OFFLINE=1)
  [[ ${SMOKE_READ_ONLY:-0} == 1 ]] && RUN_COMMAND+=(--read-only --read-only-tmpfs)
  [[ ${#env_args[@]} -gt 0 ]] && RUN_COMMAND+=("${env_args[@]}")
  RUN_COMMAND+=(--entrypoint /opt/venv/bin/vllm "$IMAGE_TAG" "${serve_args[@]}")
  return 0
}

stage_plan() {
  local recipe
  for recipe in "$@"; do
    if build_run_command "$recipe" "smoke-$recipe-DRYRUN"; then
      printf '### %s\n%s\n\n' "$recipe" "$(printf '%q ' "${RUN_COMMAND[@]}")"
    else
      printf '### %s\nSKIP: no weights at %s or %s\n\n' "$recipe" \
        "$MODELS_ROOT/$(recipe_repo "$recipe")" "$MODELS_ROOT/hub/$(recipe_repo "$recipe")"
    fi
  done
}

stage_smoke() {
  local recipe results="" name
  mkdir -p "$VARIANT_STAGE/smoke"
  podman volume inspect "$CACHE_VOLUME" >/dev/null 2>&1 || podman volume create "$CACHE_VOLUME" >/dev/null
  for recipe in "$@"; do
    name=smoke-$recipe-$$
    if ! build_run_command "$recipe" "$name"; then
      log "SKIP $recipe: no snapshot at $MODELS_ROOT{,/hub}/$(recipe_repo "$recipe")"
      results+="{\"recipe\":\"$recipe\",\"ran\":false,\"skipped_because\":\"weights absent at $MODELS_ROOT/$(recipe_repo "$recipe") and $MODELS_ROOT/hub/$(recipe_repo "$recipe")\"},"
      continue
    fi
    local model_path=$MODEL_PATH run_command=("${RUN_COMMAND[@]}") run_command_str
    run_command_str=$(printf '%q ' "${run_command[@]}")

    log "starting recipe $recipe"
    local started ready=false container_id="" launch_err=$VARIANT_STAGE/smoke/$recipe-launch.err
    started=$(date +%s)
    # A failed launch is evidence, not a reason to abandon the remaining recipes.
    if ! container_id=$("${run_command[@]}" 2>"$launch_err"); then
      container_id=""
      log "recipe $recipe failed to launch: $(tr '\n' ' ' <"$launch_err" | cut -c1-200)"
    fi
    local deadline=$(( started + ${SMOKE_START_TIMEOUT:-1800} ))
    while [[ -n $container_id ]] && (( $(date +%s) < deadline )); do
      if curl -fsS "http://127.0.0.1:$SMOKE_PORT/health" >/dev/null 2>&1; then ready=true; break; fi
      podman container exists "$name" || break
      sleep 5
    done
    local start_seconds=$(( $(date +%s) - started ))
    local smoke_out=$VARIANT_STAGE/smoke/$recipe.json smoke_rc=1
    local diff_json='{"total_paths_changed":null,"under_opt":null,"under_usr_lib_or_bin":null}'
    if $ready; then
      log "recipe $recipe healthy after ${start_seconds}s; requesting"
      python3 "$REPO_ROOT/docker/smoke_client.py" --url "http://127.0.0.1:$SMOKE_PORT/v1" \
        --label "$recipe" --out "$smoke_out" >>"$VARIANT_STAGE/smoke/$recipe.log" 2>&1 && smoke_rc=0 || smoke_rc=$?
      # Proof that the running container installed nothing and copied nothing into the
      # runtime tree: podman diff lists every path changed relative to the image.
      diff_json=$(podman diff --format json "$name" \
        | python3 -c 'import json,sys
d=json.load(sys.stdin)
allp=sorted(p for k in ("changed","added","deleted") for p in (d.get(k) or []))
print(json.dumps({"total_paths_changed":len(allp),
                  "under_opt":[p for p in allp if p.startswith("/opt")],
                  "under_usr_lib_or_bin":[p for p in allp
                                          if p.startswith(("/usr","/bin","/lib","/sbin"))]}))')
    else
      log "recipe $recipe FAILED to become healthy in ${start_seconds}s"
    fi
    podman logs "$name" > "$VARIANT_STAGE/smoke/$recipe-server.log" 2>&1 || true
    podman stop --time 30 "$name" >/dev/null 2>&1 || true
    podman rm --force "$name" >/dev/null 2>&1 || true

    cp -f "$VARIANT_STAGE/smoke/$recipe-server.log" \
      "$REPO_ROOT/receipts/production-image$SBOM_INFIX-$recipe-server.log" 2>/dev/null || true
    results+=$(python3 - "$recipe" "$ready" "$start_seconds" "$smoke_rc" "$model_path" \
        "$diff_json" "$container_id" "$VARIANT_STAGE/smoke/$recipe-server.log" "$run_command_str" \
        "${SMOKE_READ_ONLY:-0}" "$MODEL_REVISION" "$MODEL_REVISION_SOURCE" "$REPO_DIR" \
        "$launch_err" "$CACHE_MOUNT_DESC" "$IMAGE_TAG" \
        "$(podman image inspect "$IMAGE_TAG" --format '{{.Digest}}')" \
        "receipts/production-image$SBOM_INFIX-$recipe-server.log" <<'PY'
import json, sys, hashlib, os, re
(recipe, ready, seconds, rc, model_path, diff_json, cid, logpath, run_command,
 read_only, revision, revision_source, weights_host_path, launch_err,
 cache_mount, image_tag, image_digest, log_file) = sys.argv[1:19]
smoke = None
launch_error = ""
if os.path.exists(launch_err):
    launch_error = open(launch_err, encoding="utf-8", errors="replace").read().strip()[:400]
stage_out = os.path.join(os.path.dirname(logpath), recipe + ".json")
if os.path.exists(stage_out):
    smoke = json.load(open(stage_out))
changed = json.loads(diff_json)
# The engine's own banner, not our assertion, is what records the served configuration.
# These patterns were checked against the archived native-MTP server log in receipts/.
banner = {}
if os.path.exists(logpath):
    text = open(logpath, encoding="utf-8", errors="replace").read()
    for key, pattern in (
            ("enable_prefix_caching", r"enable_prefix_caching=(\w+)"),
            ("max_model_len", r"'max_model_len':\s*(\d+)"),
            ("kv_cache_dtype_requested", r"'kv_cache_dtype':\s*'([\w.]+)'"),
            ("kv_cache_dtype_resolved", r"kv_cache_dtype=(torch\.[\w]+)"),
            ("gpu_kv_cache_tokens", r"GPU KV cache size: ([\d,]+) tokens"),
            ("available_kv_cache_gib", r"Available KV cache memory: ([\d.]+) GiB"),
            ("cudagraph_memory_gib", r"CUDA graph pool memory: ([\d.]+) GiB"),
            ("num_speculative_tokens", r"num_spec_tokens=(\d+)"),
            ("cudagraph_mode", r"cudagraph_mode=<?([\w.]+)"),
            ("max_concurrency_for_tokens", r"[Mm]aximum concurrency for ([\d,]+) tokens"),
    ):
        found = re.findall(pattern, text)
        banner[key] = found[-1] if found else None
row = {
    "recipe": recipe,
    "ran": True,
    "image_tag": image_tag,
    "image_manifest_digest": image_digest,
    "run_command": run_command.strip(),
    "model_path_in_container": model_path,
    "weights_host_path": weights_host_path,
    "model_revision": revision,
    "model_revision_resolved_from": revision_source,
    "weights_mount": "read-only",
    "cache_mount": cache_mount,
    "container_rootfs_read_only": read_only == "1",
    "source_bind_mounts": [],
    "published_port": "127.0.0.1:<port>:8000, loopback only; the listening socket is inside "
                      "the container network namespace and is not reachable off-host",
    "host_root_privilege": False,
    "rootless_podman": True,
    "started_healthy": ready == "true",
    "startup_seconds": int(seconds),
    "smoke_client_exit": int(rc),
    "engine_banner": banner,
    "container_filesystem_changes_vs_image": changed,
    "container_id": cid[:12],
    "launched": bool(cid),
    "launch_error": launch_error or None,
    "server_log_sha256": hashlib.sha256(open(logpath, "rb").read()).hexdigest()
        if os.path.exists(logpath) else None,
    "server_log_file": log_file,
    "smoke": smoke,
    "text_and_image_pass": bool(smoke and smoke.get("passed")),
}
print(json.dumps(row))
PY
),
  done
  # Merge into any earlier smoke fragment so a contended GPU can be used in several short
  # windows without losing the recipes already proven. A rerun of the same recipe wins.
  mkdir -p "$VARIANT_STAGE"
  python3 - "$VARIANT_STAGE/smoke.json" "[${results%,}]" <<'PY'
import json, os, sys
path, incoming = sys.argv[1:3]
rows = []
if os.path.exists(path):
    rows = json.load(open(path)).get("recipes", [])
fresh = json.loads(incoming)
names = {row["recipe"] for row in fresh}
merged = [row for row in rows if row["recipe"] not in names] + fresh
order = {"context": 0, "hydrated": 1, "k5k6": 2, "k4": 3}
merged.sort(key=lambda row: order.get(row["recipe"], 9))
with open(path, "w") as handle:
    json.dump({"recipes": merged}, handle, indent=2)
print(f"smoke fragment now holds: {[row['recipe'] for row in merged]}")
PY
}

# ------------------------------------------------------------------------- receipt
stage_receipt() {
  python3 - "$STAGE" "$RECEIPT" <<'PY'
import json, os, sys, time
stage, out = sys.argv[1:3]


def fragment(variant, name):
    path = os.path.join(stage, variant, name + ".json")
    return json.load(open(path)) if os.path.exists(path) else {}


def acceptance_for(verify, resolution, sbom, tool, build, recipes):
    """Every gate is either measured or null. Runtime gates need a recipe that served."""
    published_by_dest = {m["dest"]: m["published_sha256"]
                         for m in verify.get("patch_modules", [])}
    measured = resolution.get("import_resolution", {}).get("measured", {})
    resolved = measured.get("modules", {})
    immutability = verify.get("immutability", {})
    ran = [r for r in recipes if r.get("ran")]
    healthy = [r for r in ran if r.get("started_healthy")]
    return {
        "modules_in_image_match_published_map":
            immutability.get("python_files_changed_are_exactly_the_patch_set")
            and all(m["matches_published_map"] for m in verify.get("patch_modules", []))
            if verify else None,
        "layer_adds_exactly_its_declared_files":
            immutability.get("python_files_changed_vs_parent_are_exactly_this_layers_adds")
            if verify else None,
        "import_machinery_resolves_to_patched_modules": (
            all(m.get("resolved_origin") in published_by_dest
                and m.get("resolved_origin_sha256")
                    == published_by_dest[m["resolved_origin"]]
                for m in resolved.values())
            and len(resolved) == len(published_by_dest))
            if resolved else None,
        "no_stale_bytecode_for_patched_modules":
            all(m.get("stale_bytecode_present") is False for m in resolved.values())
            if resolved else None,
        "only_the_imported_tree_is_patched":
            measured.get("second_source_tree_on_sys_path") is False
            if measured else None,
        "duplicate_source_tree_unreachable_from_working_dir": (
            measured.get("second_source_tree_reachable_from_cwd") is False
            and immutability.get("image_config_working_dir_is_the_duplicate_source_tree")
                is False)
            if measured and immutability else None,
        "vllm_is_not_an_editable_install": (
            measured.get("vllm_installed_editable") is False
            and measured.get("editable_install_pth_present") is False)
            if measured else None,
        "sbom_generated": bool(sbom.get("sbom")) if sbom else False,
        "no_runtime_source_bind_mounts":
            all(not r.get("source_bind_mounts") for r in healthy) if healthy else None,
        "no_startup_file_copies": all(
            not (r.get("container_filesystem_changes_vs_image") or {}).get("under_opt")
            and not (r.get("container_filesystem_changes_vs_image") or {}).get(
                "under_usr_lib_or_bin")
            for r in healthy) if healthy else None,
        "no_runtime_package_installs":
            immutability.get("runtime_package_installs") is False if verify else None,
        "weights_read_only":
            all(r["weights_mount"] == "read-only" for r in healthy) if healthy else None,
        "service_loopback_or_api_key":
            all("127.0.0.1" in r["published_port"] for r in healthy) if healthy else None,
        "no_host_root_privilege":
            all(r["host_root_privilege"] is False for r in healthy) if healthy else None,
        "image_digest_and_toolchain_recorded":
            bool(build.get("image", {}).get("manifest_digest")) and bool(tool.get("toolchain")),
    }


def smoke_section(recipes):
    ran = [r for r in recipes if r.get("ran")]
    return {
        "client": "docker/smoke_client.py, one text and one image request per recipe, "
                  "exact match; image fixture drawn by tools/vision_eval.draw_digits",
        "recipes_attempted": [r["recipe"] for r in recipes],
        "recipes_run": [r["recipe"] for r in ran],
        "recipes_skipped": {r["recipe"]: r.get("skipped_because")
                            for r in recipes if not r.get("ran")},
        "all_run_recipes_started": bool(ran) and all(r["started_healthy"] for r in ran),
        "all_run_recipes_text_and_image_exact":
            bool(ran) and all(r["text_and_image_pass"] for r in ran),
        "results": recipes,
    }


def caveats(recipes, acceptance):
    ran = [r for r in recipes if r.get("ran")]
    healthy = [r for r in ran if r.get("started_healthy")]
    notes = []
    if not healthy:
        notes.append("no recipe served a request, so every runtime gate (mounts, writes, "
                     "endpoint) is recorded as null rather than passed")
    missing = [r["recipe"] for r in recipes if not r.get("ran")]
    if missing:
        notes.append("recipes with no attempt recorded: " + ", ".join(missing))
    failed = [r["recipe"] for r in ran if not r.get("started_healthy")]
    if failed:
        notes.append("recipes attempted but not healthy within the start timeout: "
                     + ", ".join(failed))
    return notes


release = {name: fragment("release", name) for name in
           ("build", "verify", "resolution", "sbom", "toolchain", "smoke")}
if not release["build"]:
    raise SystemExit("no release build fragment: run `build` first")
release_recipes = release["smoke"].get("recipes", [])

payload = {
    "schema": "qwen38-production-image/3",
    "schema_change_from_2": "release_unit was a string naming the top-level image; it is now "
                            "an object, because the release unit and the top-level build "
                            "record are no longer the same image. A reader that treated "
                            "release_unit as text must be updated rather than silently "
                            "reinterpreting it.",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "purpose": "P0 / rank 2 immutable production runtime: the content-pinned patch modules "
               "folded into the pinned r34 digest, replacing the read-only source bind "
               "mounts the model cards' patched recipes required",
    "release_unit": {
        "tag": "localhost/vllm:gg-r34-patched-apc",
        "manifest_digest": "sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218",
        "modules": 4,
        "described_under": "superset_images, variant \"apc\"",
        "qualified_by": "receipts/qualification-5090-apc.json",
        "promoted_utc_date": "2026-08-16",
        "superseded_release_unit": {
            "tag": "localhost/vllm:gg-r34-patched",
            "manifest_digest": "sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27",
            "superseded_at": "2026-08-16, on Main's call, recorded in promotion_status",
            "still_valid_as": "the promoted image's parent, the build record at the top "
                              "level of this receipt, and the digest "
                              "receipts/qualification-5090-context.json measured. Superseded "
                              "as the recommended runtime, not invalidated as evidence.",
        },
        "why_the_top_level_still_describes_the_parent": "the top-level block is the build "
            "record of the three-module image and is pinned by other receipts. Moving it "
            "would break those pins to say something this pointer says without breaking "
            "anything. Any image under superset_images other than the one named here is "
            "explicitly not the qualified artifact.",
    },
    "upstream_status": "the patch modules are carried pending upstream acceptance, not as a "
                       "permanent fork: local-inference-lab/vllm issue #392 and PR #393 are "
                       "open, and PR #51113 is already merged upstream (vllm-project/vllm) "
                       "but absent from the r34 build. When a base image ships them, the "
                       "corresponding COPY and its digest come out of the Dockerfile.",
    "hardware_scope": "built and smoke-tested on AIBoss, one physical RTX 5090 (32,607 MiB); "
                      "no throughput number here is comparable to the rental RTX PRO 6000 "
                      "engine-budget receipts",
    "operational_scope": "the host's live qwen38-27b service is stopped for, and restored "
                         "after, any GPU window on this single-card host; the release "
                         "image's own smoke recipes ran while it was up, on a distinct "
                         "loopback port, and the promoted image's recipes ran inside the "
                         "promotion window with the unit down and restored against "
                         "receipts/aiboss-live-service-snapshot.json afterwards. No recipe "
                         "ever altered or reconfigured that unit.",
    **release["build"],
    **release["verify"],
    **release["resolution"],
    **release["toolchain"],
    **release["sbom"],
    "smoke": smoke_section(release_recipes),
}
payload["acceptance"] = acceptance_for(release["verify"], release["resolution"],
                                       release["sbom"], release["toolchain"],
                                       release["build"], release_recipes)
payload["not_claimed"] = [
    "no image here has been pushed to any public registry: repo_digests are local, and the "
    "release unit is the local manifest digest plus the verified source map",
    "the model cards do not name this local tag as something to pull. They reproduce the "
    "promoted image's content with four sha256-verified read-only mounts over the pullable "
    "public base digest, which is the closest a reader without access to this build host "
    "can get; the local manifest digest is printed beside it so the two are relatable.",
] + caveats(release_recipes, payload["acceptance"])

payload["notes_for_the_next_image_build"] = {
    "a_boolean_qualified_label_is_unfixable_by_construction": {
        "what_happened": "Dockerfile.gg-r34-patched-apc carries "
                         "io.malaiwah.image.qualified=\"false\" and an "
                         "io.malaiwah.image.role saying the image requires its own "
                         "qualification run. Both were true when written. Both became false "
                         "the moment receipts/qualification-5090-apc.json passed, and neither "
                         "can be corrected.",
        "why_it_cannot_be_fixed": "qualification necessarily happens after the bytes exist. "
                                  "Editing a label is a new layer, a new layer is a new "
                                  "manifest digest, and the corrected image is therefore no "
                                  "longer the image that was qualified. The label can only "
                                  "ever be right about a state that had not happened yet.",
        "what_to_do_instead": "carry a pointer, not a verdict. A label naming the receipt "
                              "path or a qualification URL stays true across promotion because "
                              "it asserts where to look rather than what the answer is. Or "
                              "carry no qualification label at all and track qualification "
                              "only in receipts, which is what this receipt does.",
        "precedence_rule_in_force_meanwhile": "receipts/production-image.json is authoritative "
                                              "for qualification state; io.malaiwah.image.* is "
                                              "a build-time statement only. A reader who "
                                              "trusts the label over the receipt reaches the "
                                              "exact inversion of the truth.",
        "recorded_because": "Main asked for the lesson to sit where whoever writes the next "
                            "Dockerfile will see it, rather than in a thread nobody reads.",
    },
}

supersets = []
for variant in ("apc", "convert"):
    frags = {name: fragment(variant, name) for name in
             ("build", "verify", "resolution", "sbom", "toolchain", "smoke")}
    if not frags["build"]:
        continue
    recipes = frags["smoke"].get("recipes", [])
    entry = {
        "variant": variant,
        **frags["build"],
        **frags["verify"],
        **frags["resolution"],
        **frags["sbom"],
        "smoke": smoke_section(recipes),
    }
    entry["acceptance"] = acceptance_for(frags["verify"], frags["resolution"], frags["sbom"],
                                         frags["toolchain"] or release["toolchain"],
                                         frags["build"], recipes)
    # These lines are per-variant on purpose. They used to be a shared prefix, which was
    # safe only while every superset was unqualified; the apc image is the promoted release
    # unit now, and asserting the shared wording over it would say the opposite of the truth
    # while asserting the apc wording over convert would claim the conversion image is
    # qualified. Neither is acceptable, so each variant states its own.
    entry["not_claimed"] = ({
        "apc": ["this is the promoted release unit as of 2026-08-16. It is qualified with "
                "prefix caching OFF by inheritance from receipts/qualification-5090-context.json, "
                "because the module it adds is unreachable in that mode, and with prefix "
                "caching ON only at the 8,192-token recipes, on this receipt's own smoke. "
                "The top-level block of this receipt is its parent, the "
                "three-module image, kept as the build record and as the digest "
                "receipts/qualification-5090-context.json measured; it is superseded as the "
                "recommended runtime, not invalidated as evidence",
                "the in-image labels io.malaiwah.image.qualified=\"false\" and "
                "io.malaiwah.image.role are stale as of the promotion and were deliberately "
                "not corrected: relabelling is a layer and a layer moves the manifest "
                "digest away from the bytes that were measured. This receipt, not the "
                "label, is authoritative for qualification state",
                "prefix caching at the native 262,144 window is NOT qualified on this card and is "
                "not enabled in the context edition's recipe: it refuses to start at "
                "utilisation 0.955, deadlocks at 0.9555 and livelocks at 0.9585 "
                "(receipts/qualification-5090-apc.json). The concurrent-serving variant the "
                "cards print is not qualified at any sequence count either",
                "the reuse win (11.6x and 29.3x warm TTFT) is receipts/apc-poison-repro.json "
                "arm E's, measured on this digest but on a document-reuse probe rather than "
                "on these serving gates",
                "LMCache is unmeasured by this project and is the outstanding suspect in the "
                "one user report of prefix-cache corruption we have. Nothing here says it is "
                "safe"],
        "convert": ["this is not the hardware-qualified digest; the promoted release unit "
                    "recorded in release_unit is",
                    "no published measurement depends on what this image adds",
                    "this tag is for conversion and quantization only and has not been "
                    "served or benchmarked; it exists so the missing import is fixed in an "
                    "image rather than shimmed per job",
                    "it deliberately does not change the serving digest, so no serving "
                    "receipt or qualification needs re-pinning because of it",
                    "no_runtime_package_installs is false here BY DESIGN and is not a "
                    "defect: this variant's whole purpose is one build-time pip install, "
                    "of a locally fetched wheel whose sha256 is verified before install "
                    "and with --no-deps so no inventoried package can be resolved or "
                    "upgraded. Nothing is installed at container start."],
    }.get(variant, ["this is not the hardware-qualified digest; the promoted release unit "
                    "recorded in release_unit is",
                    "no published measurement depends on what this image adds"])
    ) + caveats(recipes, entry["acceptance"])
    supersets.append(entry)
if supersets:
    payload["superset_images"] = supersets
payload["promotion_status"] = {
    "state": "PROMOTED, with prefix caching scoped",
    "promoted_utc_date": "2026-08-16",
    "release_unit_before": {
        "tag": "localhost/vllm:gg-r34-patched",
        "manifest_digest": "sha256:6eca4c693f01b6f4e112c04eacd30673b7cfbba4150e6fe2ea3ba1bbfde14c27",
        "modules": 3,
        "qualified_by": "receipts/qualification-5090-context.json, seven gates with "
                        "enable_prefix_caching=False",
    },
    "release_unit_after": {
        "tag": "localhost/vllm:gg-r34-patched-apc",
        "manifest_digest": "sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218",
        "modules": 4,
        "qualified_by": "with prefix caching off, receipts/qualification-5090-context.json "
                        "by inheritance -- the added module's changed function is "
                        "unreachable unless mamba_cache_mode is align. With prefix caching "
                        "on, receipts/qualification-5090-apc.json plus this receipt's own "
                        "four-recipe smoke on this digest.",
        "prefix_caching_scope": "enabled in the k4, k5k6 and hydrated recipes; NOT enabled "
                                "in the context edition's native 262,144 recipe, which "
                                "refuses to start at utilisation 0.955, deadlocks at 0.9555 "
                                "and livelocks at 0.9585. Align mode rounds a request to "
                                "whole 1,600-token blocks and adds MTP draft slots and a "
                                "decode block on top, and on a 32,607 MiB card that does not "
                                "coexist with the 8,388,608-pixel ceiling gate 3 needs "
                                "headroom for.",
        "adds": "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py at "
                "b431c1066dfee3ed56bfa7e71cc8606f9afadc300f22d7fc542c43835d1b22bf "
                "(upstream vllm-project/vllm#51113)",
    },
    "why_promoted": "prefix caching is the largest untapped win for repeated long prompts on "
                    "this model and it cannot be turned on safely without #51113: unpatched, "
                    "a prefill chunk ending mid-block publishes a short state and later "
                    "requests over a shared prefix consume it, returning wrong tokens with "
                    "HTTP 200. Promotion is what makes enabling it possible at all; where it "
                    "is actually enabled is a per-recipe question answered by measurement, "
                    "and the answer came back three yes and one no.",
    "not_rebuilt": "the promoted digest is byte-identical to the one built at "
                   "2026-08-16T03:16:01Z. No layer was added, no label edited, no tag "
                   "re-pointed. Rebuilding would have moved the digest away from the bytes "
                   "arms C and E of receipts/apc-poison-repro.json measured, which is the "
                   "only reason this digest rather than a fresh one was promoted.",
    "decision_record": {
        "ruled_by": "Main, 2026-08-16",
        "what_was_ruled": "promote the four-module superset exactly as it stands, and do not "
                          "bake tools/vllm-qwen-gdn-spec-gates.py (upstream #51812) into it",
        "reached_independently_first_by": "ImmutableImage and ApcPoisonRepro, who each "
                                          "declined the GDN coupling on the same evidential "
                                          "ground before the ruling",
        "earlier_brief_superseded": "an earlier instruction to promote a five-module image "
                                    "was withdrawn: no arm measured a five-module digest, so "
                                    "such an image would have carried an in-image-unmeasured "
                                    "module and its qualification would have stood alone.",
    },
    "evidence": {
        "qualification": {
            "receipt": "receipts/qualification-5090-apc.json",
            "owner": "ShipPrefixCaching; authoritative for the nine gates, the KV capacity "
                     "comparison and the promoted image's serving behaviour",
            "what_it_adds": "the in-image, on-hardware serving evidence for this digest with "
                            "prefix caching on: the original seven gates unchanged, plus a "
                            "banner gate proving the engine honoured the flags and a KV "
                            "capacity gate against the 265,122-token baseline",
        },
        "correctness_and_reuse": {
            "receipt": "receipts/apc-poison-repro.json",
            "receipt_file_sha256": "1baeb5aa5373257563c234179c3cb75fac6c77d968fa38080b3636fa1435a36e",
            "receipt_internal_content_sha256": "d9c1703a844f41cd4d7821fd9336c8920ecbe8facb96c2c03624218cdd241cd7",
            "supersedes_earlier_pin": "ed11552d4fcf1c1d317b865ac40d78ff2aaad4dcf07a9af7ad6773bcfdd3d8e3, "
                                      "re-pinned after its author added discussion, live-service "
                                      "restore evidence and per-arm digest provenance; the author "
                                      "states that was the final edit",
            "owner": "ApcPoisonRepro; authoritative for the prefix-caching correctness and TTFT "
                     "numbers, which are deliberately not restated here",
            "which_image_each_arm_speaks_about": {
                "arms_C_and_E": "statements about the four-module superset "
                                "sha256:16a936b877b90fc080181e842f47dbafc5cb8e62688799596836e34ba0b79218 "
                                "with no bind mounts, i.e. evidence about an image, and "
                                "byte-identical to the digest promoted here",
                "arm_D": "that same digest plus a read-only bind mount of the GDN module, i.e. "
                         "evidence about the module and not about any image digest",
            },
        },
    },
    "candidate_modules_not_baked_into_any_image": [
        {
            "source": "tools/vllm-qwen-gdn-spec-gates.py",
            "sha256": "7cd3f5fe763b621048af4817951a841d99c8b700d9a56ded27ccaca5a56ccbe0",
            "dest": "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
            "upstream": "vllm-project/vllm#51812",
            "status": "final and drop-in: ApcPoisonRepro served 38/38 clean with it bind-mounted over the -apc image",
            "deliberately_not_baked": "it is not coupled to prefix caching. At --max-num-seqs 1 there is no mixed batch, the only condition #51812 bites in, so that run shows the module is harmless rather than that it fixes anything. Baking it would put a module in the release unit whose benefit has never been demonstrated and whose risk surface has never been qualified, and it would move the digest away from the one arms C and E measured.",
            "how_it_is_offered_instead": "a documented optional overlay for concurrent serving, mounted read-only over the promoted image, with its digest pinned here and on the four model cards. At one stream it is provably irrelevant; at concurrency it is unmeasured by this project and rests on upstream's own numbers.",
            "reconsider_when": "a concurrent-serving profile is qualified at --max-num-seqs > 1, which would create the mixed batch #51812 acts on and give the module something to be measured against.",
        }
    ],
}


temporary = out + ".tmp"
with open(temporary, "w") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
os.replace(temporary, out)
print(f"wrote {out}")
print(json.dumps({"release": payload["acceptance"],
                  "supersets": {s["variant"]: s["acceptance"] for s in supersets}}, indent=2))
PY
}

case ${1:-} in
  build) stage_build ;;
  verify) stage_verify ;;
  sbom) stage_sbom ;;
  plan) shift; [[ $# -gt 0 ]] || die "plan needs at least one recipe"; stage_plan "$@" ;;
  smoke) shift; [[ $# -gt 0 ]] || die "smoke needs at least one recipe"; stage_smoke "$@" ;;
  receipt) stage_receipt ;;
  *) die "usage: $0 {build|verify|sbom|plan <recipe...>|smoke <recipe...>|receipt}" ;;
esac
