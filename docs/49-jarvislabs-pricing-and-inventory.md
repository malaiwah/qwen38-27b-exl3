# JarvisLabs — API, inventory, pricing, availability

**Snapshot:** 2026-08-17 ~12:43–12:50 UTC. All API reads performed with the account API key supplied out-of-band (referred to below as `<JARVIS_API_KEY>`; the key itself appears nowhere in this file).
**Access mode:** strictly read-only. Only `GET` requests were issued. No instance was created, resized, paused, resumed or destroyed.

---

## 1. API discovery

### 1.1 What the docs say vs. what the client actually uses

| Claim | Source | Verdict |
|---|---|---|
| Base URL `https://api.jarvislabs.ai/v1` | third-party profile <https://github.com/api-evangelist/jarvislabs> (README, "Base URL"), echoed by web search | **Not usable.** DNS resolves (`103.19.91.140`) but TCP/443 and TCP/80 never complete from this host: `curl -m 15 https://api.jarvislabs.ai/` → `code=000 connect=0.000000 total=15.000866`; raw `/dev/tcp/103.19.91.140/443` → fail, `/80` → fail. That repo is an unaffiliated scrape ("**This is not our API**" in its own README) and its OpenAPI is not authoritative. |
| Per-region backends `https://backend{n,c,eu}.jarvislabs.net/` | official SDK `jarvislabs==0.2.17`, `jarvislabs/constants.py` → `REGION_URLS` (wheel from PyPI, <https://pypi.org/pypi/jarvislabs/json>) | **Correct and live.** All three answer. |

### 1.2 Base URLs actually used (retrieved)

Source: `jarvislabs/constants.py` in the `jarvislabs-0.2.17-py3-none-any.whl`:

| Region id | Display code | Base URL |
|---|---|---|
| `india-noida-01` (SDK default) | IN2 | `https://backendn.jarvislabs.net/` |
| `india-chennai-01` | IN1 | `https://backendc.jarvislabs.net/` |
| `europe-01` | EU1 | `https://backendeu.jarvislabs.net/` |

Serverless (separate hosts, `SERVERLESS_REGION_URLS`): `https://serverlessn.jarvislabs.net/`, `https://serverlessc.jarvislabs.net/`. There is no serverless host for `europe-01`.

There is **no `/v1` path prefix**; endpoints hang off the root (`GET /users/fetch`, not `/v1/users/fetch`).

### 1.3 Auth header form (retrieved, verified by experiment)

`Authorization: Bearer <JARVIS_API_KEY>` — from `jarvislabs/transport.py`:
`httpx.Client(headers={"Authorization": f"Bearer {token}"})`. Token also accepted from `JL_API_KEY` env var or `~/.config/jl/config.toml` by the CLI.

Observed behaviour of the auth layer:

| Request | Status | Body |
|---|---|---|
| `GET backendc/users/balance`, no `Authorization` header | `403` | `{"message":"Not authenticated"}` |
| `GET backendc/users/balance`, `Authorization: Bearer not-a-real-key` | `401` | `{"message":"Invalid token"}` |
| `GET backendc/users/balance`, `X-API-Key: <JARVIS_API_KEY>` only | `403` | `{"message":"Not authenticated"}` — **the key is not accepted in an `X-API-Key` header** |
| `GET backendc/users/balance`, `Authorization: Bearer <JARVIS_API_KEY>` | `200` | balance returned; exact account value redacted |

### 1.4 Every endpoint called, with observed status

All calls `GET`, all with `Authorization: Bearer <JARVIS_API_KEY>` unless noted.

| Endpoint | Host | Status | Notes |
|---|---|---|---|
| `/users/balance` | backendn, backendc | `200` | balance returned; account value redacted |
| `/users/user_info` | backendn | `200` | account identity returned; PII redacted |
| `/users/fetch` | backendn, backendc, backendeu | `200` | full instance list; **identical across all three hosts** — the list is account-global, not region-scoped |
| `/users/fetch/<production-id>` | backendc | `200` | single-instance detail |
| `/users/fetch/<driver-id>` | backendc | `200` | single-instance detail |
| `/users/fetch/<local-id>` | backendc | `200` | single-instance detail |
| `/misc/server_meta` | backendn, backendc, backendeu | `200` | the pricing + availability catalogue. backendn ≡ backendc byte-identical (md5 `c3d4ed31…`); backendeu differs (md5 `fd6eb505…`), see §3.4 |
| `/misc/resource_metrics` | backendn | `200` | `{"running_instances":1,"paused_instances":0,"running_vms":2,"paused_vms":0,"deployments":0,"filesystems":0}` |
| `/misc/frameworks` | backendn | `200` | 11 templates: `pytorch fastai automatic axolotl comfyui fooocus ollama ollama_serverless tensorflow vllm vm` |
| `/misc/status?machine_id=<production-id>` | backendc | `200` | `{"status":"Running","error":"None","code":"None"}` |
| `/misc/status` (no query) | backendn | `422` | `{"detail":[{"loc":["query","machine_id"],"msg":"field required","type":"value_error.missing"}]}` |
| `/misc/` | backendn | `200` | `{"success":false}` — SDK uses this as the INR-vs-USD flag; `false` ⇒ **billed in USD** |
| `/filesystem/list` | backendn | `200` | `[]` — no shared filesystems provisioned |
| `/vpc/list` | backendn | `200` | one active Chennai VPC; identifier and CIDR redacted |
| `/vpc/<redacted>/ports` | backendc | `200` | two attached ports; addresses and machine ids redacted |
| `/scripts/` | backendc | `200` | `{"success":true,"script_meta":[]}` |
| `/ssh/` | backendc | `200` | three registered public keys; labels redacted |
| `/management/list` | backendc | `404` | `{"detail":"Not Found"}` — wrong host; deployments live on the serverless hosts |
| `/management/list` | serverlessc | `200` | `{"deployments":[]}` |
| `https://api.jarvislabs.ai/v1{,/instances,/gpus,/gpu_types,/account,/user,/me,/templates,/filesystems,/status}` | api.jarvislabs.ai | **no response** | curl exit "000", TCP connect never completes (25 s timeout each). Host is not reachable from this workstation on 80 or 443. |

**Write endpoints deliberately NOT called** (documented here only so the read/write boundary is explicit): `POST /templates/{tpl}/create`, `POST /templates/vm/cpu/create`, `POST /templates/{tpl}/resume`, `POST /templates/vm/cpu/resume`, `PUT /machines/machine_name`, `POST|DELETE /filesystem/*`, `POST|DELETE /vpc/*`, `POST|DELETE /ssh/*`, `POST|DELETE /scripts/*`, `POST /management/create`, `DELETE /management/{id}`.

Public HTML sources also used: <https://jarvislabs.ai/pricing>, <https://docs.jarvislabs.ai/faqs/>, <https://docs.jarvislabs.ai/getting_started/>, <https://docs.jarvislabs.ai/filestorage/>, <https://docs.jarvislabs.ai/cli/>, <https://docs.jarvislabs.ai/sitemap.xml> — all `200`.

---

## 2. Our instances

Source: `GET https://backendn.jarvislabs.net/users/fetch` (and per-machine `GET /users/fetch/{id}`), sampled 12:43–12:50 UTC 2026-08-17.

| alias | role | GPU | n | vCPU | RAM | disk | status | region | uptime |
|---|---|---|---:|---:|---|---|---|---|---|
| **production** | 8-GPU serving host | RTX-PRO6000 (96 GB) | 8 | 224 | 1280 GB | 1000 GB ssd | **Running** | india-chennai-01 | 7 h 41 m |
| **driver** | benchmark load driver | CPU | 0 | 32 | 128 GB | 1000 GB ssd | **Running** | india-chennai-01 | 15 h 29 m |
| **local** | single-GPU research container | RTX-PRO6000 (96 GB) | 1 | 28 | 160 GB | 100 GB ssd | **Running** | india-chennai-01 | 3 d 9 h 21 m |

All three are `framework:"vm"`/`"pytorch"` on-demand (`is_spot: false`, `frequency: "hour"`, `committed_resource_id: null`, `reservation_info: null`). `paused_instances: 0`, `paused_vms: 0` — **nothing is paused; all three are RUNNING and billing.** No filesystems, no serverless deployments.

- **production** was busy serving load and was queried read-only.
- **driver** was the CPU-only load VM on the same VPC and was not modified.
- **local** was the single-GPU research container. Public/private addresses,
  machine identifiers, account identity and accrued line-item costs are
  deliberately omitted from this public research record.

### 2.1 Burn rate (measured)

Method: lifetime accrued `cost` ÷ reported `duration` from `users/fetch` at 12:50 UTC. (A 397 s delta-sample was also taken but is too noisy to publish — per-minute billing quantisation swings the 8-GPU line between $16.1 and $17.2/hr; the lifetime average is the sound measurement.)

| alias | measured $/hr | catalogue-expected $/hr | expected build-up |
|---|---:|---:|---|
| production | **15.293** | 15.260 | 8 × $1.89 GPU + 1000 GB × $0.00014 |
| driver | **0.935** | 0.934 | $0.7936 (32 vCPU/128 GB plan) + 1000 GB × $0.00014 |
| local | **1.904** | 1.904 | 1 × $1.89 GPU + 100 GB × $0.00014 |
| **TOTAL** | **$18.13 / hr** | $18.10 / hr | ≈ **$435 / day** |

At the snapshot, account runway was **about one hour** at the measured burn rate; the exact
balance is redacted. Per the FAQ, zero balance auto-pauses instances and may delete data.

The measured-vs-catalogue agreement (≤0.25 %) independently confirms three things: multi-GPU pricing is strictly linear, storage is billed **on top of** the GPU rate while running, and the published per-GPU rates are what the account is actually charged.

---

## 3. Pricing catalogue

Source: `GET /misc/server_meta` (`"currency":"USD"`), cross-checked against the public <https://jarvislabs.ai/pricing> page. Prices are **per GPU per hour**.

### 3.1 GPU SKUs

| GPU | VRAM | arch | vCPU/GPU | RAM/GPU | region | workload | on-demand $/GPU-hr | **spot $/GPU-hr** | 1 mo | 3 mo | 6 mo | 1 yr |
|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| **RTX-PRO6000** (Blackwell) | **96 GB** | Blackwell | 28 | 160 GB | india-chennai-01 | vm | **1.89** | **0.99** | 1.49 | 1.39 | 1.29 | 1.19 |
| **RTX-PRO6000** | 96 GB | Blackwell | 28 | 160 GB | india-chennai-01 | container | **1.89** | **0.99** | 1.49 | 1.39 | 1.29 | 1.19 |
| H200 SXM | 141 GB | Hopper | 28 | 300 GB | india-noida-01 | vm + container | **3.99** | **1.99** | 3.70 | 3.60 | 3.30 | 3.20 |
| H200 SXM | 141 GB | Hopper | 16 | 200 GB | europe-01 | (unspecified) | **3.99** / **3.80** † | *none* | — | — | — | — |
| H100 SXM | 80 GB | Hopper | 24 | 200 GB | india-noida-01 | vm + container | **2.69** | **1.19** | 2.49 | 2.39 | 2.19 | 1.99 |
| H100 SXM | 80 GB | Hopper | 16 | 200 GB | europe-01 | (unspecified) | **2.99** | *none* | — | — | — | — |
| A100-80GB | 80 GB | Ampere | 28 | 112 GB | india-noida-01 | vm + container | **1.49** | **0.89** | 1.42 | 1.34 | 1.27 | 1.25 |
| A100 (40 GB) | 40 GB | Ampere | 16 | 112 GB | india-noida-01 | container | **0.89** | **0.79** | 0.85 | 0.80 | 0.76 | 0.75 |
| L4 | 24 GB | Ada | 28 | 124 GB | india-noida-01 | vm + container | **0.44** | **0.29** | 0.42 | 0.40 | 0.37 | 0.35 |
| A30 | 24 GB | Ampere | 16 | 64 GB | india-noida-01 | container | **0.41** | **0.29** | 0.39 | 0.37 | 0.35 | 0.34 |

† See §3.4 — the EU backend and the India backends disagree on the europe-01 H200 rate.

**Only one 96 GB-class SKU exists on this provider: the RTX PRO 6000 Blackwell, and only in `india-chennai-01`.** If ≥96 GB is treated as the class, H200 (141 GB, Noida + EU) also qualifies; nothing else on the platform reaches 96 GB.

### 3.2 Spot — **YES, spot exists**

Retrieved, not inferred:
- `misc/server_meta` carries a `spot_price` field per SKU (values in the table above) plus `spot_num_free_devices`, `spot_only_server`.
- The official CLI/SDK 0.2.17 exposes it: `jl create --spot` ("Create a spot GPU container instance.", `jarvislabs/cli/instance.py:168`), `jl resume --spot`, `jl run --spot`, and the create payload carries `"is_spot": bool`.
- The public pricing page has a **Spot** plan tab labelled "save up to 56%" (SSR'd HTML, `id="plan-tab-spot"`). That 56 % matches H100 exactly: 1.19/2.69 = 55.8 % off. The reserved tab labels ("1 month · save up to 21 %", "3 months · 26 %", "6 months · 32 %", "1 year · 37 %") match RTX PRO 6000 exactly: 1.49/1.39/1.29/1.19 vs 1.89 → 21.2/26.5/31.7/37.0 % off. The public page and the private catalogue are consistent.

Hard constraints on spot (retrieved from SDK validation code):
- **Spot is GPU *containers* only.** `"Spot instances are only supported for GPU containers."` (`instances.py:207`) and `"--spot is only supported for GPU container instances."` (`cli/instance.py:204`). **You cannot get a spot VM** — so a spot 8× box with root/Docker/ForceP2P control is not purchasable.
- **No spot for CPU VMs** (`instances.py:194`).
- **No spot in europe-01** — the two europe-01 rows carry no `spot_price` key at all.
- Spot capacity is gated on `num_free_devices` of the *container* row (`server_meta.py:_available_devices`), and the create path raises `"No free spot {gpu} GPUs right now."` / `"Spot is not available for {gpu} right now."`.

**Not documented anywhere:** the word "spot" appears **zero** times in the docs pages for VM, Templates, Getting Started, SDK, CLI, Serverless and Settings (checked individually). There is **no published preemption policy, no eviction notice period, no data-retention-on-eviction statement, and no spot-vs-on-demand SLA.** See §5.

### 3.3 Multi-GPU: linear, no volume discount

- Catalogue prices are explicitly "per GPU pricing" (pricing page heading) and the account's own 8-GPU instance bills at 8 × the 1-GPU rate (§2.1, measured 15.293 vs 15.260 expected). **No multi-GPU discount.**
- Max 8 GPUs per instance ("Scale to 8 GPUs per instance"; "Up to 8 GPUs per VM"; "Up to 8 GPUs per template" — pricing page). Beyond that: "Need 25+ GPUs? Talk to sales" (pricing page) — no self-serve path.
- FAQ, verbatim: *"Being a bootstrapped startup, we are not able to offer any discounts."* (<https://docs.jarvislabs.ai/faqs/>). The only structured discounts are the spot and reserved tiers already in the table.

### 3.4 CPU VMs

`cpu_meta.combinations` from `misc/server_meta`, both India regions (`india-chennai-01`, `india-noida-01`), all `available: true`; **not available in europe-01** (the EU backend returns `cpu_meta.available: false`):

| plan | on-demand $/hr | 1 mo | 3 mo | 6 mo |
|---|---:|---:|---:|---:|
| 2 vCPU / 8 GB | 0.0496 | 0.0471 | 0.0446 | 0.0422 |
| 4 vCPU / 16 GB | 0.0992 | 0.0942 | 0.0893 | 0.0843 |
| 8 vCPU / 32 GB | 0.1984 | 0.1885 | 0.1786 | 0.1686 |
| 16 vCPU / 64 GB | 0.3968 | 0.3770 | 0.3571 | 0.3373 |
| 32 vCPU / 128 GB | 0.7936 | 0.7539 | 0.7142 | 0.6746 |

Formula on the public page: "$0.012 / vCPU + $0.0032 / GB RAM per hour" — which reproduces the API numbers exactly (32×0.012 + 128×0.0032 = 0.7936). No spot tier, no 1-year tier for CPU.

### 3.5 Billing granularity, storage, egress, paused instances

| Item | Value | Source |
|---|---|---|
| Billing granularity | **Per minute.** "Instances are billed per-minute. You are only charged for the total number of minutes used." | FAQ; pricing page ("per-minute billing") |
| Instance disk (`/home`) | **$0.00014 / GB / hour** = $0.1008 / GB / 30-day month | FAQ ("Your data is safely stored at a rate of $0.00014 per GB per hour… 50 GB = $5.04/month"); pricing page quotes "$0.10/GB · month" |
| Disk billed while **running** | **Yes, on top of the GPU rate.** Measured production rate matches 8×1.89 **+** 1000 GB×0.00014 | measured, §2.1 |
| Disk billed while **paused** | **Yes.** "Pausing an instance frees up compute… You will still be charged for the storage, so if you do not plan to use the instance anytime soon, consider deleting it." | <https://docs.jarvislabs.ai/getting_started/> |
| Compute while paused | **$0** — GPU/CPU/RAM released to other users | getting_started; FAQ |
| Shared filesystems | $0.00014/GB/hr on **provisioned** capacity regardless of use; size can only grow; up to 10 TB | <https://docs.jarvislabs.ai/filestorage/> |
| Max instance storage | 2 TB per instance ("Storage – Scale up to 2TB… Need more? Contact us.") | getting_started |
| Egress | **Free in-region.** ("Egress — Free in-region") | pricing page |
| Zero-balance behaviour | Instances auto-pause at zero balance and **all data is permanently deleted** and unrecoverable; storage is released too | FAQ |
| Refunds / withdrawal | None. Credits non-refundable, no withdrawal of unused balance | FAQ |
| Currency | USD for this account (`misc/` → `{"success":false}` ⇒ USD; `server_meta.currency:"USD"`) | API |
| Reserved capacity | Rates published in `reserved_pricing` (1/3/6 months, 1 year) and as pricing-page tabs, but there is **no self-serve reserve/commit endpoint** in the SDK; instance records carry `committed_resource_id`/`reservation_info` fields that are `null` for us. Contact-sales path only. | API + SDK + pricing page |

### 3.6 Catalogue fields that are inconsistent — treat with suspicion

- `price_per_week` / `price_per_month` are present but incoherent and **unused by the official SDK** (`ServerMetaGPU` sets `extra="ignore"` and does not declare them). Examples: H100 `price_per_week: 238` **and** `price_per_month: 238` (identical — impossible); L4 `price_per_week: 840` vs $0.44/hr × 168 h = $73.9; RTX-PRO6000 `price_per_week: 840`, `price_per_month: 3600` vs $1.89/hr × 168 h = $317.5. **Do not price anything off these fields.**
- **europe-01 H200 disagreement (reproducible, checked twice):** `backendeu` reports `price_per_hour: 3.80`; `backendn` and `backendc` report `3.99` for the *same* europe-01 row. europe-01 H100 is `2.99` from all three (and note that is *higher* than the India H100 rate of 2.69).

---

## 4. Availability

Source: `misc/server_meta` at 2026-08-17 12:50 UTC. `num_free_devices` = free now; `effective_num_free_devices` = what the SDK uses to admit an on-demand create (it can exceed `num_free_devices`, evidently counting reclaimable/soft-held devices); `spot_num_free_devices` = spot-eligible free devices.

| GPU | region | workload | free now | effective free (on-demand) | spot free |
|---|---|---|---:|---:|---:|
| **RTX-PRO6000 (96 GB)** | india-chennai-01 | **vm** | **4** | **6** | 4 |
| **RTX-PRO6000 (96 GB)** | india-chennai-01 | **container** | **0** | **8** | **0** |
| H200 | india-noida-01 | vm | 8 | 8 | 8 |
| H200 | india-noida-01 | container | 8 | 8 | 8 |
| H100 | india-noida-01 | vm | 8 | 8 | 8 |
| H100 | india-noida-01 | container | 0 | 0 | 0 |
| A100-80GB | india-noida-01 | vm | 0 | 0 | 0 |
| A100-80GB | india-noida-01 | container | 0 | 0 | 0 |
| A100 (40 GB) | india-noida-01 | container | 1 | 1 | 1 |
| L4 | india-noida-01 | vm | 4 | 7 | 4 |
| L4 | india-noida-01 | container | 5 | 6 | 5 |
| A30 | india-noida-01 | container | 0 | 0 | 0 |
| H100 | europe-01 | — | 35 | 35 | n/a |
| H200 | europe-01 | — | 26 | 26 | n/a |

Reading of the 96 GB rows: **4 free RTX PRO 6000 GPUs are available right now as a VM in Chennai** (SDK would admit up to 6 on the "effective" count), and **0 are available as a spot container** — so today, spot 96 GB is priced but not purchasable. Since 8 GPUs is the per-instance cap and only 4–6 are free, **a second 8× RTX PRO 6000 VM cannot be launched right now**; a 1×, 2× or 4× VM can.

Region and shape constraints (retrieved from SDK `constants.py` / `instances.py`):
- **europe-01 offers only H100 and H200** (`EUROPE_GPU_TYPES`) and **only 1 GPU per instance** (`EUROPE_GPU_COUNTS = {1}`; error `"EU1 supports only 1 GPU per instance"`). The SDK also clamps the displayed EU free count to 1. So EU is useless for any multi-GPU topology, and offers no 96 GB card at all.
- **RTX PRO 6000 exists only in india-chennai-01.** No 96 GB-class capacity in Noida or Europe.
- Auto-routing preference order is Noida → Chennai → Europe; a paused instance can only resume in its original region.
- Filesystems and VPCs are region-local; VM and VPC must share a region.
- Servers are in India and Finland, tier-3/tier-4 DCs (FAQ).

**Limits of what could be confirmed:** these counts are a single instantaneous read of the provider's own catalogue. They are not a reservation and the FAQ explicitly disclaims availability ("Once you pause or delete an instance, the resources are released… We do not guarantee any availability of the instance once it is released"). The only way to *prove* capacity is to create an instance, which is out of scope here.

---

## 5. Could not confirm

1. **Spot preemption semantics.** No documented eviction policy, notice period, maximum runtime, price-based vs capacity-based reclaim, or what happens to `/home` on eviction. The docs never mention spot at all; the only evidence spot exists is the API field, the CLI flag, and a pricing-page tab.
2. **Whether spot prices are fixed or float.** `spot_price` is a single scalar per SKU with no bid parameter anywhere in the create payload (`is_spot` is a bare boolean), which *looks* like a fixed discount rather than an auction — but nothing states this. `[INFERENCE]`
3. **Out-of-region / internet egress pricing.** The page says only "Egress — Free in-region". Cross-region and public-internet egress rates are unstated; no egress line item was observed in any API response.
4. **Reserved/committed purchase mechanics.** Rates are published; the term lengths are 1/3/6 months and 1 year; but there is no API to buy them and no statement of whether a reservation is capacity-guaranteed or merely a discount. Contact-sales only.
5. **Per-minute proration edges.** "Billed per-minute" is stated, but there is no published minimum billable duration (e.g. whether a 20-second instance costs 1 minute) and no statement on whether the boot period is billed.
6. **The 8-GPU cap as a hard quota.** "Up to 8 GPUs per instance" is a per-instance shape limit; whether the account has an aggregate GPU quota is not exposed by any endpoint.
7. **Which of the two conflicting europe-01 H200 prices ($3.80 vs $3.99) is actually charged.** Determining this requires launching an EU H200, which was not done.
8. **`api.jarvislabs.ai` behaviour.** It is unreachable from this workstation, so it could not be confirmed whether it is a decommissioned host, a firewalled one, or simply blocked by our egress path. Everything here uses the `backend*.jarvislabs.net` hosts the official SDK uses.
9. **Whether `effective_num_free_devices > num_free_devices` means reclaimable capacity.** The SDK trusts the higher number for on-demand admission and the lower for spot; the semantics are not documented. `[INFERENCE]` that the gap represents devices held by preemptible/soft workloads.

---

## 6. Cheapest and most expensive 96 GB-class GPU-hour

**Cheapest — $0.99 per GPU-hour: RTX PRO 6000 Blackwell (96 GB) spot, GPU *container*, `india-chennai-01`.**
Source: `GET https://backendc.jarvislabs.net/misc/server_meta` → RTX-PRO6000 row, `"spot_price": 0.99`; corroborated by the "Spot · save up to 56 %" tab on <https://jarvislabs.ai/pricing>. Two caveats, both retrieved: spot is **containers only** (no root VM, so no ForceP2P/driver control), and `spot_num_free_devices` for the RTX-PRO6000 container row is **0** right now, so it is priced but not currently purchasable.
*Cheapest actually purchasable today, and cheapest with VM/root access:* **$1.89/GPU-hr** on-demand RTX PRO 6000 (4 free VM devices, `num_free_devices: 4`). Cheapest committed rate: **$1.19/GPU-hr** on a 1-year reservation (`reserved_pricing.reserved_1y`), a 37 % discount — sales-negotiated, no self-serve endpoint.

**Most expensive — $3.99 per GPU-hour: H200 SXM (141 GB) on-demand, `india-noida-01`.**
Source: `GET /misc/server_meta` → H200 row, `"price_per_hour": 3.99`; matches the "$3.99 On-demand · /hr" H200 entry on <https://jarvislabs.ai/pricing>. This is 2.11× the on-demand RTX PRO 6000 rate and 4.03× its spot rate, for 141 GB instead of 96 GB. (The europe-01 H200 row is quoted at $3.99 by the India backends and $3.80 by the EU backend — see §3.4 — so even the EU reading does not exceed $3.99.)

Add **$0.00014 / GB / hour** of attached disk to every figure above, billed while running *and* while paused.
