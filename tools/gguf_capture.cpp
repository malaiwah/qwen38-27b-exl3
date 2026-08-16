// Hidden-state capture for GGUF quants, byte-compatible with `fidelity.py capture`.
//
// The distribution-fidelity protocol scores a candidate by replaying the BF16
// hidden state taken after the final RMSNorm and before the LM head through one
// shared head.  vLLM cannot read GGUF, so a GGUF quant can only enter the same
// ladder if some other engine produces that exact tensor in that exact file
// format.  llama.cpp does: for this architecture `llm_build_qwen35` assigns
// `res->t_embd` to the post-`output_norm` tensor immediately before
// `build_lora_mm(model.output, ...)`, which is the same mathematical point the
// vLLM hook captures (the final `Qwen3_5RMSNorm` feeding `lm_head`).  With
// `embeddings = true` and `pooling_type = NONE` every position of the batch is an
// output row and `llama_get_embeddings_ith` returns those states in batch order.
//
//   gguf_capture --model Q8_0.gguf --suite SUITE_DIR --out CAP_DIR
//                --indices 0-511 --ngl 999 --threads 16
//                --expect-hidden 5120
//
// Per context it writes `hidden_NNNN.safetensors` holding a single row-major
// `hidden_states` tensor of shape [context_length - 1, n_embd] in bfloat16 -
// the same bytes `safetensors.torch.save_file({"hidden_states": h})` writes -
// and appends the engine-side facts to `capture-engine.json`, which
// `gguf_manifest.py` turns into the `capture-manifest.json` the replay side
// validates.  Nothing here is allowed to guess: a token count or token digest
// that disagrees with the suite entry, a hidden width that disagrees with
// `--expect-hidden`, a missing output row or a short write aborts the run
// instead of leaving a plausible-looking file behind.
//
// Teacher forcing needs no sampling: one `llama_decode` of the whole token
// sequence with causal attention is exactly the forward pass vLLM performs for
// a prompt, and position `n-1` is dropped because it scores no next token.
#include "llama.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#ifndef LLAMACPP_COMMIT
#    error "build_llamacpp.sh must define LLAMACPP_COMMIT: an unpinned engine cannot be a receipt"
#endif

using json = nlohmann::json;

static const char * const CAPTURE_ENGINE_SCHEMA = "qwen38-gguf-capture-engine/1";

[[noreturn]] static void die(const std::string & msg) {
    fprintf(stderr, "gguf_capture: %s\n", msg.c_str());
    exit(1);
}

// ------------------------------------------------------------------ sha256
// The suite pins each context by the sha256 of `json.dumps(ids)`; the capture
// must be able to prove it consumed those exact ids, so it hashes what it read.
struct sha256 {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint8_t  buf[64] = {};
    size_t   len = 0;      // bytes buffered
    uint64_t total = 0;    // bytes absorbed

    static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t) p[4*i] << 24 | (uint32_t) p[4*i+1] << 16 |
                   (uint32_t) p[4*i+2] << 8 | (uint32_t) p[4*i+3];
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = ror(w[i-15], 7) ^ ror(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 = ror(w[i-2], 17) ^ ror(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], x = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = x + S1 + ch + k[i] + w[i];
            const uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            x = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += x;
    }

    void update(const void * data, size_t n) {
        const uint8_t * p = (const uint8_t *) data;
        total += n;
        while (n > 0) {
            const size_t take = std::min(n, sizeof(buf) - len);
            memcpy(buf + len, p, take);
            len += take; p += take; n -= take;
            if (len == sizeof(buf)) { block(buf); len = 0; }
        }
    }

    std::string hex() {
        uint64_t bits = total * 8;
        const uint8_t one = 0x80;
        update(&one, 1);
        const uint8_t zero = 0x00;
        while (len != 56) { update(&zero, 1); }
        uint8_t be[8];
        for (int i = 0; i < 8; ++i) { be[i] = (uint8_t) (bits >> (56 - 8*i)); }
        update(be, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) { snprintf(out + 8*i, 9, "%08x", h[i]); }
        return std::string(out, 64);
    }
};

// ------------------------------------------------------------------ bfloat16
// `torch.Tensor.to(torch.bfloat16)` is `round_to_nearest_even`, so this is it:
// tie-to-even on the discarded 16 bits, verified bit-identical against
// `torch.from_numpy(x).to(torch.bfloat16)` over 2,012,449 non-NaN inputs
// (every exponent, tie patterns 0x8000/0x7fff/0x0001, zeros, denormals,
// infinities, and 2M random bit patterns).  NaN is the one input where torch's
// vectorized CPU path returns a different NaN payload (0xffff) than the scalar
// rule (0x7fc0), so a non-finite state is refused outright at the call site
// instead of being encoded: hidden states that are not finite are a broken
// candidate, not a rounding question.  Overflow needs no branch - the mantissa
// carry walks the exponent to 0xff with a zero mantissa, which is +/-inf, and
// that is caught by the same refusal.
static inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return 0x7fc0u;
    }
    const uint32_t bias = ((u >> 16) & 1u) + 0x7fffu;
    return (uint16_t) ((u + bias) >> 16);
}

// -------------------------------------------------------------- safetensors
// safetensors is a JSON header (u64 little-endian length, then the header
// padded with spaces to an 8-byte multiple) followed by the raw tensor bytes.
// One tensor named `hidden_states`, so serialization order is not a question;
// the header text below is byte-identical to what `save_file` emits.
static void write_safetensors_bf16(const std::string & path, int64_t rows, int64_t cols,
                                   const std::vector<uint16_t> & data) {
    const size_t nbytes = (size_t) rows * (size_t) cols * sizeof(uint16_t);
    if (data.size() * sizeof(uint16_t) != nbytes) {
        die("internal: payload size disagrees with [" + std::to_string(rows) + "," +
            std::to_string(cols) + "]");
    }
    char header[256];
    const int n = snprintf(header, sizeof(header),
        "{\"hidden_states\":{\"dtype\":\"BF16\",\"shape\":[%" PRId64 ",%" PRId64
        "],\"data_offsets\":[0,%zu]}}", rows, cols, nbytes);
    if (n <= 0 || (size_t) n >= sizeof(header)) {
        die("internal: safetensors header did not fit");
    }
    std::string hdr(header, (size_t) n);
    hdr.append((8 - hdr.size() % 8) % 8, ' ');
    const uint64_t hdr_len = hdr.size();

    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) { die("cannot open " + tmp); }
    uint8_t le[8];
    for (int i = 0; i < 8; ++i) { le[i] = (uint8_t) (hdr_len >> (8*i)); }
    const bool ok = fwrite(le, 1, 8, f) == 8 &&
                    fwrite(hdr.data(), 1, hdr.size(), f) == hdr.size() &&
                    (nbytes == 0 || fwrite(data.data(), 1, nbytes, f) == nbytes);
    const bool flushed = fflush(f) == 0;
    const bool closed = fclose(f) == 0;
    if (!ok || !flushed || !closed) {
        remove(tmp.c_str());
        die("short write on " + tmp);
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        die("cannot rename " + tmp + " -> " + path);
    }
}

// --------------------------------------------------------------------- util
static std::string read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { die("cannot read " + path); }
    std::string out;
    char buf[1 << 16];
    size_t got;
    while ((got = fread(buf, 1, sizeof(buf), f)) > 0) { out.append(buf, got); }
    const bool bad = ferror(f) != 0;
    fclose(f);
    if (bad) { die("read error on " + path); }
    return out;
}

static int64_t file_size(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { die("cannot open " + path); }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); die("cannot seek " + path); }
    const long n = ftell(f);
    fclose(f);
    if (n < 0) { die("cannot size " + path); }
    return (int64_t) n;
}

// `json.dumps(ids)` with CPython's default separators, which is what the suite
// manifest hashed.
static std::string python_json_ids(const std::vector<llama_token> & ids) {
    std::string s = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) { s += ", "; }
        s += std::to_string(ids[i]);
    }
    s += "]";
    return s;
}

// "0,3,7-9" -> {0,3,7,8,9}; ascending, duplicate-free, or the run aborts.
static std::vector<int> parse_indices(const std::string & spec) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos <= spec.size()) {
        const size_t comma = spec.find(',', pos);
        std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? spec.size() + 1 : comma + 1;
        item.erase(0, item.find_first_not_of(" \t") == std::string::npos ? item.size() : item.find_first_not_of(" \t"));
        while (!item.empty() && isspace((unsigned char) item.back())) { item.pop_back(); }
        if (item.empty()) { continue; }
        const size_t dash = item.find('-', 1);
        long lo = 0, hi = 0;
        if (dash == std::string::npos) {
            lo = hi = strtol(item.c_str(), nullptr, 10);
        } else {
            lo = strtol(item.substr(0, dash).c_str(), nullptr, 10);
            const std::string tail = item.substr(dash + 1);
            hi = tail.empty() ? -1 : strtol(tail.c_str(), nullptr, 10);
        }
        if (lo < 0 || hi < lo) { die("bad --indices item '" + item + "'"); }
        for (long i = lo; i <= hi; ++i) { out.push_back((int) i); }
    }
    std::sort(out.begin(), out.end());
    if (std::adjacent_find(out.begin(), out.end()) != out.end()) {
        die("--indices repeats an index");
    }
    return out;
}

struct params {
    std::string model;
    std::string suite;
    std::string out;
    std::string indices = "all";
    int  ngl = 0;
    int  threads = 16;
    int  ubatch = 0;      // 0 = one physical batch per context, as vLLM capture does
    int  expect_hidden = 0;
    bool flash_attn = false;
    bool cpu_only = false;
};

static void usage() {
    fprintf(stderr,
        "usage: gguf_capture --model FILE.gguf --suite SUITE_DIR --out CAP_DIR\n"
        "                    [--indices 0,3,7-9|all] [--ngl N] [--threads N]\n"
        "                    [--ubatch N] [--expect-hidden N] [--flash-attn] [--cpu-only]\n"
        "\n"
        "  --indices        contexts to capture, by suite index (default: all)\n"
        "  --ngl            layers offloaded to GPU (0 keeps the weights on CPU)\n"
        "  --cpu-only       offer the engine no device at all (upstream `--device none`).\n"
        "                   --ngl 0 keeps weights off the GPU but still schedules ops on\n"
        "                   one when a device exists; this is how you touch no GPU.\n"
        "  --ubatch         physical batch; default 0 evaluates each context in ONE ubatch,\n"
        "                   which is what `fidelity.py capture` does (max_num_batched_tokens\n"
        "                   = context length).  A smaller value cuts the device logit buffer\n"
        "                   (n_vocab x ubatch x 4 B) at the cost of splitting the forward pass\n"
        "  --expect-hidden  abort unless the hidden width is exactly this (5120 for Qwen3.8-27B)\n");
}

int main(int argc, char ** argv) {
    params p;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) { die(std::string("missing value for ") + what); }
            return argv[++i];
        };
        if      (a == "--model")         { p.model = next("--model"); }
        else if (a == "--suite")         { p.suite = next("--suite"); }
        else if (a == "--out")           { p.out = next("--out"); }
        else if (a == "--indices")       { p.indices = next("--indices"); }
        else if (a == "--ngl")           { p.ngl = atoi(next("--ngl").c_str()); }
        else if (a == "--threads")       { p.threads = atoi(next("--threads").c_str()); }
        else if (a == "--ubatch")        { p.ubatch = atoi(next("--ubatch").c_str()); }
        else if (a == "--expect-hidden") { p.expect_hidden = atoi(next("--expect-hidden").c_str()); }
        else if (a == "--flash-attn")    { p.flash_attn = true; }
        else if (a == "--cpu-only")      { p.cpu_only = true; }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { usage(); die("unknown argument '" + a + "'"); }
    }
    if (p.model.empty() || p.suite.empty() || p.out.empty()) {
        usage();
        die("--model, --suite and --out are required");
    }
    if (p.threads <= 0 || p.ubatch < 0) { die("--threads must be positive, --ubatch >= 0"); }

    // ---------------------------------------------------------- the suite
    const json suite = json::parse(read_file(p.suite + "/suite-manifest.json"));
    const int ctx_len = suite.at("context_length").get<int>();
    if (ctx_len < 2) { die("suite context_length must be at least 2"); }
    // Suites from v5 onward record the width they were built for; treat it as
    // binding, so a wrong model fails before it spends a single forward pass.
    const int suite_hidden = suite.value("hidden_size", 0);
    std::vector<json> chosen;
    if (p.indices == "all") {
        for (const auto & c : suite.at("context_index")) { chosen.push_back(c); }
    } else {
        std::vector<json> by_index;
        for (const auto & c : suite.at("context_index")) { by_index.push_back(c); }
        for (const int want : parse_indices(p.indices)) {
            bool found = false;
            for (const auto & c : by_index) {
                if (c.at("index").get<int>() == want) { chosen.push_back(c); found = true; break; }
            }
            if (!found) { die("suite has no context index " + std::to_string(want)); }
        }
    }
    if (chosen.empty()) { die("no contexts selected"); }

    // ---------------------------------------------------------- the engine
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = p.ngl;
    mparams.check_tensors = true;   // a corrupt quant must fail loudly, not silently
    // Upstream's `--device none`: a devices list holding only the terminator, which
    // leaves the model with no device and every op on the CPU backend.
    static ggml_backend_dev_t no_devices[] = { nullptr };
    if (p.cpu_only) {
        mparams.devices = no_devices;
        if (p.ngl != 0) { die("--cpu-only and --ngl > 0 contradict each other"); }
    }

    llama_model * model = llama_model_load_from_file(p.model.c_str(), mparams);
    if (!model) { die("failed to load " + p.model); }

    const int n_embd     = llama_model_n_embd(model);
    const int n_embd_out = llama_model_n_embd_out(model);
    if (n_embd_out <= 0) { die("model reports no output embedding width"); }
    if (p.expect_hidden && n_embd_out != p.expect_hidden) {
        die("hidden width " + std::to_string(n_embd_out) + " != --expect-hidden " +
            std::to_string(p.expect_hidden));
    }
    if (suite_hidden && n_embd_out != suite_hidden) {
        die("hidden width " + std::to_string(n_embd_out) + " != suite hidden_size " +
            std::to_string(suite_hidden));
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                 = (uint32_t) ctx_len;
    cparams.n_batch               = (uint32_t) ctx_len;
    cparams.n_ubatch              = (uint32_t) (p.ubatch > 0 ? std::min(p.ubatch, ctx_len) : ctx_len);
    cparams.n_seq_max             = 1;
    cparams.n_outputs_max         = (uint32_t) ctx_len;   // every position is an output row
    cparams.n_outputs_max_per_seq = (uint32_t) ctx_len;   // default is 1: it would reject the batch
    cparams.n_threads             = p.threads;
    cparams.n_threads_batch       = p.threads;
    cparams.pooling_type          = LLAMA_POOLING_TYPE_NONE;
    cparams.attention_type        = LLAMA_ATTENTION_TYPE_CAUSAL;
    cparams.flash_attn_type       = p.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                                 : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.embeddings            = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { die("failed to create context"); }
    if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE) {
        die("engine refused pooling type NONE; per-position states are unavailable");
    }
    if ((int) llama_n_ctx(ctx) < ctx_len) {
        die("engine context " + std::to_string(llama_n_ctx(ctx)) + " < suite context " +
            std::to_string(ctx_len));
    }

    char desc[256] = {};
    llama_model_desc(model, desc, sizeof(desc));
    char ftype[128] = {};
    if (llama_model_meta_val_str(model, "general.file_type", ftype, sizeof(ftype)) < 0) {
        snprintf(ftype, sizeof(ftype), "%s", "unset");
    }
    char arch[128] = {};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) {
        snprintf(arch, sizeof(arch), "%s", "unknown");
    }

    json engine = {
        {"schema", CAPTURE_ENGINE_SCHEMA},
        {"engine", "llama.cpp"},
        {"llamacpp_commit", LLAMACPP_COMMIT},
        {"model_path", p.model},
        {"model_size_bytes", file_size(p.model)},
        {"model_desc", desc},
        {"architecture", arch},
        {"general_file_type", ftype},
        {"n_embd", n_embd},
        {"n_embd_out", n_embd_out},
        {"n_ctx", (int) llama_n_ctx(ctx)},
        {"n_batch", (int) llama_n_batch(ctx)},
        {"n_ubatch", (int) llama_n_ubatch(ctx)},
        {"n_threads", p.threads},
        {"n_gpu_layers", p.ngl},
        {"flash_attn", p.flash_attn},
        {"devices", p.cpu_only ? "none" : "default"},
        {"pooling_type", "none"},
        {"attention_type", "causal"},
        {"embeddings", true},
        {"hidden_dtype", "bfloat16"},
        {"hidden_rounding", "round_to_nearest_even"},
        {"dropped_position", "last"},
        {"suite", p.suite},
        {"suite_context_length", ctx_len},
        {"suite_token_sha256", suite.value("suite_token_sha256", "")},
    };

    // A capture directory may be filled by several invocations (one per index
    // range).  Different engine settings inside one directory would make the
    // resulting manifest a lie, so they are refused rather than merged.
    const std::string engine_path = p.out + "/capture-engine.json";
    json contexts = json::array();
    {
        FILE * probe = fopen(engine_path.c_str(), "rb");
        if (probe) {
            fclose(probe);
            const json prior = json::parse(read_file(engine_path));
            for (auto it = engine.begin(); it != engine.end(); ++it) {
                if (!prior.contains(it.key()) || prior.at(it.key()) != it.value()) {
                    die("capture-engine.json in " + p.out + " disagrees on '" + it.key() +
                        "'; use a new output directory");
                }
            }
            contexts = prior.value("captures", json::array());
        }
    }

    llama_batch batch = llama_batch_init(ctx_len, 0, 1);
    std::vector<uint16_t> rowbuf((size_t) (ctx_len - 1) * (size_t) n_embd_out);
    const int64_t started = (int64_t) time(nullptr);

    for (const auto & c : chosen) {
        const int index = c.at("index").get<int>();
        const std::string token_file = p.suite + "/" + c.at("file").get<std::string>();
        const json ids_json = json::parse(read_file(token_file));
        if (!ids_json.is_array()) { die(token_file + " is not a token array"); }

        std::vector<llama_token> ids;
        ids.reserve(ids_json.size());
        for (const auto & t : ids_json) { ids.push_back(t.get<llama_token>()); }

        const int n_tokens = (int) ids.size();
        if (c.contains("tokens") && c.at("tokens").get<int>() != n_tokens) {
            die("context " + std::to_string(index) + ": suite says " +
                std::to_string(c.at("tokens").get<int>()) + " tokens, file has " +
                std::to_string(n_tokens));
        }
        if (n_tokens != ctx_len) {
            die("context " + std::to_string(index) + ": " + std::to_string(n_tokens) +
                " tokens but suite context_length is " + std::to_string(ctx_len));
        }
        sha256 sh;
        const std::string dumped = python_json_ids(ids);
        sh.update(dumped.data(), dumped.size());
        const std::string token_sha = sh.hex();
        if (c.contains("token_sha256") && c.at("token_sha256").get<std::string>() != token_sha) {
            die("context " + std::to_string(index) + ": token digest drift (" + token_sha +
                " != " + c.at("token_sha256").get<std::string>() + ")");
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = n_tokens;
        for (int i = 0; i < n_tokens; ++i) {
            batch.token[i]     = ids[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;   // every position must come back as an output row
        }
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            die("llama_decode failed with " + std::to_string(rc) + " on context " +
                std::to_string(index));
        }
        // The last position scores no next token, so it is dropped - but it is
        // still read first, because a missing final row means the engine did not
        // evaluate everything we handed it.
        if (llama_get_embeddings_ith(ctx, n_tokens - 1) == nullptr) {
            die("context " + std::to_string(index) + ": engine returned no state for the "
                "final position; the batch was not fully evaluated");
        }
        for (int i = 0; i < n_tokens - 1; ++i) {
            const float * src = llama_get_embeddings_ith(ctx, i);
            if (src == nullptr) {
                die("context " + std::to_string(index) + ": no hidden state at position " +
                    std::to_string(i));
            }
            uint16_t * dst = rowbuf.data() + (size_t) i * (size_t) n_embd_out;
            for (int j = 0; j < n_embd_out; ++j) {
                const uint16_t v = f32_to_bf16_rne(src[j]);
                if ((v & 0x7f80u) == 0x7f80u) {
                    die("context " + std::to_string(index) + ": non-finite hidden state at "
                        "position " + std::to_string(i) + " column " + std::to_string(j));
                }
                dst[j] = v;
            }
        }

        char name[64];
        snprintf(name, sizeof(name), "hidden_%04d.safetensors", index);
        write_safetensors_bf16(p.out + "/" + name, n_tokens - 1, n_embd_out, rowbuf);

        json record = {
            {"index", index},
            {"token_file", c.at("file").get<std::string>()},
            {"tokens", n_tokens},
            {"token_sha256", token_sha},
            {"file", name},
            {"shape", {n_tokens - 1, n_embd_out}},
        };
        bool replaced = false;
        for (auto & prior : contexts) {
            if (prior.at("index").get<int>() == index) { prior = record; replaced = true; break; }
        }
        if (!replaced) { contexts.push_back(record); }
        std::sort(contexts.begin(), contexts.end(), [](const json & a, const json & b) {
            return a.at("index").get<int>() < b.at("index").get<int>();
        });
        engine["captures"] = contexts;
        engine["elapsed_sec"] = (int64_t) time(nullptr) - started;
        {
            const std::string tmp = engine_path + ".tmp";
            FILE * f = fopen(tmp.c_str(), "wb");
            if (!f) { die("cannot write " + tmp); }
            const std::string text = engine.dump(2);
            const bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
            const bool closed = fclose(f) == 0;
            if (!ok || !closed) { remove(tmp.c_str()); die("short write on " + tmp); }
            if (rename(tmp.c_str(), engine_path.c_str()) != 0) {
                die("cannot rename " + tmp);
            }
        }
        printf("captured %s rows=%d cols=%d tokens=%d\n", name, n_tokens - 1, n_embd_out, n_tokens);
        fflush(stdout);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("capture_done {\"contexts\": %zu, \"engine\": \"%s\", \"elapsed_sec\": %" PRId64 "}\n",
           chosen.size(), engine_path.c_str(), (int64_t) time(nullptr) - started);
    return 0;
}
