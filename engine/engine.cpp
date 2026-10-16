// Adapted from llama2.c by Andrej Karpathy (MIT): https://github.com/karpathy/llama2.c
// Changes: wrapped in a C++ class, added save and restore, replaced exit() with exceptions.
// The forward pass, tokenizer and sampler math are unchanged so output matches run.c.

#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

void rmsnorm(float* o, const float* x, const float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// W (d,n) @ x (n,) -> xout (d,). Single threaded on purpose: a fixed summation order
// is part of the determinism guarantee.
void matmul(float* xout, const float* x, const float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

uint32_t random_u32(uint64_t* state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(uint64_t* state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_argmax(const float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(const float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

template <typename T>
int compare_prob(const void* a, const void* b) {
    const T* a_ = static_cast<const T*>(a);
    const T* b_ = static_cast<const T*>(b);
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

// Top-p (nucleus) sampling: sample from the smallest set of tokens whose probability exceeds topp.
template <typename T>
int sample_topp(const float* probabilities, int n, float topp, T* probindex, float coin) {
    int n0 = 0;
    // values smaller than (1 - topp) / (n - 1) can't be part of the result, so skip them before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    // qsort, not std::sort: ties must break the same way run.c breaks them
    qsort(probindex, n0, sizeof(T), compare_prob<T>);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

}  // namespace

Engine::Engine(const std::string& model_path, const std::string& tokenizer_path, float temperature, float topp)
    : temperature_(temperature), topp_(topp) {
    load_model(model_path);
    load_tokenizer(tokenizer_path);
    probindex_.resize(config_.vocab_size);
}

Engine::~Engine() {
    if (data_ && data_ != MAP_FAILED) munmap(data_, file_size_);
    if (fd_ != -1) close(fd_);
}

void Engine::load_model(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ == -1) throw std::runtime_error("can't open model: " + path);
    struct stat st;
    if (fstat(fd_, &st) != 0 || static_cast<size_t>(st.st_size) < sizeof(Config)) {
        throw std::runtime_error("model file too small: " + path);
    }
    file_size_ = st.st_size;
    data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);

    Config& p = config_;
    std::memcpy(&p, data_, sizeof(Config));
    // A negative vocab size in the header means the classifier weights aren't shared
    bool shared_weights = p.vocab_size > 0;
    p.vocab_size = std::abs(p.vocab_size);
    if (p.dim <= 0 || p.n_layers <= 0 || p.n_heads <= 0 || p.n_kv_heads <= 0 || p.seq_len <= 0) {
        throw std::runtime_error("bad model header: " + path);
    }

    int head_size = p.dim / p.n_heads;
    uint64_t n_layers = p.n_layers;
    const float* ptr = reinterpret_cast<const float*>(static_cast<const char*>(data_) + sizeof(Config));
    Weights& w = weights_;
    w.token_embedding_table = ptr;
    ptr += p.vocab_size * p.dim;
    w.rms_att_weight = ptr;
    ptr += n_layers * p.dim;
    w.wq = ptr;
    ptr += n_layers * p.dim * (p.n_heads * head_size);
    w.wk = ptr;
    ptr += n_layers * p.dim * (p.n_kv_heads * head_size);
    w.wv = ptr;
    ptr += n_layers * p.dim * (p.n_kv_heads * head_size);
    w.wo = ptr;
    ptr += n_layers * (p.n_heads * head_size) * p.dim;
    w.rms_ffn_weight = ptr;
    ptr += n_layers * p.dim;
    w.w1 = ptr;
    ptr += n_layers * p.dim * p.hidden_dim;
    w.w2 = ptr;
    ptr += n_layers * p.hidden_dim * p.dim;
    w.w3 = ptr;
    ptr += n_layers * p.dim * p.hidden_dim;
    w.rms_final_weight = ptr;
    ptr += p.dim;
    ptr += p.seq_len * head_size / 2;  // skip what used to be freq_cis_real (RoPE)
    ptr += p.seq_len * head_size / 2;  // skip what used to be freq_cis_imag (RoPE)
    w.wcls = shared_weights ? w.token_embedding_table : ptr;
    if (!shared_weights) ptr += static_cast<uint64_t>(p.vocab_size) * p.dim;

    const char* end = static_cast<const char*>(data_) + file_size_;
    if (reinterpret_cast<const char*>(ptr) > end) throw std::runtime_error("model file truncated: " + path);

    int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    RunState& s = state_;
    s.x.assign(p.dim, 0.0f);
    s.xb.assign(p.dim, 0.0f);
    s.xb2.assign(p.dim, 0.0f);
    s.hb.assign(p.hidden_dim, 0.0f);
    s.hb2.assign(p.hidden_dim, 0.0f);
    s.q.assign(p.dim, 0.0f);
    s.key_cache.assign(static_cast<size_t>(p.n_layers) * p.seq_len * kv_dim, 0.0f);
    s.value_cache.assign(static_cast<size_t>(p.n_layers) * p.seq_len * kv_dim, 0.0f);
    s.att.assign(static_cast<size_t>(p.n_heads) * p.seq_len, 0.0f);
    s.logits.assign(p.vocab_size, 0.0f);
}

void Engine::load_tokenizer(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("can't open tokenizer: " + path);
    int max_token_length;
    bool ok = fread(&max_token_length, sizeof(int), 1, f) == 1;
    vocab_.resize(config_.vocab_size);
    vocab_scores_.resize(config_.vocab_size);
    for (int i = 0; ok && i < config_.vocab_size; i++) {
        int len;
        ok = fread(&vocab_scores_[i], sizeof(float), 1, f) == 1 && fread(&len, sizeof(int), 1, f) == 1 && len >= 0;
        if (ok && len > 0) {
            vocab_[i].resize(len);
            ok = fread(vocab_[i].data(), len, 1, f) == 1;
        }
    }
    fclose(f);
    if (!ok) throw std::runtime_error("bad tokenizer file: " + path);

    sorted_vocab_.reserve(vocab_.size());
    for (int i = 0; i < config_.vocab_size; i++) sorted_vocab_.emplace_back(vocab_[i], i);
    std::sort(sorted_vocab_.begin(), sorted_vocab_.end());
}

float* Engine::forward(int token, int pos) {
    const Config* p = &config_;
    const Weights* w = &weights_;
    RunState* s = &state_;
    float* x = s->x.data();
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;  // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    std::memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(*x));

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb.data(), x, w->rms_att_weight + l * dim, dim);

        // This position's key and value are written straight into the KV cache
        size_t loff = static_cast<size_t>(l) * p->seq_len * kv_dim;
        float* k = s->key_cache.data() + loff + pos * kv_dim;
        float* v = s->value_cache.data() + loff + pos * kv_dim;

        matmul(s->q.data(), s->xb.data(), w->wq + l * dim * dim, dim, dim);
        matmul(k, s->xb.data(), w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(v, s->xb.data(), w->wv + l * dim * kv_dim, dim, kv_dim);

        // RoPE: rotate q and k in each head by an angle that depends on pos
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;  // 2 = q and k, 1 = q only
            for (int r = 0; r < rotn; r++) {
                float* vec = r == 0 ? s->q.data() : k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        for (int h = 0; h < p->n_heads; h++) {
            const float* q = s->q.data() + h * head_size;
            float* att = s->att.data() + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                const float* kt = s->key_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * kt[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            float* xb = s->xb.data() + h * head_size;
            std::memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float* vt = s->value_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * vt[i];
                }
            }
        }

        matmul(s->xb2.data(), s->xb.data(), w->wo + l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        rmsnorm(s->xb.data(), x, w->rms_ffn_weight + l * dim, dim);

        // FFN: w2(silu(w1(x)) * w3(x))
        matmul(s->hb.data(), s->xb.data(), w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2.data(), s->xb.data(), w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        matmul(s->xb.data(), s->hb.data(), w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits.data(), x, w->wcls, p->dim, p->vocab_size);
    return s->logits.data();
}

int Engine::sample(float* logits) {
    int n = config_.vocab_size;
    if (temperature_ == 0.0f) {
        return sample_argmax(logits, n);
    }
    for (int q = 0; q < n; q++) {
        logits[q] /= temperature_;
    }
    softmax(logits, n);
    // The only source of randomness. Each sample advances rng_state_ exactly once.
    float coin = random_f32(&rng_state_);
    if (topp_ <= 0 || topp_ >= 1) {
        return sample_mult(logits, n, coin);
    }
    return sample_topp(logits, n, topp_, probindex_.data(), coin);
}

int Engine::lookup(const std::string& piece) const {
    auto it = std::lower_bound(sorted_vocab_.begin(), sorted_vocab_.end(), std::make_pair(piece, -1));
    return it != sorted_vocab_.end() && it->first == piece ? it->second : -1;
}

std::vector<int> Engine::encode(const std::string& text) const {
    std::vector<int> tokens{BOS};

    // sentencepiece's add_dummy_prefix: prepend a space, but only if text isn't empty
    if (!text.empty()) tokens.push_back(lookup(" "));

    // Split the UTF-8 input into codepoints; unknown codepoints fall back to one token per byte
    std::string buf;
    for (size_t i = 0; i < text.size(); i++) {
        unsigned char c = text[i];
        if ((c & 0xC0) != 0x80) buf.clear();  // not a continuation byte: a new codepoint starts
        buf += static_cast<char>(c);

        bool next_is_continuation = i + 1 < text.size() && (static_cast<unsigned char>(text[i + 1]) & 0xC0) == 0x80;
        if (next_is_continuation && buf.size() < 4) continue;

        int id = lookup(buf);
        if (id != -1) {
            tokens.push_back(id);
        } else {
            // +3 because the first 3 vocab entries are <unk>, <s>, </s>
            for (unsigned char b : buf) tokens.push_back(b + 3);
        }
        buf.clear();
    }

    // Greedily merge the best-scoring adjacent pair until no pair is in the vocab
    for (;;) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            int id = lookup(vocab_[tokens[i]] + vocab_[tokens[i + 1]]);
            if (id != -1 && vocab_scores_[id] > best_score) {
                best_score = vocab_scores_[id];
                best_id = id;
                best_idx = static_cast<int>(i);
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }
    return tokens;
}

std::string Engine::decode(int prev, int token) const {
    std::string piece = vocab_.at(token);
    // After BOS, sentencepiece strips the leading space
    if (prev == BOS && !piece.empty() && piece[0] == ' ') piece.erase(0, 1);
    // Raw byte tokens look like "<0x0A>"
    unsigned char byte_val;
    if (sscanf(piece.c_str(), "<0x%02hhX>", &byte_val) == 1) return std::string(1, static_cast<char>(byte_val));
    return piece;
}

void Engine::start(const std::string& prompt, uint64_t seed) {
    // xorshift never leaves state 0, so seed 0 would sample the same coin forever
    if (seed == 0) throw std::invalid_argument("seed must be non-zero");
    std::vector<int> tokens = encode(prompt);
    if (static_cast<int>(tokens.size()) > config_.seq_len) throw std::invalid_argument("prompt too long");
    restore({tokens, static_cast<int>(tokens.size()), seed});
}

int Engine::next_token() {
    int pos = static_cast<int>(tokens_.size()) - 1;  // position of the last token, the one we feed in
    if (pos < 0) throw std::logic_error("next_token() before start() or restore()");
    if (pos >= config_.seq_len) throw std::out_of_range("context is full");
    int next = sample(forward(tokens_.back(), pos));
    tokens_.push_back(next);
    return next;
}

EngineState Engine::save() const {
    return {tokens_, static_cast<int>(tokens_.size()), rng_state_};
}

void Engine::restore(const EngineState& s) {
    if (s.tokens.empty() || s.tokens[0] != BOS) throw std::invalid_argument("tokens must start with BOS");
    if (s.pos != static_cast<int>(s.tokens.size())) throw std::invalid_argument("pos must equal len(tokens)");
    if (static_cast<int>(s.tokens.size()) > config_.seq_len + 1) throw std::invalid_argument("too many tokens");
    for (int t : s.tokens) {
        if (t < 0 || t >= config_.vocab_size) throw std::invalid_argument("token out of range");
    }
    if (s.rng_state == 0) throw std::invalid_argument("rng_state must be non-zero");

    // Rebuild the KV cache by running every token except the last through the model.
    // Same inputs in the same order as the original run, so the cache is bit for bit the same.
    // The last token goes in on the next next_token() call, just like during generation.
    tokens_ = s.tokens;
    for (size_t pos = 0; pos + 1 < tokens_.size(); pos++) {
        forward(tokens_[pos], static_cast<int>(pos));
    }
    rng_state_ = s.rng_state;
}
