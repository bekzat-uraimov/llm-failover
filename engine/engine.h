#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Everything needed to rebuild a generation on another engine. The KV cache is not
// here on purpose: restore() rebuilds it by replaying the tokens.
struct EngineState {
    std::vector<int> tokens;  // every token so far, prompt included, starting with BOS
    int pos = 0;              // next position to generate, always tokens.size()
    uint64_t rng_state = 0;   // sampler state after the last sample
};

class Engine {
public:
    static constexpr int BOS = 1;

    Engine(const std::string& model_path, const std::string& tokenizer_path,
           float temperature = 1.0f, float topp = 0.9f);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void start(const std::string& prompt, uint64_t seed);
    int next_token();  // forward + sample, updates state
    EngineState save() const;
    void restore(const EngineState& s);  // replay tokens through forward, then set rng

    std::string decode(int prev, int token) const;
    const std::vector<int>& tokens() const { return tokens_; }
    int seq_len() const { return config_.seq_len; }

private:
    struct Config {
        int dim;
        int hidden_dim;
        int n_layers;
        int n_heads;
        int n_kv_heads;
        int vocab_size;
        int seq_len;
    };

    // Pointers into the memory-mapped checkpoint
    struct Weights {
        const float* token_embedding_table;
        const float* rms_att_weight;
        const float* rms_ffn_weight;
        const float* wq;
        const float* wk;
        const float* wv;
        const float* wo;
        const float* w1;
        const float* w2;
        const float* w3;
        const float* rms_final_weight;
        const float* wcls;
    };

    // Scratch buffers for one forward pass, plus the KV cache
    struct RunState {
        std::vector<float> x, xb, xb2, hb, hb2, q, att, logits;
        std::vector<float> key_cache, value_cache;
    };

    struct ProbIndex {
        float prob;
        int index;
    };

    void load_model(const std::string& path);
    void load_tokenizer(const std::string& path);
    float* forward(int token, int pos);
    int sample(float* logits);
    std::vector<int> encode(const std::string& text) const;
    int lookup(const std::string& piece) const;

    Config config_{};
    Weights weights_{};
    RunState state_;
    int fd_ = -1;
    void* data_ = nullptr;
    size_t file_size_ = 0;

    std::vector<std::string> vocab_;
    std::vector<float> vocab_scores_;
    std::vector<std::pair<std::string, int>> sorted_vocab_;

    float temperature_;
    float topp_;
    std::vector<ProbIndex> probindex_;

    std::vector<int> tokens_;
    uint64_t rng_state_ = 0;
};
