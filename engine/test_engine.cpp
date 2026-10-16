// Engine tests. Usage: test_engine <model.bin> <tokenizer.bin>

#include "engine.h"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string model_path;
std::string tokenizer_path;
int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "  %s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            failures++;                                                   \
        }                                                                 \
    } while (0)

template <typename F>
bool throws(F fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

std::vector<int> generate(Engine& e, int n) {
    std::vector<int> out;
    for (int i = 0; i < n; i++) out.push_back(e.next_token());
    return out;
}

const char* PROMPT = "Once upon a time";

void test_same_seed_same_tokens() {
    Engine a(model_path, tokenizer_path), b(model_path, tokenizer_path);
    a.start(PROMPT, 42);
    b.start(PROMPT, 42);
    CHECK(generate(a, 100) == generate(b, 100));
}

void test_different_seed_diverges() {
    Engine a(model_path, tokenizer_path), b(model_path, tokenizer_path);
    a.start(PROMPT, 42);
    b.start(PROMPT, 43);
    CHECK(generate(a, 100) != generate(b, 100));
}

// The day 2 "done when": 100 straight == 50 + restore + 50, on a different engine
void test_restore_on_fresh_engine_continues_identically() {
    Engine a(model_path, tokenizer_path);
    a.start(PROMPT, 42);
    std::vector<int> first = generate(a, 50);
    EngineState saved = a.save();
    std::vector<int> straight = first;
    for (int t : generate(a, 50)) straight.push_back(t);

    Engine b(model_path, tokenizer_path);
    b.restore(saved);
    std::vector<int> resumed = first;
    for (int t : generate(b, 50)) resumed.push_back(t);

    CHECK(resumed == straight);
    CHECK(a.tokens() == b.tokens());
}

void test_restore_rewinds_same_engine() {
    Engine e(model_path, tokenizer_path);
    e.start(PROMPT, 7);
    generate(e, 20);
    EngineState saved = e.save();
    std::vector<int> expected = generate(e, 30);

    e.restore(saved);
    CHECK(generate(e, 30) == expected);
}

void test_restore_with_only_the_prompt() {
    Engine a(model_path, tokenizer_path), b(model_path, tokenizer_path);
    a.start(PROMPT, 42);
    b.restore(a.save());
    CHECK(generate(a, 20) == generate(b, 20));
}

void test_save_reports_position_and_rng() {
    Engine e(model_path, tokenizer_path);
    e.start(PROMPT, 42);
    EngineState s = e.save();
    CHECK(s.tokens.front() == Engine::BOS);
    CHECK(s.pos == static_cast<int>(s.tokens.size()));
    CHECK(s.rng_state == 42);

    e.next_token();
    EngineState s2 = e.save();
    CHECK(s2.pos == s.pos + 1);
    CHECK(s2.rng_state != 42);
}

void test_restore_rejects_bad_state() {
    Engine e(model_path, tokenizer_path);
    CHECK(throws([&] { e.restore({{}, 0, 1}); }));                  // empty
    CHECK(throws([&] { e.restore({{5, 6}, 2, 1}); }));              // no BOS
    CHECK(throws([&] { e.restore({{1, 6}, 5, 1}); }));              // pos mismatch
    CHECK(throws([&] { e.restore({{1, 999999}, 2, 1}); }));         // token out of range
    CHECK(throws([&] { e.restore({{1, 6}, 2, 0}); }));              // rng 0 never advances
    CHECK(throws([&] { e.start(PROMPT, 0); }));
    CHECK(throws([&] { e.next_token(); }));                         // nothing started
}

void test_stops_at_context_limit() {
    Engine e(model_path, tokenizer_path);
    e.start("", 42);
    while (static_cast<int>(e.tokens().size()) <= e.seq_len()) e.next_token();
    CHECK(throws([&] { e.next_token(); }));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model.bin> <tokenizer.bin>\n", argv[0]);
        return 2;
    }
    model_path = argv[1];
    tokenizer_path = argv[2];

    std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"same_seed_same_tokens", test_same_seed_same_tokens},
        {"different_seed_diverges", test_different_seed_diverges},
        {"restore_on_fresh_engine_continues_identically", test_restore_on_fresh_engine_continues_identically},
        {"restore_rewinds_same_engine", test_restore_rewinds_same_engine},
        {"restore_with_only_the_prompt", test_restore_with_only_the_prompt},
        {"save_reports_position_and_rng", test_save_reports_position_and_rng},
        {"restore_rejects_bad_state", test_restore_rejects_bad_state},
        {"stops_at_context_limit", test_stops_at_context_limit},
    };
    for (auto& [name, fn] : tests) {
        int before = failures;
        fn();
        std::printf("%s ... %s\n", name, failures == before ? "ok" : "FAIL");
    }
    std::printf("%zu tests, %d failed checks\n", tests.size(), failures);
    return failures == 0 ? 0 : 1;
}
