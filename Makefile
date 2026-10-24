# No -ffast-math: it lets the compiler reorder float math, and replay has to be bit-exact
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

MODEL = models/stories15M.bin
TOKENIZER = models/tokenizer.bin
# Identical weights on every worker is one of the determinism assumptions, so check them
MODEL_SHA256 = cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a
TOKENIZER_SHA256 = 50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361
PORT = 9001
# llama2.c's own binary, built from a pinned commit, is the reference our output is checked against
LLAMA2C_COMMIT = 350e04fe35433e6d2941dce5a1f53308f87058eb

.PHONY: all model reference test demo worker router clean

all: build/worker build/test_engine

build/worker: worker/worker.cpp worker/json.h engine/engine.cpp engine/engine.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ worker/worker.cpp engine/engine.cpp

build/test_engine: engine/test_engine.cpp engine/engine.cpp engine/engine.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ engine/test_engine.cpp engine/engine.cpp

model: $(MODEL) $(TOKENIZER)
	@printf '%s  %s\n%s  %s\n' $(MODEL_SHA256) $(MODEL) $(TOKENIZER_SHA256) $(TOKENIZER) | shasum -a 256 -c --quiet

$(MODEL):
	@mkdir -p models
	curl -fL -o $@ https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

$(TOKENIZER):
	@mkdir -p models
	curl -fL -o $@ https://raw.githubusercontent.com/karpathy/llama2.c/$(LLAMA2C_COMMIT)/tokenizer.bin

reference: build/llama2.c/run

build/llama2.c/run:
	@mkdir -p build
	rm -rf build/llama2.c
	git clone -q https://github.com/karpathy/llama2.c build/llama2.c
	git -C build/llama2.c checkout -q $(LLAMA2C_COMMIT)
	$(MAKE) -C build/llama2.c run

test: all model reference
	./build/test_engine $(MODEL) $(TOKENIZER)
	python3 -m unittest discover -v

demo: all model
	python3 scripts/demo.py

worker: build/worker model
	./build/worker --port $(PORT)

router: model
	python3 -m router.router --worker 127.0.0.1:9001 --worker 127.0.0.1:9002

clean:
	rm -rf build
