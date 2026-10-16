# No -ffast-math: it lets the compiler reorder float math, and replay has to be bit-exact
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

MODEL = models/stories15M.bin
TOKENIZER = models/tokenizer.bin
PORT = 9001

.PHONY: all model test worker router clean

all: build/worker build/test_engine

build/worker: worker/worker.cpp worker/json.h engine/engine.cpp engine/engine.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ worker/worker.cpp engine/engine.cpp

build/test_engine: engine/test_engine.cpp engine/engine.cpp engine/engine.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ engine/test_engine.cpp engine/engine.cpp

model: $(MODEL) $(TOKENIZER)

$(MODEL):
	@mkdir -p models
	curl -fL -o $@ https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

$(TOKENIZER):
	@mkdir -p models
	curl -fL -o $@ https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin

test: all model
	./build/test_engine $(MODEL) $(TOKENIZER)
	python3 -m unittest discover -v

worker: build/worker model
	./build/worker --port $(PORT)

router: model
	python3 -m router.router --worker 127.0.0.1:9001 --worker 127.0.0.1:9002

clean:
	rm -rf build
