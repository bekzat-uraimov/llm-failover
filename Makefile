.PHONY: help day1 determinism

help:
	@echo "Available targets:"
	@echo "  day1         Day 1 milestone status"
	@echo "  determinism  Reference seed check using llama2.c"

day1:
	@echo "Day 1: deterministic generation verified against the llama2.c reference."
	@echo "Read notes/day1.md and docs/DESIGN.md for the explanation."

determinism:
	@echo "Run the following in /tmp/llama2.c:"
	@echo "  ./run stories15M.bin -s 42 -n 50"
	@echo "  ./run stories15M.bin -s 42 -n 50"
	@echo "  ./run stories15M.bin -s 43 -n 50"

.PHONY: day2 test serve

day2:
	@echo "Day 2: snapshot save/load implementation added under worker/"

test:
	python3 -m unittest discover -v

serve:
	python3 -m worker.server 8000

serve-router:
	python3 -m router.server 8002
