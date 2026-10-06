.PHONY: test worker router

test:
	python3 -m unittest discover -v

worker:
	python3 -m worker.server 8001

router:
	python3 -m router.server 8002
