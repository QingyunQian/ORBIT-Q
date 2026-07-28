PYPI_INDEX ?= https://pypi.org/simple
TASK ?= 01
REPEAT ?= 4
.PHONY: setup lint test verify doctor reference compare reference-all compare-all

setup:
	uv sync --index-url $(PYPI_INDEX)

lint:
	ruff check .

test:
	uv run python -m unittest discover -s tests -v

verify:
	./bench verify

doctor:
	./bench env doctor

reference:
	./bench run $(TASK) --solution reference --repeat $(REPEAT) --no-build

compare:
	./bench run $(TASK) --solution optimized --compare-to reference --repeat $(REPEAT) --no-build

reference-all:
	./bench run all --solution reference --repeat $(REPEAT) --no-build

compare-all:
	./bench run all --solution optimized --compare-to reference --repeat $(REPEAT) --no-build
