# Every target is a thin wrapper around a plain command, so the project also works without make:
#   make data    ->  python -m scripts.get_synthea && python -m scripts.get_synthea_sample
#                    && python -m scripts.get_guidelines
#   make ingest  ->  python -m healthrag.ingest
#   make index   ->  python -m scripts.build_index
#   make compare ->  python -m scripts.compare_retrieval
#   make router-labels   ->  python -m scripts.make_router_labels
#   make train-router    ->  python -m scripts.train_router
#   make compare-routers ->  python -m scripts.compare_routers
#   make ask Q="..."     ->  python -m healthrag.ask "..."
#   make test    ->  python -m pytest -q

ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
endif

.PHONY: venv data ingest index compare router-labels train-router compare-routers ask test

venv:
	python3.11 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m scripts.get_synthea
	$(PYTHON) -m scripts.get_synthea_sample
	$(PYTHON) -m scripts.get_guidelines

ingest:
	$(PYTHON) -m healthrag.ingest

index:
	$(PYTHON) -m scripts.build_index

compare:
	$(PYTHON) -m scripts.compare_retrieval

router-labels:
	$(PYTHON) -m scripts.make_router_labels

train-router:
	$(PYTHON) -m scripts.train_router

compare-routers:
	$(PYTHON) -m scripts.compare_routers

ask:
	$(PYTHON) -m healthrag.ask "$(Q)"

test:
	$(PYTHON) -m pytest -q
