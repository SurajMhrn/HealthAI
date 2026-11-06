# Every target is a thin wrapper around a plain command, so the project also works without make:
#   make data    ->  python -m scripts.get_synthea && python -m scripts.get_synthea_sample
#                    && python -m scripts.get_guidelines
#   make ingest  ->  python -m healthrag.ingest
#   make test    ->  python -m pytest -q

ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
endif

.PHONY: venv data ingest test

venv:
	python3.11 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m scripts.get_synthea
	$(PYTHON) -m scripts.get_synthea_sample
	$(PYTHON) -m scripts.get_guidelines

ingest:
	$(PYTHON) -m healthrag.ingest

test:
	$(PYTHON) -m pytest -q
