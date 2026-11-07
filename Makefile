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
#   make eval-guard      ->  python -m scripts.eval_guardrails
#   make golden          ->  python -m scripts.make_golden
#   make eval-run        ->  python -m eval.run            (all five configs, judge-free metrics)
#   make eval-ragas      ->  python -m eval.ragas_run      (needs requirements-eval.txt)
#   make eval-report     ->  python -m eval.report
#   make eval            ->  regression config + RAGAS + python -m eval.regress
#   make test    ->  python -m pytest -q

ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
endif

.PHONY: venv data ingest index compare router-labels train-router compare-routers ask eval-guard golden eval-run eval-ragas eval-report eval test

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

eval-guard:
	$(PYTHON) -m scripts.eval_guardrails

golden:
	$(PYTHON) -m scripts.make_golden

eval-run:
	$(PYTHON) -m eval.run

eval-ragas:
	$(PYTHON) -m eval.ragas_run

eval-report:
	$(PYTHON) -m eval.report

eval:
	$(PYTHON) -m eval.run --config rerank_tiered
	$(PYTHON) -m eval.ragas_run --config rerank_tiered
	$(PYTHON) -m eval.regress

test:
	$(PYTHON) -m pytest -q
