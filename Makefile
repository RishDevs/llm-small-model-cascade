PY ?= .venv/bin/python
CONFIG ?= configs/default.yaml
C = --config $(CONFIG)

.PHONY: setup label label-local label-free train export evaluate bench serve test pipeline

setup:
	uv venv --python 3.12
	uv pip install -e ".[dev]"

label:
	$(PY) -m cascade.teacher $(C) --backend vllm

label-local:
	$(PY) -m cascade.teacher $(C) --backend http

label-free:
	$(PY) -m cascade.teacher $(C) --mode free --splits train --limit 2000

train:
	$(PY) -m cascade.train $(C)

export:
	$(PY) -m cascade.export $(C)

evaluate:
	$(PY) -m cascade.evaluate $(C)

bench:
	$(PY) -m cascade.bench $(C)

serve:
	CASCADE_CONFIG=$(CONFIG) $(PY) -m uvicorn cascade.serve:build_app --factory --port 8000

test:
	$(PY) -m pytest -q

pipeline: train export evaluate bench
