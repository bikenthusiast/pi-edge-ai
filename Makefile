# Targets are grouped by WHERE they run. Everything above `deploy` is Mac-only.
.PHONY: help check-python setup models test lint bench bench-pi deploy deploy-dry run-pi test-pi pipeline pipeline-mqtt mqtt-watch report ssh clean

PY      ?= python3.12
PI_HOST ?= pi
PI_PATH ?= ~/pi-edge-ai
export PI_HOST PI_PATH

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

######################
#  Workflow for Mac  #
######################

check-python:
	@command -v $(PY) >/dev/null 2>&1 || { \
	  printf 'Error: %s not found.\n  uv python install 3.12\n  make setup PY=$$(uv python find 3.12)\n' '$(PY)'; \
	  exit 1; }
	@$(PY) -c "import sys,platform; \
	  sys.version_info >= (3,11) or sys.exit(f'Error: need Python >= 3.11, got {platform.python_version()} at {sys.executable}'); \
	  print(f'Using Python {platform.python_version()} ({platform.machine()})')"

setup: check-python  ## create the dev venv on the Mac
	$(PY) -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements-dev.txt -e .
	@.venv/bin/python -c "import platform; \
	  print('Note: x86_64 — LiteRT inference tests will skip, they run on the Pi.') \
	  if platform.machine() != 'arm64' else \
	  print('arm64 — optional: pip install -r requirements-mac-extra.txt')" 

models:  ## download model weights (not in git)
	bash scripts/fetch_models.sh

test:  ## run the Mac-side test suite (hardware tests skipped)
	.venv/bin/python -m pytest -q

lint: 
	.venv/bin/ruff check src tests

bench:  ## compare all models in models/ (needs LiteRT — use bench-pi on Intel Macs)
	@.venv/bin/python -c "import ai_edge_litert" 2>/dev/null || { \
	  echo "LiteRT is unavailable here (macOS x86_64 has no wheels)."; \
	  echo "Run the comparison on the Pi instead:  make bench-pi"; \
	  exit 1; }
	.venv/bin/python -m edge.vision.benchmark models/*.tflite --runs 20


# --- Mac -> Pi -------------------------------------------------------------
deploy-dry:  ## show what deploy would copy, without copying
	bash scripts/deploy.sh --dry-run --itemize-changes

deploy: test  ## sync src/ + models/ to the Pi (runs tests first)
	bash scripts/deploy.sh

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache

###############################
#  Workflow for Raspberry Pi  #
###############################

bench-pi:  ## compare all models in models
	ssh $(PI_HOST) 'cd $(PI_PATH) && .venv/bin/python -m edge.vision.benchmark models/*.tflite --runs 20'

run-pi:  ## run the classifier on the Pi over SSH
	ssh $(PI_HOST) 'cd $(PI_PATH) && .venv/bin/python -m edge.vision.classify samples/parrot.jpg'

test-pi:  ## run the full suite ON the Pi, including hardware tests
	ssh $(PI_HOST) 'cd $(PI_PATH) && .venv/bin/python -m pytest -q -m "not hardware"'
	ssh $(PI_HOST) 'cd $(PI_PATH) && .venv/bin/python -m pytest -q -m hardware'

pipeline:  ## run the live pipeline ON the Pi for 60 s
	ssh $(PI_HOST) 'cd $(PI_PATH) && .venv/bin/python -m edge.vision.pipeline --max-seconds 60'

pipeline-mqtt:  ## run the live pipeline ON the Pi for 60 s and publish via MQTT
	ssh $(PI_HOST) 'cd $(PI_PATH) && .venv/bin/python -m edge.vision.pipeline --max-seconds 60 --mqtt-host localhost'

mqtt-watch:  ## follow all edge/# MQTT messages on the Pi's broker
	ssh -t $(PI_HOST) 'mosquitto_sub -h localhost -t "edge/#" -v'

report:  ## show the last 24 h of events from the Pi
	ssh $(PI_HOST) 'cd $(PI_PATH) && .venv/bin/python -m edge.events.report --hours 24'

ssh:
	ssh $(PI_HOST)