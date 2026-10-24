PYTHON ?= python3

.PHONY: install test lint self-check

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m compileall -q airigs tests
	@echo "compileall: ok"

# Runs a quick benchmark and validates the payload against the server-required
# field list. Exits non-zero when this host cannot produce an uploadable rig
# (for example: no torch, no NVML). That is the point of the check.
self-check:
	$(PYTHON) -m airigs --self-check
