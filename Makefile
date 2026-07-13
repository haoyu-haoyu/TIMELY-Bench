.PHONY: help install-public inspect-results reproduce-synthetic verify-public test public-checks

PYTHON ?= python3

help:
	@echo "TIMELY-Bench public reproducibility commands"
	@echo ""
	@echo "  make install-public       Install lightweight result-inspection dependencies"
	@echo "  make inspect-results      Print the tracked V3 aggregate results"
	@echo "  make reproduce-synthetic  Regenerate and validate the synthetic V3 fixture"
	@echo "  make verify-public        Check the repository's public-release boundary"
	@echo "  make test                 Run public-release and synthetic regression tests"
	@echo "  make public-checks        Run all non-credentialed reproducibility checks"

install-public:
	$(PYTHON) -m pip install -r requirements-public.txt

inspect-results:
	$(PYTHON) tools/summarize_public_results.py

reproduce-synthetic:
	$(PYTHON) synthetic/generate.py --check

verify-public:
	$(PYTHON) tools/verify_public_release.py .

test:
	$(PYTHON) -m unittest discover -s tests/public_release -p 'test_*.py' -v

public-checks: inspect-results reproduce-synthetic verify-public test
	@echo "All public reproducibility checks passed."
