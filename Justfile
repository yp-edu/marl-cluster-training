default:
  @just --choose

install:
	uv run pre-commit install
	uv sync

checks:
	uv run pre-commit run --all-files

test-assets:
	@echo "No test assets to resolve"

tests:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v

sync-experiments clean="":
	for d in results/experiments/*/wandb/offline-run-*; do \
		uv run wandb sync --sync-all "$$d"; \
		if "{{clean}}" == "clean"; then \
			rm -r "$$d"; \
		fi \
	done

sync-benchmarks clean="":
	for d in results/benchmarks/*/wandb/offline-run-*; do \
		uv run wandb sync --sync-all "$$d"; \
		if "{{clean}}" == "clean"; then \
			rm -r "$$d"; \
		fi \
	done

wandb-sync clean="": (sync-experiments "{{clean}}") (sync-benchmarks "{{clean}}")
	@echo "Done"
