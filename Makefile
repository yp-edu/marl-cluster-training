.PHONY: install
install:
	uv run pre-commit install
	uv sync

.PHONY: checks
checks:
	uv run pre-commit run --all-files

.PHONY: test-assets
test-assets:
	@echo "No test assets to resolve"

.PHONY: tests
tests:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v

.PHONY: wandb-sync
wandb-sync:
	uv run --no-sync wandb sync results/experiments/*/wandb/offline-run-*
	uv run --no-sync wandb sync results/benchmarks/*/wandb/offline-run-*

.PHONY: wandb-sync-and-clean
wandb-sync-and-clean:
	for d in results/experiments/*/wandb/offline-run-*; do \
		uv run wandb sync --sync-all "$$d"; \
		rm -r "$$d"; \
	done
	for d in results/benchmarks/*/wandb/offline-run-*; do \
		uv run wandb sync --sync-all "$$d"; \
		rm -r "$$d"; \
	done
