.PHONY: help sync test lint format verify-h1 verify-env compare-adaptation run-surgery verify-h9 verify-h10 verify-h11 verify-h12 build-tenet train-smoke profile-h2-local gpu-smoke smoke report clean pod-preflight pod-sync-daemon pod-sync-stop pod-launch pod-wait nebius-preflight nebius-start nebius-stop nebius-s3-pull nebius-mlflow-backfill nebius-up nebius-down nebius-iac-validate

HARNESS ?= harness
POD_HOST ?=
POD_WORKDIR ?= ~/Small-Model-Lab
POD_SESSION ?= sml_pod
POD_LOCAL_DIR ?= $(CURDIR)
POD_PATHS ?= results,releases
POD_MIN_VRAM_MIB ?= 6144
POD_MAX_WAIT_SEC ?= 180
POD_CMD ?=
POD_ON_CHECKPOINT ?=

help:
	@echo "Small Model Lab — Available Commands:"
	@echo "  make sync               Install dependencies via uv (Python 3.12)"
	@echo "  make test               Run full unit & integration test suite via pytest"
	@echo "  make lint               Run ruff linter and format check"
	@echo "  make format             Auto-format codebase via ruff"
	@echo "  make verify-h1          Run H1 GPT-2 124M mathematical & causal equivalence verification"
	@echo "  make verify-env         Run H5 StatefulToolEnv & Gated Rubric anti-hacking verification"
	@echo "  make compare-adaptation Run H6 Context vs. Weights adaptation comparison (7 strategies)"
	@echo "  make run-surgery        Run H7/H8 CKA mixed-precision, rank-1 subspace & Pareto verification"
	@echo "  make verify-h9          Run H9 Lancet bi-projected surgery verification (nano/LLM)"
	@echo "  make verify-h10         Run H10 unslop GRPO verification (gated reward shaping)"
	@echo "  make verify-h11         Run H11 OpenRouter FP8 provider readiness & HF Hub verification"
	@echo "  make verify-h12         Run H12 Gram-Schmidt subspace-safe merge (TIES biprojected) verification"
	@echo "  make build-tenet        Build Tenet release bundle (GRPO-tuned weights + readiness report)"
	@echo "  make train-smoke        Run end-to-end M2+M3 local pretraining smoke test (MPS/CUDA/CPU)"
	@echo "  make profile-h2-local   Run 5-stage MFU throughput waterfall locally (MPS/CPU)"
	@echo "  make gpu-smoke          Dry-run the full M4-M5 GPU script in fast smoke mode"
	@echo "  make report             Compile markdown report tables from results/"
	@echo "  make pod-preflight      Run harness pod preflight (VRAM/disk/ETA budget)"
	@echo "  make pod-sync-daemon    Start local 60s rsync sync daemon (harness pod sync)"
	@echo "  make pod-sync-stop      Stop the local sync daemon (harness pod sync --stop)"
	@echo "  make pod-launch         Launch a pod command via harness pod launch"
	@echo "  make pod-wait           Wait for pod checkpoint via harness pod wait"
	@echo "  make nebius-preflight   Run Nebius GPU VM & storage preflight checks"
	@echo "  make nebius-start       Start the Nebius GPU instance"
	@echo "  make nebius-stop        Stop the Nebius GPU instance (\$0/hr GPU billing)"
	@echo "  make nebius-s3-pull     Sync results/ down from Nebius Object Storage"
	@echo "  make nebius-mlflow-backfill  Backfill MLflow history from local runs"
	@echo "  make nebius-up          Provision Nebius IaC (Terraform) & bootstrap VM (Ansible)"
	@echo "  make nebius-down        Stop Nebius GPU VM via Terraform / nebius_ctl.sh"
	@echo "  make nebius-iac-validate  Validate Terraform & Ansible IaC definitions"

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

verify-h1:
	uv run sml verify-h1 --experiment experiments/h1_gpt2_parity.yaml --output-dir results --run-id h1_gpt2_parity

verify-env:
	uv run sml verify-env --experiment experiments/h5_env_verification.yaml --output-dir results --run-id h5_env_verification

compare-adaptation:
	uv run sml compare-adaptation --experiment experiments/h6_adaptation_comparison.yaml --output-dir results --run-id h6_adaptation_comparison

run-surgery:
	uv run sml run-surgery --experiment experiments/h7_h8_surgery_pareto.yaml --output-dir results --run-id h7_h8_surgery_pareto

verify-h9:
	uv run sml verify-h9 --experiment experiments/h9_biprojected_surgery.yaml --output-dir results --run-id h9_biprojected_surgery

verify-h10:
	uv run sml verify-h10 --experiment experiments/h10_unslop_grpo.yaml --output-dir results --run-id h10_unslop_grpo

verify-h11:
	uv run sml verify-h11 --experiment experiments/h11_provider_fp8_readiness.yaml --output-dir results --run-id h11_provider_fp8_readiness

verify-h12:
	uv run sml verify-h12 --experiment experiments/h12_subspace_safe_merge.yaml --output-dir results --run-id h12_subspace_safe_merge

build-tenet:
	uv run sml build-tenet --output-dir results/tenet_releases --release-id Tenet-3B-Qwen2.5-Unslop-v1

materialize-tenet:
	uv run sml materialize-tenet --release-id Tenet-3B-Qwen2.5-Unslop-v1 --output-dir results/tenet_materialized --device mps

train-smoke:
	uv run sml train-smoke --steps 20 --device auto

profile-h2-local:
	uv run sml profile-h2 --device auto --seq-len 256 --micro-batch-size 2 --warmup-steps 1 --measure-steps 2 --run-id smoke_h2_local

gpu-smoke:
	SMOKE_ONLY=1 START_WATCHDOG=0 SHARD_DIR=shards/smoke_gpu RESULTS_DIR=results/smoke_gpu REPORT_PATH=reports/sprint1/smoke_gpu.md bash ./scripts/remote/run_m4_m5_gpu.sh

smoke:
	uv run sml dry-run --experiment experiments/h3_wsd_100m.yaml --output-dir results/dry_run_m0
	uv run sml report --results-dir results/dry_run_m0 --output reports/sprint1/dry_run_preview.md --include-dry-runs

pod-preflight:
	@test -n "$(POD_HOST)" || { echo "POD_HOST is required (make pod-preflight POD_HOST=<host>)"; exit 1; }
	$(HARNESS) pod preflight --host "$(POD_HOST)" --workdir "$(POD_WORKDIR)" --min-free-vram-mib "$(POD_MIN_VRAM_MIB)"

pod-sync-daemon:
	@test -n "$(POD_HOST)" || { echo "POD_HOST is required (make pod-sync-daemon POD_HOST=<host>)"; exit 1; }
	$(HARNESS) pod sync --host "$(POD_HOST)" --remote-dir "$(POD_WORKDIR)" --local-dir "$(POD_LOCAL_DIR)" --paths "$(POD_PATHS)" --daemon --interval 60

pod-sync-stop:
	@test -n "$(POD_HOST)" || { echo "POD_HOST is required (make pod-sync-stop POD_HOST=<host>)"; exit 1; }
	$(HARNESS) pod sync --host "$(POD_HOST)" --remote-dir "$(POD_WORKDIR)" --local-dir "$(POD_LOCAL_DIR)" --paths "$(POD_PATHS)" --stop

pod-launch:
	@test -n "$(POD_HOST)" || { echo "POD_HOST is required (make pod-launch POD_HOST=<host> POD_CMD=...)"; exit 1; }
	@test -n "$(POD_CMD)" || { echo "POD_CMD is required (make pod-launch POD_HOST=<host> POD_CMD=...)"; exit 1; }
	$(HARNESS) pod launch --host "$(POD_HOST)" --workdir "$(POD_WORKDIR)" --session "$(POD_SESSION)" --stagger-vram-mib "$(POD_MIN_VRAM_MIB)" $(if $(POD_ON_CHECKPOINT),--on-checkpoint "$(POD_ON_CHECKPOINT)",) -- $(POD_CMD)

pod-wait:
	@test -n "$(POD_HOST)" || { echo "POD_HOST is required (make pod-wait POD_HOST=<host>)"; exit 1; }
	$(HARNESS) pod wait --host "$(POD_HOST)" --workdir "$(POD_WORKDIR)" --max-wait-sec "$(POD_MAX_WAIT_SEC)" --local-dir "$(POD_LOCAL_DIR)"

nebius-preflight:
	@bash -c 'set -a; [ -f .env.nebius ] && source .env.nebius; set +a; bash scripts/remote/nebius_ctl.sh preflight'

nebius-start:
	@bash -c 'set -a; [ -f .env.nebius ] && source .env.nebius; set +a; bash scripts/remote/nebius_ctl.sh start'

nebius-stop:
	@bash -c 'set -a; [ -f .env.nebius ] && source .env.nebius; set +a; bash scripts/remote/nebius_ctl.sh stop'

nebius-s3-pull:
	@bash -c 'set -a; [ -f .env.nebius ] && source .env.nebius; set +a; bash scripts/remote/nebius_ctl.sh s3-sync-down'

nebius-mlflow-backfill:
	@bash -c 'set -a; [ -f .env.nebius ] && source .env.nebius; set +a; uv run python scripts/hub/backfill_mlflow_history.py'

nebius-up:
	cd infra/terraform/nebius && terraform init && terraform apply -auto-approve
	cd infra/ansible && ansible-playbook playbook.yml

nebius-down:
	@if [ -f infra/terraform/nebius/terraform.tfstate ]; then \
		cd infra/terraform/nebius && terraform apply -auto-approve -var="instance_stopped=true"; \
	else \
		bash -c 'set -a; [ -f .env.nebius ] && source .env.nebius; set +a; bash scripts/remote/nebius_ctl.sh stop'; \
	fi

nebius-iac-validate:
	cd infra/terraform/nebius && terraform fmt -check -recursive && terraform init -backend=false && terraform validate
	cd infra/ansible && ansible-playbook -i localhost, --syntax-check playbook.yml

report:
	uv run sml report --results-dir results --output reports/sprint1/summary.md

clean:
	rm -rf results/dry_run_* results/smoke_* shards/smoke_* reports/sprint1/dry_run_preview.md reports/sprint1/smoke_gpu.md .pytest_cache .ruff_cache
