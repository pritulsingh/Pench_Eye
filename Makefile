# Pench Eye — Developer Commands
# Usage: make <target>

.PHONY: help up down logs backend frontend seed reseed test test-reid typecheck build-frontend migrate clean reid-train reid-eval reid-calibrate reid-embeddings

help:
	@echo "Pench Eye — Available Commands"
	@echo "=============================================="
	@echo "  make up         Start all services (Docker Compose)"
	@echo "  make down       Stop all services"
	@echo "  make logs       Tail logs from all services"
	@echo "  make seed       Seed demo data (skips if already seeded)"
	@echo "  make reseed     Wipe demo rows and seed again"
	@echo "  make test       Run backend tests"
	@echo "  make test-reid  Run Re-ID dataset/model/training tests only"
	@echo "  make typecheck  Type-check the frontend"
	@echo "  make migrate    Run Alembic migrations (PostgreSQL deployments)"
	@echo "  make clean      Remove Docker volumes (WARNING: destroys data)"
	@echo ""
	@echo "Local development (no Docker, SQLite by default):"
	@echo "  make backend    Start backend dev server on :8000"
	@echo "  make frontend   Start frontend dev server on :5173"
	@echo ""
	@echo "Tiger Re-ID (see docs/reid_training.md):"
	@echo "  make reid-train      DATA=data/reid [OUT=ml/weights/tiger_reid] [DEVICE=cuda]"
	@echo "  make reid-eval       DATA=data/reid [SPLIT=test]"
	@echo "  make reid-calibrate  DATA=data/reid [SPLIT=val]"

up:
	docker compose up -d
	@echo "Dashboard: http://localhost:5173"
	@echo "API docs:  http://localhost:8000/docs"
	@echo "MinIO:     http://localhost:9001"

down:
	docker compose down

logs:
	docker compose logs -f

backend:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

seed:
	python scripts/seed_demo_data.py

reseed:
	python scripts/seed_demo_data.py --reset

test:
	cd backend && python -m pytest tests/ -v --tb=short

test-reid:
	cd backend && python -m pytest tests/test_reid_dataset.py tests/test_reid_model.py tests/test_reid_training.py -v --tb=short

# ── Tiger Re-ID (docs/reid_training.md) ──────────────────────
DATA ?= data/reid
OUT ?= ml/weights/tiger_reid
DEVICE ?= auto
SPLIT ?= test

reid-train:
	python -m ml.reid.train --data $(DATA) --output $(OUT) --device $(DEVICE)

reid-eval:
	python -m ml.reid.evaluate --checkpoint $(OUT)/best.pt --data $(DATA) --split $(SPLIT) --device $(DEVICE) --roc

reid-calibrate:
	python -m ml.reid.calibrate_thresholds --checkpoint $(OUT)/best.pt --data $(DATA) --split val --device $(DEVICE) --output $(OUT)/thresholds.json

reid-embeddings:
	python -m ml.reid.extract_embeddings --checkpoint $(OUT)/best.pt --input $(DATA) --output $(OUT)/embeddings.parquet --device $(DEVICE) --include-quality

typecheck:
	cd frontend && npx tsc --noEmit

build-frontend:
	cd frontend && npm run build

migrate:
	cd backend && alembic upgrade head

clean:
	docker compose down -v
	@echo "All Docker volumes removed."
