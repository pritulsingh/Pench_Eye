# Pench Eye
## Wildlife Intelligence Platform for Pench Tiger Reserve
### Camera-trap triage · individual tiger identification · geospatial monitoring · alerting · analytics

> **Demo notice.** Out of the box this project runs in **demo mode**: inference is
> deterministic and simulated, and reserve boundaries are hand-authored
> approximations. Nothing here is a validated tiger-identification model or
> surveyed GIS data, and the UI labels simulated content everywhere it appears.

---

## What is Pench Eye?

Camera-trap networks generate far more frames than teams can review. Most frames are
empty, and the useful ones still need an ecologist to decide *which* tiger was
photographed. Pench Eye is one application that closes that loop:

1. Frames arrive (upload, or a simulated capture in demo mode).
2. Triage removes blanks and duplicates so reviewers only see useful frames.
3. Detection identifies whether an animal is present, and which species.
4. Identification proposes an individual tiger, or routes the frame to human review.
5. The resulting sighting is written to the database with camera, zone and coordinates.
6. Alert rules evaluate the sighting and the health of the camera network.
7. Map, analytics and dashboard all read from those same rows.

Everything is served by a single FastAPI backend and a single React frontend.

## Architecture

```text
        Camera trap frame  ──►  POST /api/v1/images/upload
        (or POST /api/v1/demo/simulate for a synthetic frame)
                                     │
      ┌──────────────────────────────▼───────────────────────────────┐
      │  PipelineService  (backend/app/services/pipeline_service.py) │
      │                                                              │
      │  validate ─► decode/hash ─► triage ─► detect ─► identify      │
      │                                 │        │          │        │
      │                                 └────────┴──────────┘        │
      │                                   InferencePipeline           │
      │                        DemoInference | ProductionInference    │
      └────────┬─────────────────────────────────────────────────────┘
               │ writes
      ┌────────▼───────────────────────────────────────────────────┐
      │  Database:  camera_stations · zones · images · observations │
      │             tigers · embeddings · review_queue · alerts     │
      └────────┬───────────────────────┬──────────────┬────────────┘
               │                       │              │
        MapService              AnalyticsService   AlertService
               │                       │              │
      ┌────────▼───────────────────────▼──────────────▼────────────┐
      │  REST API  /api/v1/{map,analytics,alerts,cameras,tigers,…}  │
      └────────────────────────────┬───────────────────────────────┘
                                   │
      ┌────────────────────────────▼───────────────────────────────┐
      │  React + Vite frontend                                     │
      │  Command Center · Reserve Map (Leaflet) · Cameras · Tigers  │
      │  Detections · Gallery · Alerts · Analytics · Review · Demo  │
      └────────────────────────────────────────────────────────────┘
```

Image bytes go to MinIO/S3 when reachable and to `storage/` on the local filesystem
otherwise; both are served through `GET /api/v1/images/{id}/file`.

## Features

**Command center** — camera count and health, detections (total and last 7 days),
identified individuals, open alerts, ingestion and blank-filter counters, storage
recoverable, mean identity confidence, a 14-day detection trend, images per camera,
detections by zone, most active tigers, recent identifications, active alerts and the
latest captures.

**Reserve map** — Leaflet map with toggleable layers for the reserve boundary, core
zone, buffer zones, corridor and village-interface belt, tourism gates, camera markers
coloured by state (active / detection <24 h / warning / offline / maintenance), tiger
and other-wildlife sightings, and movement paths between consecutive detections.
Camera popups carry ID, name, zone, state, last activity, last detection, detection
count and open alerts; sighting popups carry tiger, timestamp, camera, coordinates,
confidence and the frame itself. Filter by individual tiger and by period.

**Camera traps** — filterable list (zone, status, free text) with marker state,
battery, last seen, last detection and detection count. Detail pages add a location
map, a detection timeline, recent detections and an image gallery.

**Tiger identification and profiles** — catalogue with per-individual detections,
camera count and mean confidence; profiles show first/last detection, frequently used
cameras, zone distribution, detections per month, a gallery, a chronological timeline
and a movement map with per-leg distance and elapsed time.

**Detections** — every sighting with server-side filters for camera, tiger, zone,
species, minimum confidence, period and explicit date range, plus pagination.

**Image gallery** — grid of captures filterable by camera, tiger, species, status and
date, with a detail dialog showing full metadata.

**Triage** — current blank/subject/quarantine counts, quality distribution, blank
frames by camera, and a quarantine queue where frames can be restored or deleted.

**Human review** — queue of ambiguous identities with candidate matches and scores;
confirm a candidate, create a new individual, or reject the detection.

**Alerts** — five rules: tiger in a village-adjacent zone (critical), camera offline
past a configurable threshold, unusual movement rate between cameras, abnormally high
24-hour activity, and low identity confidence. Alerts carry severity, type, timestamp,
location and message, and can be acknowledged, resolved or reopened. Rule evaluation
is idempotent (deduplicated).

**Analytics** — detections over time (all/tiger/blank), by camera, by hour, by
weekday, by zone, species distribution, identity-confidence distribution, most
frequent camera-to-camera movements, and most detected individuals.

**Demo mode** — generates a synthetic frame for a real camera row and pushes it
through the same pipeline, then shows the full trace: capture → triage → detection →
identification → sighting → alerts, with links to the map, tiger profile, alerts and
analytics.

## Tech stack

| Layer     | Technology |
|-----------|------------|
| Backend   | FastAPI, SQLAlchemy 2 (async), Pydantic v2 |
| Database  | PostgreSQL + PostGIS + pgvector under Docker; SQLite locally |
| Storage   | MinIO / S3, with local-filesystem fallback |
| ML        | Pluggable `InferencePipeline`; OpenCV / Torch / Ultralytics in production mode |
| Frontend  | React 18, TypeScript, Vite, Tailwind, React Router, Leaflet, Recharts |
| Tooling   | Docker Compose, Make, pytest |

## Project structure

```text
pench-eye/
├── backend/
│   ├── app/
│   │   ├── api/v1/            # images, triage, tigers, observations, reviews,
│   │   │                      # cameras, dashboard, search, map, alerts,
│   │   │                      # analytics, demo
│   │   ├── core/              # config, database, portable column types,
│   │   │                      # geo reference data, demo seeding, storage
│   │   ├── models/            # camera_station, zone, image, observation, tiger,
│   │   │                      # embedding, review_queue, alert, triage_run
│   │   ├── schemas/           # Pydantic request/response models
│   │   ├── services/          # pipeline, inference, reid_gallery, map, alert,
│   │   │                      # analytics, camera, tiger, observation, review,
│   │   │                      # simulation
│   │   └── ingestion/         # frame abstraction + directory ingestion
│   ├── alembic/               # migrations (PostgreSQL deployments)
│   └── tests/                 # API, pipeline, triage, re-id dataset/model/training
├── frontend/src/
│   ├── api/client.ts          # typed API client
│   ├── components/            # layout + UI primitives
│   ├── features/map/          # Leaflet map component and marker config
│   ├── hooks/useApi.ts        # loading / error / reload fetch helper
│   ├── pages/                 # one file per route
│   └── types/                 # shared API types
├── ml/
│   ├── detection/             # animal detector
│   ├── triage/                # blank-frame classifier
│   ├── demo/                  # deterministic demo reference data
│   ├── reid/                  # individual identification
│   │   ├── dataset/           # discovery, sequence-safe splitting, DataLoaders
│   │   ├── preprocessing.py   # shared train/inference transform
│   │   ├── augmentation.py    # stripe-safe augmentation
│   │   ├── model.py           # backbone → 512-d embedding + ArcFace head
│   │   ├── losses.py          # ArcFace CE + batch-hard triplet
│   │   ├── metrics.py         # Rank-k, mAP, similarity stats, ROC
│   │   ├── quality.py         # crop / match reliability gating
│   │   ├── checkpoint.py      # self-describing checkpoint format
│   │   ├── train.py           # training CLI
│   │   ├── evaluate.py        # evaluation CLI
│   │   ├── calibrate_thresholds.py
│   │   └── extract_embeddings.py
│   └── weights/               # trained checkpoints (git-ignored)
├── data/reid_example/         # dataset layout documentation (no images)
├── docs/reid_training.md      # Re-ID training guide
├── scripts/                   # demo seeding, sample image generation
├── storage/                   # local image storage + SQLite database
├── docker-compose.yml
├── Makefile
└── .env.example
```

## Setup

Requirements: Python 3.11+ and Node 18+ for local development, or Docker for the full
stack.

```bash
cp .env.example .env          # optional — sensible defaults are built in
```

### Running locally (no infrastructure required)

The backend defaults to SQLite at `storage/pench_eye.db` and to local-filesystem
image storage, and seeds the demo dataset on first start.

```bash
# Backend  → http://localhost:8000  (docs at /docs)
cd backend
python -m venv ../.venv
../.venv/Scripts/pip install -r requirements.txt     # Windows
# ../.venv/bin/pip install -r requirements.txt       # macOS / Linux
uvicorn app.main:app --reload --port 8000

# Frontend → http://localhost:5173
cd ../frontend
npm install
npm run dev
```

Vite proxies `/api` to `http://localhost:8000`, so no frontend configuration is needed
in development.

### Running with Docker

```bash
docker compose up -d
```

This starts PostgreSQL (PostGIS + pgvector), MinIO, the backend and the frontend.
Dashboard on `:5173`, API docs on `:8000/docs`, MinIO console on `:9001`.

### Make targets

```bash
make backend     # backend dev server
make frontend    # frontend dev server
make seed        # seed demo data (no-op if already seeded)
make reseed      # delete demo rows and seed again
make test        # backend test suite
make typecheck   # frontend type check
make up / down / logs / clean
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` / `DATABASE_URL_SYNC` | SQLite in `storage/` | Database connection; Compose supplies PostgreSQL |
| `ML_MODE` | `demo` | `demo` = simulated inference, `production` = real models (Re-ID checkpoint required) |
| `REID_CHECKPOINT_PATH` | empty | Trained Re-ID checkpoint; empty probes `ml/weights/tiger_reid/best.pt` |
| `BLANK_THRESHOLD` | `0.95` | Blank-frame quarantine threshold |
| `AUTO_MATCH_THRESHOLD` | `0.90` | Accept an identity automatically — **placeholder, calibrate it** |
| `REVIEW_THRESHOLD` | `0.75` | Route to human review above this — **placeholder, calibrate it** |
| `NEW_INDIVIDUAL_THRESHOLD` | `0.60` | Below this, treat as a new individual — **placeholder, calibrate it** |
| `CAMERA_OFFLINE_HOURS` | `48` | Camera-offline alert threshold |
| `HIGH_ACTIVITY_DETECTIONS_PER_DAY` | `5` | High-activity alert threshold |
| `LOW_CONFIDENCE_THRESHOLD` | `0.80` | Low-confidence alert threshold |
| `MAX_UPLOAD_BYTES` | `15728640` | Upload size limit |
| `ALLOWED_UPLOAD_EXTENSIONS` | `.jpg,.jpeg,.png` | Accepted upload types |
| `GEO_DATA_SOURCE` | `demo` | Set to `official` once real GIS layers are installed |
| `MINIO_*` | see `.env.example` | Object storage; falls back to local disk |
| `CORS_ORIGINS` | localhost dev origins | Allowed browser origins |
| `SECRET_KEY` | `changeme…` | Replace before any deployment |
| `VITE_API_URL` | empty | Leave empty in dev; Vite proxies `/api` |

Secrets belong in `.env`, which is git-ignored. `.env.example` documents every
variable and contains no real credentials.

## Demo mode

```text
POST /api/v1/demo/simulate  { "camera_id": "CAM-004", "count": 1 }
```

A synthetic night-IR style frame is generated locally, attributed to a real camera
row, and run through the production code path. Every row it creates is flagged
`is_demo=true`. The Demo Mode page renders the resulting trace step by step, and the
banner, sidebar and `/api/v1/demo/status` endpoint all report that demo inference is
active.

To leave demo mode: set `ML_MODE=production`, install model weights under
`ml/weights/`, and restart. `ProductionInference` then drives the real detector and
blank classifier, falling back to the deterministic path per-stage if a model or
checkpoint is missing.

## ML pipeline

`backend/app/services/inference_service.py` defines the whole ML contract:

```python
triage(image_hash, width, height)        -> TriageOutput
detect(image_hash)                       -> DetectionOutput
identify(image_hash, known_tiger_codes)  -> IdentityOutput
```

`DemoInference` implements all three deterministically. `ProductionInference`
overrides `triage_frame` and `detect_frame` with pixel-based paths that delegate to
`ml/triage/blank_classifier.py` and `ml/detection/tiger_detector.py`, and
`identify_frame` with the trained Re-ID encoder. Replacing a model means
implementing this interface — no API, schema, or UI change is required.

Triage and detection degrade to the deterministic path when their weights are
missing. **Identification does not.** In production mode without a trained
checkpoint, `identify_frame` raises `ReIDUnavailable`, the detection is recorded
with `decision="identity_unavailable"` and queued for human review, and
`/health` reports `reid_available: false`. The API never presents a simulated
embedding as a real identification.

Identity decisions use the configured thresholds: at or above `AUTO_MATCH_THRESHOLD`
the sighting is auto-matched; between that and `REVIEW_THRESHOLD` it enters the
human-review queue with ranked candidates; below `NEW_INDIVIDUAL_THRESHOLD` a new
individual is created. Embeddings live in `embeddings` — `vector(512)` with pgvector,
JSON-encoded on SQLite. `ml/reid/quality.py` additionally flags small or blurred
crops, flank mismatches, sparse galleries and ambiguous top-2 scores, and can
downgrade an auto-match to human review.

## Training a real tiger Re-ID model

The stripe-based individual identification model is **not shipped trained** —
there are no weights in this repository. The full training, evaluation and
calibration pipeline is implemented and tested, and becomes a working identifier
only once you supply identity-labelled tiger flank crops.

```text
labelled data → dataset preparation → training → evaluation
    → threshold calibration → best.pt → ML_MODE=production
    → real 512-d embeddings → pgvector cosine search
    → IdentityDecisionEngine → auto-match / review / new individual
```

Architecture: ResNet50 (ImageNet-pretrained) → global pooling → BNNeck → 512-d
L2-normalised embedding, trained with an ArcFace head plus optional batch-hard
triplet loss. The ArcFace logits are training-only; the persisted vector is
always the 512-d normalised embedding the application already stores.

```bash
# 1. Train (see docs/reid_training.md for data requirements)
python -m ml.reid.train --data data/reid --output ml/weights/tiger_reid \
    --backbone resnet50 --embedding-dim 512 --epochs 50 --batch-size 32 --device cuda

# 2. Evaluate on a held-out split — Rank-1/5/10, mAP, similarity separation
python -m ml.reid.evaluate --checkpoint ml/weights/tiger_reid/best.pt \
    --data data/reid --split test --roc

# 3. Derive thresholds from measured distributions (the shipped values are placeholders)
python -m ml.reid.calibrate_thresholds --checkpoint ml/weights/tiger_reid/best.pt \
    --data data/reid --split val --output ml/weights/tiger_reid/thresholds.json

# 4. Activate
#    .env: ML_MODE=production
#          REID_CHECKPOINT_PATH=ml/weights/tiger_reid/best.pt
#          AUTO_MATCH_THRESHOLD / REVIEW_THRESHOLD / NEW_INDIVIDUAL_THRESHOLD from step 3
```

Data splits are made at the **capture-sequence** level, not per image: burst
frames are near duplicates, and letting them straddle train and validation
inflates Rank-1 without improving real recognition. Augmentation is deliberately
conservative because stripe geometry is the identity signal — horizontal flip is
off by default, since mirroring a left flank fabricates a right flank the animal
does not have.

Full guide, data requirements and failure modes: [docs/reid_training.md](docs/reid_training.md).
Dataset layout example: [data/reid_example/](data/reid_example/README.md).

**This project does not contain a validated tiger identification model.** Demo
mode produces deterministic placeholder embeddings; production mode requires a
checkpoint you train and evaluate yourself, and any claim of accuracy must come
from your own `ml.reid.evaluate` output.

## Map and GIS

`backend/app/core/geo.py` holds the reference geography: a reserve boundary, core
zone, two buffer zones, an eastern corridor, a village-interface belt, five gates and
sixteen camera positions. Zone polygons are stored as GeoJSON geometries in the
`zones` table, so the same rows work on PostGIS and SQLite.

**These polygons are simulated.** They are inspired by the general location of Pench
Tiger Reserve but are not surveyed boundaries. To install real data, replace
`PENCH_ZONES` with authoritative GeoJSON, re-seed, and set `GEO_DATA_SOURCE=official`
— the map and the API disclaimer update automatically.

The map is data-driven: `GET /api/v1/map/overview` returns center, bounds, zones,
gates, cameras, sightings and movement tracks in one round trip.

## API overview

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Service, ML mode, model version, demo flag |
| `GET /api/v1/dashboard/stats` | Command-center KPIs |
| `GET /api/v1/map/overview` | Everything the map needs |
| `GET /api/v1/map/zones` · `/zones/geojson` · `/gates` · `/cameras` · `/sightings` · `/movement` | Individual map layers |
| `GET /api/v1/cameras` · `/cameras/{id}` · `/cameras/{id}/observations` | Camera list, detail, detections |
| `POST` / `PATCH /api/v1/cameras` | Register or update a station |
| `GET /api/v1/tigers` · `/tigers/{code}` · `/{code}/observations` · `/{code}/gallery` | Catalogue, profile, sightings, images |
| `GET /api/v1/observations` | Filterable detections (camera, tiger, zone, species, confidence, dates) |
| `POST /api/v1/images/upload` · `/images/batch` | Run the pipeline on uploads |
| `GET /api/v1/images` · `/images/{id}` · `/images/{id}/file` | Gallery, metadata, bytes |
| `POST /api/v1/images/{id}/restore` · `/delete` | Quarantine actions |
| `GET /api/v1/triage/report` · `/triage/runs` · `/triage/quarantine` | Triage state |
| `GET /api/v1/reviews` · `POST /{id}/approve` · `/reject` · `/new-tiger` | Human review |
| `GET /api/v1/alerts` · `/alerts/summary` · `POST /alerts/evaluate` · `PATCH /alerts/{id}` | Alerting |
| `GET /api/v1/analytics/overview` | Analytics aggregates |
| `GET /api/v1/demo/status` · `POST /api/v1/demo/simulate` | Demo mode |

Full interactive documentation: `http://localhost:8000/docs`.

## Suggested demo flow

Command Center → Reserve Map (click a camera, then a sighting) → Tiger Profile
(gallery, timeline, movement) → Gallery (open a capture, read its metadata) → Alerts →
Analytics → Demo Mode (simulate a capture and follow it back to the map).

## Testing

```bash
cd backend && python -m pytest tests/ -v      # API, pipeline, triage, re-id
cd frontend && npx tsc --noEmit               # type check
cd frontend && npm run build                  # production build
```

The API tests exercise the app end-to-end through Starlette's test client: health,
dashboard, map layers, camera list and detail, tiger profile and gallery, detection
filters, alert evaluation, analytics, the review queue, upload validation and a full
demo simulation. The Re-ID tests cover dataset discovery, split reproducibility,
capture-sequence leakage prevention, `[B, 512]` output shape, L2 normalisation,
checkpoint round-trip, a real one-batch loss decrease, resume, Rank-k/mAP
correctness, threshold calibration, embedding extraction and the
encoder ↔ checkpoint integration. They use a `tiny` backbone so no pretrained
weights are downloaded, and tests needing OpenCV skip cleanly when it is absent.

## Security notes

Uploads are validated on extension, size and decodability; filenames are sanitised to
a basename before use, and local storage keys are resolved inside the storage root to
block traversal. Enum-valued query parameters are validated and return 422 rather than
500. There is **no authentication layer** — the API is open by design for a prototype,
and must be placed behind authentication and authorisation before it is exposed to a
network.

## Future improvements

- Authentication, authorisation and per-user audit trails
- Real GIS layers from the Forest Department / WII, replacing the simulated polygons
- Train the Re-ID model on labelled Pench flank crops and publish Rank-1/mAP
- Segmentation-based flank extraction to replace the current crop heuristic
- Background job queue so large batch ingestion does not block requests
- Home-range estimation and occupancy modelling on top of the sighting data
- Push notifications (SMS/WhatsApp) for critical alerts to field teams
- Frontend code splitting to bring the bundle below the 500 kB warning threshold
