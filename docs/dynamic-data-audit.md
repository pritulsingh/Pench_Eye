# Dynamic Data Audit

Audit scope: identify every place where the app still relies on hardcoded or synthetic domain data, then map each item to the database/API/frontend change needed to make PostgreSQL the source of truth.

## Current Architecture

- Backend is already split into models, services, routers, and schemas.
- Frontend already consumes a typed REST client instead of importing local JSON for most screens.
- The main source of hardcoded domain data is the demo/geography/seed layer, not the React component tree.
- The app currently starts in demo mode and auto-seeds simulated reserve geometry, cameras, tigers, observations, images, alerts, and review-queue rows.
- Static UI configuration exists in the frontend and is acceptable where it does not represent actual application data.

## Hardcoded / Synthetic Data Findings

| File | Hardcoded data found | What it represents | Current source | Desired source | API / DB change required | Frontend change required | Priority |
|---|---|---|---|---|---|---|---|
| [backend/app/core/geo.py](../backend/app/core/geo.py) | Reserve center/bounds, polygon geometries, gates, camera coordinates/names/codes | Simulated geography and camera network | Hand-authored constants | Database-backed geometry and reference tables | Replace simulated geometry with authoritative zone rows or GIS import; keep gates only if persisted elsewhere | Map should render from API payloads only; neutral center is fine for empty DB | High |
| [backend/app/core/seed.py](../backend/app/core/seed.py) | `DEMO_TIGERS`, seeded cameras, synthetic observations, demo images, demo alerts, review queue entries | Full fake production dataset | Startup/demo seed logic | Real rows inserted from ML/backend ingestion and admin setup | Stop auto-seeding domain data in production path; keep only optional dev/demo seed tooling | Frontend must handle empty state instead of relying on seeded records | High |
| [backend/app/main.py](../backend/app/main.py) | Auto-seeds demo dataset when `ML_MODE=demo` | Boot-time fake data population | Lifespan startup hook | Empty database until real data arrives | Gate demo seeding behind an explicit local/demo-only path; production should not fabricate data | Dashboard/map/profile pages must render zero/empty states | High |
| [backend/app/api/v1/demo.py](../backend/app/api/v1/demo.py) | Demo status/simulation endpoints | Simulated capture flow for hackathon demo | Demo simulation API | Optional dev-only simulation, not source of truth | Keep only as a dev/demo utility; do not use it as a production data source | Demo UI should stay clearly labeled as simulation-only | Medium |
| [backend/app/services/simulation_service.py](../backend/app/services/simulation_service.py) | Synthetic capture generation and demo observation creation | Local demo pipeline runner | Demo-only producer | Real ingestion boundary from ML pipeline | No ML changes; ensure real observation persistence path is distinct from demo simulation | Demo page must not be treated as production data flow | Medium |
| [backend/app/services/inference_service.py](../backend/app/services/inference_service.py) | Deterministic demo inference behavior and demo tiger identity selection | Simulated ML outputs | Demo inference implementation | External ML pipeline output | Leave ML alone per requirements; only the integration boundary matters | Frontend should consume backend outputs, not demo assumptions | Medium |
| [backend/app/services/analytics_service.py](../backend/app/services/analytics_service.py) | Demo-data ratio flag and analytics derived from seeded rows | Demo-vs-real dataset labeling | Computed from rows | Database-derived analytics | No new model required; aggregation endpoints already exist | Dashboard/analytics should show zeroes and empty charts when DB is empty | Medium |
| [backend/app/api/v1/map_data.py](../backend/app/api/v1/map_data.py) | Static gates/GeoJSON endpoints backed by `geo.py` constants | Reserve map layers and gates | Simulated geo fixtures | Persisted zones/gates or imported GIS data | Keep endpoint shape, but back it with real zone records where possible | Map must tolerate no zones/gates without breaking | High |
| [frontend/src/pages/TigerProfile.tsx](../frontend/src/pages/TigerProfile.tsx) | Hardcoded fallback map center `[21.7, 79.26]` | Neutral Pench map center | UI fallback constant | Neutral map center only | No DB change needed; this is UI-only | Keep as a safe empty-state center, not tiger location data | Low |
| [frontend/src/components/ui/DemoWarning.tsx](../frontend/src/components/ui/DemoWarning.tsx) | Demo mode banner text | Simulation disclaimer | Frontend UI config | Static UI warning | No data change required | Keep; it helps prevent confusion | Low |
| [frontend/src/features/map/mapConfig.ts](../frontend/src/features/map/mapConfig.ts) | Tile URLs, marker colors/labels, icon rendering | Visual configuration | Static UI config | Static UI config | No change required | Keep; this is not domain data | Low |
| [frontend/src/pages/Dashboard.tsx](../frontend/src/pages/Dashboard.tsx) | Default copy for empty states and demo prompting | Empty-state text | Frontend UI strings | API-returned counts + empty state | No DB change required | Adjust only if copy needs to mention empty DB explicitly | Medium |
| [scripts/seed_demo_data.py](../scripts/seed_demo_data.py) | Demo seeding entry point | Dev/demo dataset setup | Synthetic seed tool | Optional local demo convenience | Keep as an explicit seed tool only | No frontend change | Medium |
| [ml/demo/demo_mode.py](../ml/demo/demo_mode.py) | Hardcoded demo tiger identities and decisions | Simulated ML demo catalog | Demo ML fixture | External ML output | Do not modify for this task | None | Out of scope |

## Existing Database Entities

The schema already contains the core tables needed for the target architecture:

- `tigers`
- `camera_stations`
- `observations`
- `alerts`
- `images`
- `zones`
- `review_queue`
- `embeddings`
- `triage_runs`

Observations already carry the right bridge fields for the ML boundary:

- `tiger_id`
- `camera_id`
- `timestamp`
- `latitude`
- `longitude`
- `identity_confidence`
- `image_id`
- `match_type`
- `review_status`
- `is_demo`

## Existing APIs

Backend routes already exist for:

- `/api/v1/tigers`
- `/api/v1/cameras`
- `/api/v1/observations`
- `/api/v1/alerts`
- `/api/v1/dashboard/stats`
- `/api/v1/map/overview`
- `/api/v1/map/cameras`
- `/api/v1/map/sightings`
- `/api/v1/map/movement`
- `/api/v1/analytics/overview`
- `/api/v1/images`
- `/api/v1/reviews`
- `/api/v1/demo`

This means the missing work is mostly contract cleanup, empty-state handling, and replacing demo-seeded data with real persisted records, not inventing a new API surface.

## Missing / Incomplete API Contract Items

These are the main gaps relative to the requested production flow:

1. A clear production ingestion contract for the ML output payload to create an observation row.
2. Optional explicit endpoints for tiger movement history derived from observations, if you want to expose it separately from the map overview payload.
3. A clearer division between demo-only endpoints and production API behavior when the database is empty.
4. A zone/gate persistence strategy so the map does not depend on simulated GIS constants.

## Frontend Surfaces That Must Become Fully Dynamic

- Dashboard summary cards and charts.
- Tiger catalog and tiger profile pages.
- Camera list and camera detail pages.
- Observation list and filters.
- Alerts list and summary.
- Map markers, sightings, and movement paths.
- Gallery, review queue, and triage panels where they currently assume demo content exists.

## Minimum Change Set

1. Stop auto-populating the production database with simulated reserve/tiger/camera/observation data.
2. Keep the existing API routes, but ensure they return real database rows or empty collections/zero counts.
3. Add or refine a persistence boundary for ML outputs so the backend stores one observation per detection result.
4. Make the frontend treat empty collections as normal, with explicit empty states for tigers, cameras, observations, alerts, and movement.
5. Replace hardcoded map data sources with DB-backed zones/cameras/observations.

## Migration Plan

### Phase 1: Data-source cleanup

- Remove demo seeding from the production startup path.
- Keep demo mode only as an explicit local/demo workflow.
- Confirm that every list/detail API returns zero rows cleanly when the database is empty.

### Phase 2: Map and reference data

- Move reserve zones and camera reference data to persisted rows or an import step.
- Keep the map centered on Pench with no fabricated tiger positions.

### Phase 3: Core dashboard surfaces

- Validate tiger, camera, observation, and alert endpoints against an empty database.
- Update empty states so the UI remains useful with zero data.

### Phase 4: Movement and profiles

- Continue deriving movement from ordered observations grouped by tiger.
- Ensure tiger profile pages and map movement layers use the same observation-backed source.

### Phase 5: ML integration contract

- Document the exact observation payload the backend persists after ML inference.
- Keep the ML pipeline unchanged; only the backend ingestion boundary should consume it.

## Empty-State Targets

- Tigers: "No tiger profiles available yet."
- Cameras: "No camera data available yet."
- Observations: "No tiger observations available yet."
- Alerts: "No alerts available."
- Movement: "No movement history available yet."
- Images: "No image available."
- Dashboard metrics: zero values.

## Audit Conclusion

The application is already more modular than a typical hackathon prototype. The main issue is not missing backend structure; it is that the demo seed and simulated geography layers currently make the app look populated before real data exists. The safest migration is to preserve the current API shape, remove demo seeding from the production path, and make the frontend treat empty database responses as the normal case.