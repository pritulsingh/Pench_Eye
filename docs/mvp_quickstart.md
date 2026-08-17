# Pench Eye MVP: Amur Tiger Re-ID with MegaDescriptor

## Quick Start Guide

This MVP simplifies the system to focus on the core tiger identification workflow:

```
Upload → Detect → MegaDescriptor Embedding → Match → Store → Path
```

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (CPU or CUDA)
- PostgreSQL 13+ (optional; SQLite works for MVP)

### Installation

1. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   
   cd ../ml
   pip install -r requirements.txt
   
   cd ../frontend
   npm install
   ```

2. **Configure environment** (backend/.env):
   ```env
   ML_MODE=production
   MEGADESCRIPTOR_MODEL_NAME=facebookresearch/omnivore:megadescriptor_768
   MEGADESCRIPTOR_CACHE_DIR=.cache/megadescriptor
   MEGADESCRIPTOR_SIMILARITY_THRESHOLD=0.75
   DATABASE_URL=sqlite+aiosqlite:///storage/pench_eye.db
   ```

3. **Initialize database**:
   ```bash
   cd backend
   python -m app.core.database  # Creates schema
   ```

### Running the MVP

**Terminal 1: Backend API**
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Frontend (dev server)**
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173

### Core Workflow

1. **Upload Image**: Navigate to `/upload`
   - Select camera station from dropdown
   - Upload tiger image (JPEG/PNG)
   - Backend runs: triage → detect → MegaDescriptor embedding

2. **Match Decision**: MVP uses simple threshold-based matching
   - Query: Generate embedding for new image
   - Gallery: Search existing embeddings
   - If `max_similarity >= 0.75` → Known Tiger
   - Else → Create New Tiger

3. **Observation Stored**: System records:
   - Tiger ID (TIGER_001, TIGER_002, etc.)
   - Camera location
   - Timestamp
   - Similarity score (as confidence)
   - Embedding vector (768-d)

4. **View Movement Path**: Navigate to `/map`
   - Filter by tiger → shows all observations
   - Map draws polyline through camera locations in time order
   - Distance and time displayed for each leg

### Key Differences from Full System

**Removed for MVP**:
- Complex Re-ID training pipeline
- Stripe extraction and flank processing
- Quality gating and ambiguity scoring
- Metric learning (ArcFace, triplet loss)
- Human review queue for ambiguous matches

**Simplified to**:
- Pretrained MegaDescriptor (no training)
- Full tiger image → embedding
- Cosine similarity → yes/no match
- Auto-match all detections ≥ threshold

### Database Schema

Key tables:

```sql
Tigers
  id (UUID)
  tiger_id (e.g., "TIGER_001")
  created_at

Embeddings
  id (UUID)
  tiger_id (FK)
  observation_id (FK)
  embedding (vector[768], L2-normalized)
  model_version ("megadescriptor-v1")

Observations
  id (UUID)
  tiger_id (FK)
  camera_id
  image_id (FK)
  timestamp
  latitude, longitude
  similarity (match score)
  identity_confidence (similarity score)
  match_type (AUTO_MATCH, NEW_INDIVIDUAL)

CameraStations
  id (UUID)
  camera_id (e.g., "CAM_01")
  name
  latitude, longitude
```

### API Endpoints (MVP)

**Image Processing**:
- `POST /api/v1/images/upload` → Process image, generate embedding, match/create tiger

**Tiger Management**:
- `GET /api/v1/tigers` → List all tigers
- `GET /api/v1/tigers/{tiger_id}/profile` → Tiger details + observations

**Observations**:
- `GET /api/v1/observations` → List all sightings

**Map Data**:
- `GET /api/v1/map/overview` → Cameras, sightings, movement paths
- `GET /api/v1/map/movement` → Tiger movement tracks (polylines)

### Configuration

**Backend (backend/app/core/config.py)**:

```python
# MegaDescriptor
MEGADESCRIPTOR_MODEL_NAME = "facebookresearch/omnivore:megadescriptor_768"
MEGADESCRIPTOR_CACHE_DIR = ".cache/megadescriptor"
MEGADESCRIPTOR_SIMILARITY_THRESHOLD = 0.75  # Adjust based on validation data

# Embedding dimension (changed from 512 for MegaDescriptor)
EMBEDDING_DIM = 768

# Database
ML_MODE = MLMode.PRODUCTION  # Use MegaDescriptor
DATABASE_URL = "sqlite+aiosqlite:///storage/pench_eye.db"  # or PostgreSQL
```

### Extending the MVP

Once the basic pipeline works, you can extend by:

1. **Calibrating thresholds** on real tiger data
   - Collect matched/non-matched pairs
   - Plot similarity distributions
   - Adjust `MEGADESCRIPTOR_SIMILARITY_THRESHOLD`

2. **Fine-tuning MegaDescriptor**
   - Use existing Re-ID training pipeline with real tiger data
   - Replace MegaDescriptor in `ml/megadescriptor/model.py`
   - Keep same interface: `get_embedding(image) → 768-d vector`

3. **Adding quality checks**
   - Re-enable flank extraction
   - Implement crop quality scoring
   - Flag low-quality matches for review

4. **Tracking improvement**
   - Store multi-view embeddings (multiple flank sides)
   - Weight embeddings by quality
   - Use advanced matching strategies

### Troubleshooting

**MegaDescriptor download fails**:
- Ensure internet connectivity
- Check Hugging Face token if model is private
- Manually download and cache locally

**"Identity unavailable" errors**:
- Indicates MegaDescriptor loading failed
- Check backend logs for TensorFlow/PyTorch version conflicts
- Verify CUDA compatibility if using GPU

**Low matching accuracy**:
- Increase `MEGADESCRIPTOR_SIMILARITY_THRESHOLD` to reduce false matches
- Decrease to catch more variants of same tiger
- Collect ground truth data for calibration

**Slow embedding generation**:
- Use GPU: `CUDA_VISIBLE_DEVICES=0`
- Batch process images when possible
- Profile with: `python -m cProfile -s cumtime ...`

### Demo Mode

For development without internet/GPU:

```env
ML_MODE=demo  # Uses deterministic simulated embeddings
```

This generates reproducible fake embeddings for testing UI without downloading models.

### Performance Notes

- **MegaDescriptor inference**: ~200-500ms per image (CPU), ~50-100ms (GPU)
- **Similarity search**: O(n) for n embeddings; pgvector provides O(log n) with indexing
- **Full pipeline**: ~1-2 seconds per image (includes I/O)

### Next Steps

1. Test with real camera-trap images
2. Collect validation set (known tiger pairs)
3. Calibrate similarity threshold on validation data
4. Deploy with PostgreSQL + pgvector for scale
5. Implement fine-tuning pipeline when data volume grows

---

**For detailed technical documentation, see [docs/megadescriptor_integration.md](../docs/megadescriptor_integration.md)**
