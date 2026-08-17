# MegaDescriptor Integration Technical Guide

## Overview

This document explains the technical implementation of MegaDescriptor integration into Pench Eye for the MVP version of Amur tiger re-identification.

---

## Architecture

### Component Hierarchy

```
Frontend (Upload.tsx)
    ↓
API (POST /api/v1/images/upload)
    ↓
PipelineService (pipeline_service.py)
    ├─ Validate upload
    ├─ Decode image
    ├─ Triage (YOLO)
    ├─ Detect (YOLO)
    └─ _identify() → ProductionInference.identify_frame()
           ↓
    InferencePipeline (inference_service.py)
    ├─ DemoInference (simulated)
    └─ ProductionInference (real models)
           ├─ detect_frame() [YOLO]
           ├─ triage_frame() [YOLO]
           └─ identify_frame() → MegaDescriptor
                  ↓
    MegaDescriptorEmbeddingService (megadescriptor_service.py)
           ↓
    MegaDescriptor (ml/megadescriptor/model.py)
           ├─ Download from Hugging Face / torch.hub
           ├─ Preprocess image (PIL → tensor)
           ├─ Forward pass (generate embedding)
           └─ L2 normalize (result)
           
Similarity Search & Matching
    ↓
Database Operations
    ├─ Create Tiger (if new)
    ├─ Store Observation
    └─ Store Embedding
    
Response → Frontend → Map Display
```

---

## Module Details

### 1. MegaDescriptor Model (`ml/megadescriptor/model.py`)

#### Class: `MegaDescriptor`

**Initialization**:
```python
model = MegaDescriptor(
    model_name="facebookresearch/omnivore:megadescriptor_768",
    cache_dir=".cache/megadescriptor",
    device="cuda" or "cpu"  # Auto-detect if None
)
```

**Model Loading Strategy**:
1. First attempt: `torch.hub.load()` (simplest, no dependencies)
2. Fallback: `transformers.AutoModel.from_pretrained()` (more features)
3. Error: Raises `RuntimeError` with guidance

**Key Methods**:

1. **`preprocess_image(image)`**
   - Input: Path, PIL Image, or numpy array
   - Process:
     - Load as PIL RGB
     - Resize to 224×224 (standard vision model size)
     - Convert to tensor (C, H, W)
     - Apply ImageNet normalization
     - Move to device
   - Output: torch.Tensor shape (1, 3, 224, 224)

2. **`get_embedding(image)`**
   ```python
   # Single image embedding
   embedding = model.get_embedding("tiger.jpg")
   # Returns: np.array shape (768,), L2-normalized
   ```
   
   - Calls `preprocess_image()`
   - Forward pass through model
   - Extract embedding (handle various output formats)
   - L2 normalize: `emb / ||emb||_2`
   - Convert numpy, return

3. **`get_embeddings_batch(images)`**
   ```python
   # Multiple images
   embeddings = model.get_embeddings_batch([img1, img2, ...])
   # Returns: np.array shape (n_images, 768)
   ```

4. **`cosine_similarity(emb1, emb2)`** (static)
   ```python
   # Dot product of L2-normalized vectors = cosine similarity
   sim = MegaDescriptor.cosine_similarity(emb1, emb2)
   # Returns: float in [-1, 1]
   ```

5. **`cosine_similarities_batch(query, references)`** (static)
   ```python
   # Query vs many references
   sims = MegaDescriptor.cosine_similarities_batch(query_emb, ref_embeddings)
   # query_emb: (768,)
   # ref_embeddings: (n, 768)
   # Returns: (n,) array of similarities
   ```

**Device Handling**:
- Auto-detects CUDA: `torch.cuda.is_available()`
- Falls back to CPU if GPU unavailable
- All operations `.to(self.device)`

**Normalization**:
- L2 normalization: `emb = emb / norm(emb)`
- Ensures all embeddings lie on unit hypersphere
- Cosine similarity = dot product for normalized vectors

---

### 2. Embedding Service (`backend/app/services/megadescriptor_service.py`)

#### Class: `MegaDescriptorEmbeddingService`

**Purpose**: Wrapper for backend use, handles:
- Global model caching
- Multiple input formats (bytes, path, PIL, numpy)
- Batch operations
- Error handling

**Key Methods**:

1. **`get_embedding(image_data)`**
   ```python
   # Accepts multiple formats
   embedding = service.get_embedding(image_bytes)  # bytes
   embedding = service.get_embedding(path)         # Path
   embedding = service.get_embedding(pil_image)    # PIL Image
   embedding = service.get_embedding(np_array)     # numpy array
   ```

2. **`find_most_similar(query_embedding, reference_embeddings)`**
   ```python
   # Find best match in list of references
   result = service.find_most_similar(
       query_embedding,  # (768,)
       [
           {"embedding": emb1, "tiger_id": "TIGER_001"},
           {"embedding": emb2, "tiger_id": "TIGER_002"},
       ]
   )
   # Returns: {tiger_id, similarity, index}
   ```

**Global Caching**:
```python
_model_instance = None  # Loaded once, reused

def get_megadescriptor_model(...):
    global _model_instance
    if _model_instance is None:
        _model_instance = MegaDescriptor(...)
    return _model_instance
```

**Error Handling**:
- Wraps model loading exceptions
- Provides user-friendly error messages
- Raises `RuntimeError` if model unavailable

---

### 3. Inference Service (`backend/app/services/inference_service.py`)

#### Class: `ProductionInference`

**Replaces** the old Re-ID complexity with MegaDescriptor.

**Key Method: `identify_frame()`**
```python
identity_output = pipeline.identify_frame(
    pixels=np.array(image),           # numpy RGB array
    image_hash="abc123...",           # sha256 for logging
    known_tiger_codes=[],             # unused for MVP
    flank_side="unknown"              # unused for MVP
)
```

**Returns** `IdentityOutput`:
```python
@dataclass
class IdentityOutput:
    embedding: List[float]                # 768-d vector as list
    flank_side: str = "unknown"           # MVP: always "unknown"
    model_version: str                    # "production-megadescriptor-v1"
    similarity: float = 0.0               # MVP: unused (set by caller)
    suggested_tiger_code: Optional[str]   # MVP: None
    candidates: List[Dict[str, Any]]      # MVP: empty list
    is_demo: bool = False
    preprocessing_version: Optional[str]  # "megadescriptor-pil"
    quality_score: Optional[float] = None # MVP: None
    quality_warnings: List[str]           # MVP: empty list
```

**Process**:
1. Call `_get_megadescriptor()` (lazy load)
2. Generate embedding: `md.get_embedding(pixels)`
3. Convert to list: `embedding.tolist()`
4. Return `IdentityOutput` with embedding

**Error Handling**:
- If model fails to load: raise `MegaDescriptorUnavailable`
- Embedding generation fails: raise `MegaDescriptorUnavailable`
- Pipeline catches and queues for human review

---

### 4. Pipeline Service (`backend/app/services/pipeline_service.py`)

#### Method: `_identify()`

**New MVP Logic**:

```python
async def _identify(db, pixels, image_hash):
    """
    Generate embedding and find similarity with existing tigers.
    """
    # Step 1: Generate embedding
    identity = pipeline.identify_frame(pixels, image_hash, [])
    
    # Step 2: Query existing embeddings
    existing = await db.execute(
        select(Embedding.embedding, Embedding.tiger_id)
        .where(Embedding.tiger_id.isnot(None))
    )
    
    # Step 3: Find best match
    best_similarity = -1.0
    best_tiger_id = None
    
    for emb_vector, tiger_id in existing:
        similarity = cosine_similarity(identity.embedding, emb_vector)
        if similarity > best_similarity:
            best_similarity = similarity
            best_tiger_id = tiger_id
    
    # Step 4: Decision
    if best_similarity >= MEGADESCRIPTOR_SIMILARITY_THRESHOLD:
        return identity, {
            "decision": "auto_match",
            "tiger_id": best_tiger_id,
            "similarity": best_similarity
        }, None
    else:
        return identity, {
            "decision": "new_individual",
            "similarity": best_similarity
        }, None
```

**Key Points**:
- No quality gating
- No review queue
- Simple threshold check
- Direct tiger ID lookup

---

### 5. Data Models

#### `Embedding` Model (unchanged from existing)
```python
class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(GUID(), primary_key=True)
    embedding_id = Column(String(100), unique=True)
    observation_id = Column(GUID, ForeignKey("observations.id"))
    tiger_id = Column(GUID, ForeignKey("tigers.id"), nullable=True)
    
    # KEY: Uses dynamic dimension from config
    embedding = Column(VectorType(settings.EMBEDDING_DIM), nullable=False)
    
    model_version = Column(String(100))      # "megadescriptor-v1"
    flank_side = Column(String(20))          # Always "unknown" for MVP
```

#### Database Storage
- **PostgreSQL**: Native `vector(768)` type via pgvector
- **SQLite**: JSON-encoded float array (fallback)

#### Dimension Change
```python
# In config.py:
EMBEDDING_DIM: int = 768  # Changed from 512

# In Embedding model:
embedding = Column(VectorType(settings.EMBEDDING_DIM), ...)  # Dynamic!
```

---

## Configuration Flow

### Environment → Settings → Pipeline

```python
# 1. Environment (.env file)
MEGADESCRIPTOR_MODEL_NAME=facebookresearch/omnivore:megadescriptor_768
MEGADESCRIPTOR_CACHE_DIR=.cache/megadescriptor
MEGADESCRIPTOR_SIMILARITY_THRESHOLD=0.75
EMBEDDING_DIM=768
ML_MODE=production

# 2. Settings (config.py)
settings = Settings()
assert settings.EMBEDDING_DIM == 768
assert settings.MEGADESCRIPTOR_SIMILARITY_THRESHOLD == 0.75

# 3. Pipeline initialization (main.py)
inference_pipeline = build_pipeline()  # Uses ML_MODE
assert inference_pipeline.is_demo == False  # Production mode

# 4. First request
ProductionInference._get_megadescriptor()
  → MegaDescriptorEmbeddingService()
    → MegaDescriptor(model_name, cache_dir)
```

---

## Data Flow: Image Upload

### Step-by-step tracking of a single tiger image:

```
1. POST /api/v1/images/upload
   Content: image bytes + camera_id="CAM_01"

2. PipelineService.process_image()
   a. validate_upload(filename, content)  → safe_name
   b. decode_image(content)               → pixels, W, H
   c. sha256(content)                     → image_hash
   d. Create Image record (status=TRIAGED)
   e. Store in storage system

3. Triage
   ProductionInference.triage_frame(pixels)
   → BlankImageClassifier.classify()
   → TriageOutput(is_blank=False, ...)

4. Detection
   ProductionInference.detect_frame(pixels)
   → TigerDetector.detect()
   → DetectionOutput(present=True, species="tiger", ...)

5. Identification (MegaDescriptor MVP path)
   a. ProductionInference.identify_frame(pixels)
      → MegaDescriptorEmbeddingService.get_embedding(pixels)
      → MegaDescriptor model inference
      → embedding (768-d L2-norm) ✓
   
   b. PipelineService._identify()
      → Query existing embeddings from database
      → Calculate cosine similarities
      → Find argmax(similarities)
      
      If max_similarity >= 0.75:
         decision = "auto_match"
         tiger_id = best_match.tiger_id ✓
      Else:
         decision = "new_individual"
         tiger_id = TigerService.create_tiger() ✓
   
   c. similarity_result = {
         "decision": "auto_match" | "new_individual",
         "tiger_id": uuid,
         "similarity": 0.91
      }

6. Observation Creation
   observation = Observation(
       tiger_id=tiger_id,
       camera_id="CAM_01",
       image_id=image.id,
       latitude=camera.latitude,
       longitude=camera.longitude,
       identity_confidence=similarity,
       match_type=MatchType.AUTO_MATCH,
       review_status=ReviewStatus.APPROVED,
   )
   db.add(observation)
   await db.commit()

7. Embedding Storage
   embedding = Embedding(
       tiger_id=tiger_id,
       observation_id=observation.id,
       embedding=identity.embedding,  # List of 768 floats
       model_version="megadescriptor-v1",
   )
   db.add(embedding)
   await db.commit()

8. Update tiger statistics
   TigerService.update_tiger_stats(tiger_id)
   → Total observations, first/last seen

9. Alert evaluation (unchanged)
   AlertService.evaluate_detection(observation, camera, tiger)

10. Return response to frontend
    {
        "tiger_code": "TIGER_001",
        "is_new": false,
        "similarity": 0.91,
        "confidence": "high",
        "observation_id": "OBS-123",
        "camera_id": "CAM_01"
    }

11. Frontend display
    Upload.tsx shows:
      Tiger: TIGER_001 ✓
      Similarity: 91% ✓
      Match: Existing Tiger ✓
      Links to map ✓
```

---

## Model Details

### MegaDescriptor Architecture (Conceptual)

MegaDescriptor is a vision transformer trained on diverse visual recognition tasks:

```
Input Image (224×224)
    ↓
Vision Transformer (ViT) backbone
    ├─ Multi-head self-attention layers
    ├─ Token embeddings
    └─ Class token [CLS]
    ↓
Global pooling ([CLS] token)
    ↓
Output: 768-dimensional feature vector
    ↓
L2 normalization (MVP requirement)
    ↓
Normalized embedding (lie on unit hypersphere)
```

**Why 768 dimensions?**
- Standard ViT output size
- Good balance between expressiveness and efficiency
- Sufficient for visual similarity matching

**Why L2 normalization?**
- Makes cosine similarity = dot product
- Faster similarity computation
- Invariant to scale
- Required for pgvector's cosine operator

---

## Similarity Matching Algorithm

### Cosine Similarity for L2-Normalized Embeddings

```python
# Given normalized embeddings (norm = 1)
query_emb = [x1, x2, ..., x768]  # ||query_emb|| = 1
ref_emb = [y1, y2, ..., y768]    # ||ref_emb|| = 1

# Cosine similarity = dot product
similarity = sum(x_i * y_i for i in 1..768)

# Range: [-1, 1]
# Interpretation:
#   1.0   = identical
#   0.5   = moderate similarity
#   0.0   = orthogonal (uncorrelated)
#  -0.5   = moderately dissimilar
#  -1.0   = opposite

# MVP threshold: 0.75
# Rationale:
#   - High confidence match
#   - Low false positive rate
#   - Calibration needed on real data
```

### Batch Similarity Search
```python
# Query: (768,)
# Gallery: (1000, 768)
# Result: (1000,) similarities

similarities = np.dot(gallery, query)  # Fast vectorized operation

best_idx = np.argmax(similarities)
best_similarity = similarities[best_idx]

if best_similarity >= 0.75:
    # Match found
    tiger_id = gallery_tiger_ids[best_idx]
```

---

## Error Handling

### Exception Hierarchy

```
Exception
├─ MegaDescriptorUnavailable (inherits RuntimeError)
│  ├─ Model download failed
│  ├─ Model loading failed
│  └─ Embedding generation failed
│
└─ ImageValidationError (inherits ValueError)
   ├─ File too large
   ├─ Invalid format
   └─ Unreadable image
```

### Error Flow

```
MegaDescriptor model not available
    ↓
identify_frame() raises MegaDescriptorUnavailable
    ↓
PipelineService._identify() catches exception
    ↓
Returns (None, None, "error message")
    ↓
_create_observation_from_detection() checks identity_error
    ↓
Sets review_status=PENDING_REVIEW
    ↓
Observation stored without tiger_id
    ↓
Response indicates "identity_unavailable"
    ↓
Frontend shows warning
    ↓
Human can manually assign tiger later
```

---

## Testing Strategy

### Unit Tests
- MegaDescriptor model loading (mocked)
- Embedding generation (mock data)
- Cosine similarity calculations

### Integration Tests
- End-to-end pipeline with mock images
- Database operations
- Observation and embedding storage
- Movement track calculation

### Manual Testing
1. Start backend with `ML_MODE=production`
2. Upload tiger image via `/upload` page
3. Verify embedding generated in logs
4. Check observation in database
5. View movement path on map
6. Upload second image of same tiger
7. Verify path extends

---

## Performance Optimization

### Current Bottlenecks
1. **Model download**: ~500MB, one-time cost
2. **Inference**: ~200-500ms per image (CPU)
3. **Similarity search**: O(n) linear scan

### Optimization Strategies

**For Inference**:
- Batch processing: `model.get_embeddings_batch([img1, img2, ...])`
- GPU usage: ~5-10x speedup
- Model quantization: ~2x speedup, slight accuracy loss

**For Search**:
- PostgreSQL + pgvector: O(log n) with IVFFlat index
- Approximate nearest neighbor (ANN): Trade accuracy for speed
- Batch indexing: Group embeddings by tiger

**Code Example (Batch)**:
```python
# Instead of:
for image in images:
    emb = model.get_embedding(image)  # 200ms × N

# Use:
embeddings = model.get_embeddings_batch(images)  # 200ms × N / batch_size
```

---

## Future Extensions

### 1. Fine-tuning MegaDescriptor
```python
# ml/megadescriptor/finetune.py
class FineTunedMegaDescriptor(MegaDescriptor):
    def __init__(self, checkpoint_path):
        self.model = torch.load(checkpoint_path)  # Pre-trained, then fine-tuned
        # Keep same interface!
    
    def get_embedding(self, image):  # Same signature
        # ...same preprocessing...
        output = self.model(tensor)
        return F.normalize(output, p=2, dim=1)[0].cpu().numpy()

# Usage: No change to rest of pipeline!
```

### 2. Multi-view Embeddings
```python
# Store embeddings from multiple image crops
embeddings = {
    "full": get_embedding(full_image),
    "left_flank": get_embedding(left_crop),
    "right_flank": get_embedding(right_crop),
}

# Aggregate for matching
avg_emb = np.mean([embeddings[k] for k in embeddings], axis=0)
```

### 3. Embedding Reranking
```python
# Instead of single best match
top_k = 5
scores = cosine_similarities_batch(query, gallery)
top_indices = np.argsort(scores)[-top_k:]

# Use all top-k, not just best
# Aggregate by tiger for better signal
```

---

## Debugging Guide

### Check Model Loading
```python
from app.services.megadescriptor_service import get_megadescriptor_model

try:
    model = get_megadescriptor_model()
    print(f"Model loaded: {model}")
except Exception as e:
    print(f"Failed to load: {e}")
```

### Generate Test Embedding
```python
import numpy as np
from PIL import Image

model = get_megadescriptor_model()
test_image = Image.new("RGB", (224, 224), color=(100, 100, 100))
embedding = model.get_embedding(test_image)

print(f"Embedding shape: {embedding.shape}")  # Should be (768,)
print(f"Norm: {np.linalg.norm(embedding)}")   # Should be ~1.0
```

### Check Similarity Calculation
```python
emb1 = np.random.randn(768)
emb1 /= np.linalg.norm(emb1)

emb2 = np.random.randn(768)
emb2 /= np.linalg.norm(emb2)

sim = np.dot(emb1, emb2)
print(f"Similarity: {sim:.4f}")  # Between -1 and 1
```

### Verify Database Storage
```python
from sqlalchemy import select
from app.models.embedding import Embedding

async with db.begin():
    result = await db.execute(select(Embedding).limit(5))
    for emb in result.scalars():
        print(f"Tiger: {emb.tiger_id}")
        print(f"Embedding length: {len(emb.embedding)}")
        print(f"Model version: {emb.model_version}")
```

---

## References

- MegaDescriptor Paper: Meta AI Research
- Hugging Face Model: `facebookresearch/omnivore`
- pgvector Extension: https://github.com/pgvector/pgvector
- L2 Normalization: https://en.wikipedia.org/wiki/Cosine_similarity

---

**Last Updated**: 2026-08-16  
**Status**: MVP Complete ✅
