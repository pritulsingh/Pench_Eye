# Training a real tiger Re-ID model

This document covers turning identity-labelled tiger images into a trained
checkpoint that Pench Eye can serve in production mode.

> **Read this first.** Running the training script does not create a working
> tiger identifier. The pipeline is complete and tested, but a Re-ID model is
> only as good as its labelled data. With no real labelled tiger images you get a
> checkpoint that produces 512-d vectors with no meaningful identity structure.
> Nothing in this project is a scientifically validated tiger identification
> model until you have trained on real data *and* published the evaluation
> numbers from `ml/reid/evaluate.py`.

---

## 1. What data is required

Identity-labelled **flank crops** — images cropped to the side of a tiger's body,
each labelled with which individual it shows. Stripe patterns on the flank are
the identity signal; heads, tails and wide landscape shots contribute little.

You need, at minimum:

| Requirement | Minimum | Recommended |
|---|---|---|
| Individuals (identities) | 2 | 20+ |
| Images per individual | 2 | 15–30 |
| Distinct capture sequences per individual | 2 | 5+ |
| Distinct cameras per individual | 1 | 3+ |
| Usable flank crop edge | 96 px | 224 px+ |

Two images of one tiger from a single burst are effectively one observation.
Coverage across **separate encounters, cameras, lighting and seasons** is what
makes a model generalise; more frames of the same moment do not.

## 2. Required identity labels

An identity label is a stable string per individual, e.g. `TIGER_001`,
`PENCH_F12`, `Collarwali`. It must be consistent across every image of that
animal. Labels come from directory names or a CSV column — see below.

## 3. Recommended flank crops

* Crop to the torso; exclude most of the head and tail.
* One flank per crop. Do not stitch both sides into one image.
* Record the flank side (`left` / `right`) when known — put `_left` / `_right` in
  the filename or use the `flank` CSV column. **Left and right flanks are
  different patterns**, so a left-flank query cannot be matched against a
  right-flank gallery entry, and the quality layer flags that mismatch.
* Keep the original aspect ratio; do not stretch.

## 4. Dataset layout

### Directory layout (preferred)

```
data/reid/
├── train/
│   ├── TIGER_001/
│   │   ├── CAM03_20240117_0001.jpg
│   │   └── CAM03_20240117_0002.jpg
│   └── TIGER_002/
├── val/
│   └── TIGER_001/
└── test/
    └── TIGER_001/
```

The `train/val/test` level is optional. Without it, put identity folders directly
under the root and let the splitter divide them.

### Flat CSV annotations

```csv
image_path,identity_id,split,sequence_id,flank
crops/img001.jpg,TIGER_001,train,CAM03_20240117,left
crops/img002.jpg,TIGER_001,train,CAM03_20240117,left
crops/img003.jpg,TIGER_002,val,CAM07_20240118,right
```

Only `image_path` and `identity_id` are required. Relative paths resolve against
the CSV's directory.

An example skeleton with dummy metadata lives in
[`data/reid_example/`](../data/reid_example/README.md). It documents the format;
it contains no tiger images and cannot train anything.

## 5. Split strategy — sequence-level, not image-level

The splitter (`ml/reid/dataset/splitting.py`) divides data at the **capture
sequence** level, never at the image level.

Why this matters: consecutive frames from one camera burst are near duplicates.
Put one in train and its neighbour in validation, and validation Rank-1 measures
memorisation, not recognition — an inflated number that collapses in the field.

Sequence IDs come from the `sequence_id` CSV column, or are inferred by stripping
a trailing frame counter (`CAM003_20240117_0002.jpg` → `CAM003_20240117`). Name
your files so bursts share a prefix, or supply the column explicitly.

The splitter also aims to place every identity in train *and* evaluation, since a
query with no gallery mate is unanswerable and gets excluded from metrics.
Identities that cannot support this stay train-only and are reported.

Corrupt files are detected at discovery and excluded, and identities below
`--min-images-per-identity` are dropped with a warning.

## 6. How to train

```bash
python -m ml.reid.train \
    --data data/reid \
    --output ml/weights/tiger_reid \
    --backbone resnet50 \
    --embedding-dim 512 \
    --epochs 50 \
    --batch-size 32 \
    --num-instances 4 \
    --lr 3e-4 \
    --triplet-weight 1.0 \
    --device cuda \
    --amp
```

What it does:

* ResNet50 (ImageNet-pretrained) → global pooling → 512-d embedding → L2 norm.
* ArcFace head with cross-entropy, plus batch-hard triplet loss weighted by
  `--triplet-weight` (`0` disables it).
* P×K batches (`--num-instances` images per identity) so triplet mining has
  positive pairs to work with.
* Cosine LR schedule with warmup; AMP on CUDA when `--amp` is passed.
* Best checkpoint selected on **validation Rank-1**, not validation loss.

Key options:

| Flag | Purpose |
|---|---|
| `--backbone` | `resnet18/34/50`, `osnet_x1_0` (needs `torchreid`), `tiny` (tests) |
| `--triplet-weight` | Weight λ on the triplet term |
| `--arcface-scale` / `--arcface-margin` | ArcFace `s` and `m` |
| `--no-augmentation` | Disable all augmentation |
| `--horizontal-flip` | Enable mirroring — **off by default**, see below |
| `--early-stopping-patience` | Stop after N epochs without Rank-1 improvement |
| `--max-steps-per-epoch` | Truncate epochs for smoke tests |

### A note on augmentation

Stripe geometry *is* the identity. `ml/reid/augmentation.py` therefore applies
photometric and sensor-level noise (brightness, contrast, mild blur, JPEG
artefacts, small rotation, cutout, mild random-resized crop) and deliberately
excludes vertical flips, large rotations, shear, perspective warp and
grayscaling.

Horizontal flip is **disabled by default**: mirroring a left flank fabricates a
right flank the animal does not have. Enable it only if you train flank-agnostic
and accept the label noise.

## 7. How to resume

```bash
python -m ml.reid.train \
    --data data/reid \
    --output ml/weights/tiger_reid \
    --epochs 80 \
    --resume ml/weights/tiger_reid/latest.pt
```

Model, ArcFace head, optimizer, scheduler and AMP scaler state are all restored,
and training continues from the recorded epoch.

## 8. How to evaluate

```bash
python -m ml.reid.evaluate \
    --checkpoint ml/weights/tiger_reid/best.pt \
    --data data/reid \
    --split test \
    --output ml/weights/tiger_reid/eval_test.json \
    --roc
```

Reports Rank-1 / Rank-5 / Rank-10, mAP, same- vs different-identity cosine
statistics and separation, and optionally a TAR/FAR verification curve. Two
protocols are computed:

* **Leave-one-out** — every image queries all others; self and same-sequence
  images excluded.
* **Query/gallery** — one held-out image per identity queries a disjoint gallery.

If a split cannot support evaluation (identities with a single image), the tool
reports `evaluable: false` and `num_queries: 0` rather than inventing a score.

**Interpreting the numbers.** Rank-1 near chance (≈ 1/num_identities) means the
model learned nothing useful. A same/different separation below ~0.05 means no
threshold can make it reliable — collect more and better data rather than tuning.

## 9. How to calibrate thresholds

The shipped defaults (`AUTO_MATCH_THRESHOLD=0.90`, `REVIEW_THRESHOLD=0.75`,
`NEW_INDIVIDUAL_THRESHOLD=0.60`) are **placeholders, not evidence**. Derive real
values from your model:

```bash
python -m ml.reid.calibrate_thresholds \
    --checkpoint ml/weights/tiger_reid/best.pt \
    --data data/reid \
    --split val \
    --target-far 0.01 \
    --output ml/weights/tiger_reid/thresholds.json
```

How each is chosen:

* **auto_match** — lowest threshold meeting the false-accept budget
  (`--target-far`, default 1%). A false accept merges two different tigers into
  one identity, the most damaging error this system can make.
* **review** — captures `--review-recall` (default 95%) of true same-identity
  pairs, so genuine matches reach a human instead of being discarded.
* **new_individual** — a low percentile of the same-identity distribution, below
  which a genuine match is very unlikely.

Copy the output into `.env`:

```bash
AUTO_MATCH_THRESHOLD=<auto_match_threshold>
REVIEW_THRESHOLD=<review_threshold>
NEW_INDIVIDUAL_THRESHOLD=<new_individual_threshold>
```

Recalibrate after every retrain — thresholds are properties of a specific model.

## 10. How to export the production checkpoint

`best.pt` is already production-shaped. Either point the app at it:

```bash
REID_CHECKPOINT_PATH=ml/weights/tiger_reid/best.pt
```

or leave `REID_CHECKPOINT_PATH` empty and rely on the default search order:

```
ml/weights/tiger_reid/best.pt
ml/weights/tiger_reid/latest.pt
ml/weights/tiger_reid.pt
```

Each checkpoint carries architecture, backbone, embedding dimension,
preprocessing config, identity mapping, epoch and metrics — enough to rebuild the
exact model that produced a stored embedding.

To batch-export embeddings (e.g. to seed a gallery or analyse offline):

```bash
python -m ml.reid.extract_embeddings \
    --checkpoint ml/weights/tiger_reid/best.pt \
    --input data/reid \
    --output embeddings.parquet \
    --include-quality
```

Each row carries image path, identity, sequence, flank, the 512-d embedding,
`model_version` and `preprocessing_version`. Those version fields are not
decoration: embeddings from different models or preprocessing occupy different
spaces and must never be compared.

## 11. How to activate production inference

```bash
# .env
ML_MODE=production
REID_CHECKPOINT_PATH=ml/weights/tiger_reid/best.pt
AUTO_MATCH_THRESHOLD=<from calibration>
REVIEW_THRESHOLD=<from calibration>
NEW_INDIVIDUAL_THRESHOLD=<from calibration>
```

Restart the backend and confirm:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/demo/status
```

Both report `ml_mode`, the active Re-ID `model_version`, and whether the
checkpoint is `available` and `validated`.

**If the checkpoint is missing, production mode does not fall back to demo
embeddings.** `TigerReIDEncoder.encode()` raises `ReIDModelUnavailable`, the
pipeline records the detection with `decision="identity_unavailable"` and queues
it for human review, and `/health` reports identification as unavailable. The API
never presents a simulated embedding as a real identification.

Flow end to end:

```
labelled data → dataset preparation → training → evaluation
    → threshold calibration → best.pt → ML_MODE=production
    → real 512-d embeddings → pgvector cosine search
    → IdentityDecisionEngine → auto-match / review / new individual
```

## 12. Limitations and failure modes

**The model cannot become reliable by running the training script.** It needs
real labelled tiger images with genuine variation. Synthetic or dummy images
exercise the code path and nothing more.

Known failure modes, and what the system does about them:

| Failure mode | Handling |
|---|---|
| **Unknown individual** — a tiger absent from the gallery | Low top-1 similarity → `new_individual`; flagged `possible_domain_shift_or_unknown_individual` |
| **Poor-quality crop** — blur, low contrast, over/underexposure | `ml/reid/quality.assess_crop` scores sharpness and contrast; warnings attached to the embedding |
| **Crop too small** — below 96 px | Blocking reason; match downgraded to human review |
| **Partial visibility / occlusion** | Reduced quality score → review recommended |
| **Left/right flank mismatch** | `assess_match` marks the match unreliable — opposite flanks are different patterns |
| **Sparse gallery** — fewer than 3 enrolled images | `sparse_gallery` warning; review recommended |
| **Ambiguous top-2** — score gap < 0.05 | Auto-match downgraded to human review |
| **Domain shift** — deployment cameras unlike training cameras | Low similarities across the whole gallery; flagged, but the real fix is retraining on local imagery |
| **Class imbalance** — a few individuals dominate | Reported in the dataset summary; ArcFace helps but cannot invent data |
| **Model/preprocessing drift** | `model_version` and `preprocessing_version` are stored per embedding and used to filter searches |

Two points worth stating plainly:

* **Cosine similarity is not biological certainty.** A high score is evidence,
  weighed against crop quality, gallery depth and flank agreement. Auto-match
  exists to save reviewer time, not to replace the reviewer.
* **A false auto-match corrupts the data.** Two tigers recorded as one distorts
  population counts and movement history. That is why calibration optimises the
  false-accept rate first.

## 13. Public datasets

This project does not download or redistribute any dataset. If you obtain one
legitimately (e.g. ATRW — Amur Tiger Re-identification in the Wild, ~92
individuals), the loader adapts: arrange crops as `IDENTITY/*.jpg` or write a CSV
with `image_path,identity_id`. ATRW is Amur tigers under different conditions to
Pench, so expect domain shift and treat it as pretraining, not a substitute for
local data.

## 14. Running the tests

```bash
cd backend && python -m pytest tests/test_reid_dataset.py tests/test_reid_model.py tests/test_reid_training.py -v
```

These cover dataset discovery, split reproducibility, sequence-leakage
prevention, `[B, 512]` output shape, L2 normalisation, checkpoint round-trip,
one-batch loss decrease, resume, Rank-k/mAP correctness, threshold calibration,
embedding extraction and the encoder ↔ checkpoint integration. They use the
`tiny` backbone so no pretrained weights are downloaded.
