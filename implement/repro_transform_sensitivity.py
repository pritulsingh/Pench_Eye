"""
Reproduction harness (READ-ONLY diagnostic — no code changes to the pipeline).

Measures how the CURRENT production embedding path responds to identity-
preserving transforms, using the exact same call the backend uses:

    ProductionInference.identify_frame  ->  MegaDescriptorEmbeddingService.get_embedding
    (which internally does MegaDescriptor.preprocess_image -> resize(224,224) -> ImageNet norm)

For each tiger we compute cosine(original, variant) for every transform, and we
also compute the cross-tiger similarity distribution (different individuals).

This tells us:
  - how much each transform moves the embedding (flip/rotation/crop)
  - whether same-tiger-transformed still ranks above different-tiger
  - whether a single fixed threshold can separate the two populations

Nothing here is written to the DB. Metrics are printed, not fabricated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.megadescriptor.model import MegaDescriptor  # noqa: E402

IMPL = Path(__file__).resolve().parent
TRANSFORMS = [
    "flip_horizontal",
    "flip_vertical",
    "rotate_ccw15",
    "rotate_cw15",
    "rotate_cw30",
    "crop90",
    "crop80",
    "crop70",
    "blur_light",
    "blur_heavy",
    "bright_up",
    "bright_down",
    "contrast_up",
    "contrast_down",
]


def embed(model: MegaDescriptor, path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return model.get_embedding(img)  # already L2-normalized (768,)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main() -> None:
    tigers = sorted(d for d in IMPL.iterdir() if d.is_dir() and d.name.startswith("tiger_"))
    if not tigers:
        print("No tiger_* folders found.")
        return

    print(f"Loading MegaDescriptor (this downloads/loads weights once)...")
    model = MegaDescriptor()
    print("Model ready.\n")

    originals: dict[str, np.ndarray] = {}
    variant_sims: dict[str, list[float]] = {t: [] for t in TRANSFORMS}

    for tdir in tigers:
        orig_path = tdir / "original.png"
        if not orig_path.exists():
            continue
        orig_emb = embed(model, orig_path)
        originals[tdir.name] = orig_emb
        for t in TRANSFORMS:
            p = tdir / f"{t}.png"
            if not p.exists():
                continue
            v = embed(model, p)
            variant_sims[t].append(cos(orig_emb, v))

    print("=== SAME-TIGER: cosine(original, transform) ===")
    print(f"{'transform':<18}{'n':>4}{'mean':>9}{'min':>9}{'max':>9}")
    same_tiger_all = []
    for t in TRANSFORMS:
        s = variant_sims[t]
        if not s:
            continue
        arr = np.array(s)
        same_tiger_all.extend(s)
        print(f"{t:<18}{len(s):>4}{arr.mean():>9.4f}{arr.min():>9.4f}{arr.max():>9.4f}")

    # Cross-tiger (different individuals) baseline using originals.
    names = list(originals)
    cross = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cross.append(cos(originals[names[i]], originals[names[j]]))
    cross_arr = np.array(cross) if cross else np.array([0.0])

    print("\n=== DIFFERENT-TIGER: cosine(original_i, original_j) ===")
    print(f"n={len(cross)} mean={cross_arr.mean():.4f} min={cross_arr.min():.4f} "
          f"max={cross_arr.max():.4f} p95={np.percentile(cross_arr,95):.4f}")

    # Focus on the geometric transforms we care about (flip / rotation / crop).
    geo = ["flip_horizontal", "rotate_ccw15", "rotate_cw15", "rotate_cw30",
           "crop90", "crop80", "crop70"]
    geo_sims = [x for t in geo for x in variant_sims[t]]
    geo_arr = np.array(geo_sims) if geo_sims else np.array([1.0])

    print("\n=== SEPARABILITY (geometric transforms vs different-tiger) ===")
    print(f"same-tiger geometric : n={len(geo_sims)} mean={geo_arr.mean():.4f} "
          f"min={geo_arr.min():.4f} p05={np.percentile(geo_arr,5):.4f}")
    print(f"different-tiger       : max={cross_arr.max():.4f} "
          f"p95={np.percentile(cross_arr,95):.4f}")
    gap = geo_arr.min() - cross_arr.max()
    print(f"worst-case margin (min same-geo - max diff) = {gap:.4f} "
          f"({'SEPARABLE' if gap > 0 else 'OVERLAP — single threshold cannot separate'})")

    print(f"\nCurrent HIGH_MATCH_THRESHOLD = 0.85")
    below = (geo_arr < 0.85).sum()
    print(f"geometric same-tiger pairs falling BELOW 0.85 (would be MISSED as new): "
          f"{below}/{len(geo_arr)} ({100*below/len(geo_arr):.0f}%)")


if __name__ == "__main__":
    main()
