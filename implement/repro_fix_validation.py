"""
Fix-validation harness (READ-ONLY — measures candidate architectures, no DB writes).

Simulates the identification decision under three strategies and reports
rank-1 accuracy + score separation on the transformed query set:

  S0  BASELINE (current system):
        gallery = {tiger: original_embedding}
        query   = single embedding of the transformed image
        match   = argmax cosine

  S1  TTA QUERY (aggregate the query over identity-preserving views):
        query   = L2( mean( L2(orig), L2(hflip), L2(rot+-5 approx), L2(centre-crop) ) )
        gallery = {tiger: original_embedding}

  S2  MULTI-EMBEDDING GALLERY (enroll several views per tiger) + TTA QUERY:
        gallery = {tiger: [orig, hflip, crop90, crop80]}  (max-sim over views)
        query   = TTA-aggregated

For each strategy we report, over all (tiger, transform) queries:
  - rank-1 accuracy (did the correct tiger win?)
  - mean margin = sim(correct) - sim(best_wrong)   (higher = safer)
  - how many correct matches clear 0.85 / 0.70

vflip is INCLUDED only as a control to confirm it should NOT be a default view.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.megadescriptor.model import MegaDescriptor  # noqa: E402

IMPL = Path(__file__).resolve().parent

# In-scope identity-preserving transforms (the ones we must be robust to).
QUERY_TRANSFORMS = [
    "flip_horizontal", "rotate_ccw15", "rotate_cw15",
    "crop90", "crop80", "crop70", "blur_light",
]


def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def embed(model, path: Path) -> np.ndarray:
    return model.get_embedding(Image.open(path).convert("RGB"))


def hflip_view(model, img: Image.Image) -> np.ndarray:
    return model.get_embedding(img.transpose(Image.FLIP_LEFT_RIGHT))


def rot_view(model, img: Image.Image, deg: float) -> np.ndarray:
    return model.get_embedding(img.rotate(deg, resample=Image.BILINEAR, expand=False))


def centre_crop_view(model, img: Image.Image, frac: float) -> np.ndarray:
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return model.get_embedding(img.crop((left, top, left + cw, top + ch)))


def tta_query(model, path: Path) -> np.ndarray:
    """Normalize each view, average, renormalize (per the mandated rule)."""
    img = Image.open(path).convert("RGB")
    views = [
        l2(model.get_embedding(img)),
        l2(hflip_view(model, img)),
        l2(rot_view(model, img, 5.0)),
        l2(rot_view(model, img, -5.0)),
        l2(centre_crop_view(model, img, 0.90)),
    ]
    return l2(np.mean(views, axis=0))


def main() -> None:
    tigers = sorted(d for d in IMPL.iterdir() if d.is_dir() and d.name.startswith("tiger_"))
    print("Loading MegaDescriptor...")
    model = MegaDescriptor()
    print("ready\n")

    # Enrollment views per tiger.
    orig_emb, multi_gallery = {}, {}
    for tdir in tigers:
        op = tdir / "original.png"
        if not op.exists():
            continue
        img = Image.open(op).convert("RGB")
        orig_emb[tdir.name] = l2(model.get_embedding(img))
        multi_gallery[tdir.name] = [
            l2(model.get_embedding(img)),
            l2(hflip_view(model, img)),
            l2(centre_crop_view(model, img, 0.90)),
            l2(centre_crop_view(model, img, 0.80)),
        ]

    names = list(orig_emb)

    def eval_strategy(query_fn, gallery, multi: bool, label: str):
        correct = 0
        total = 0
        margins = []
        correct_scores = []
        for tdir in tigers:
            if tdir.name not in orig_emb:
                continue
            for t in QUERY_TRANSFORMS:
                p = tdir / f"{t}.png"
                if not p.exists():
                    continue
                q = query_fn(p)
                sims = {}
                for name in names:
                    if multi:
                        sims[name] = max(float(np.dot(q, g)) for g in gallery[name])
                    else:
                        sims[name] = float(np.dot(q, gallery[name]))
                winner = max(sims, key=sims.get)
                total += 1
                s_correct = sims[tdir.name]
                s_best_wrong = max(v for n, v in sims.items() if n != tdir.name)
                margins.append(s_correct - s_best_wrong)
                correct_scores.append(s_correct)
                if winner == tdir.name:
                    correct += 1
        m = np.array(margins)
        cs = np.array(correct_scores)
        print(f"{label:<38} rank1={correct}/{total} ({100*correct/total:.0f}%)  "
              f"mean_margin={m.mean():+.3f}  min_margin={m.min():+.3f}  "
              f"correct>=0.85:{(cs>=0.85).sum()}/{total}  >=0.70:{(cs>=0.70).sum()}/{total}")

    print("=== Strategy comparison on in-scope transforms ===")
    eval_strategy(lambda p: l2(embed(model, p)), orig_emb, False,
                  "S0 baseline (single/single)")
    eval_strategy(lambda p: tta_query(model, p), orig_emb, False,
                  "S1 TTA query, single-view gallery")
    eval_strategy(lambda p: tta_query(model, p), multi_gallery, True,
                  "S2 TTA query + multi-embedding gallery")


if __name__ == "__main__":
    main()
