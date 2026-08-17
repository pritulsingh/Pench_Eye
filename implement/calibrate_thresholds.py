"""
Threshold calibration harness (READ-ONLY — no DB writes).

Reproduces the *production* decision path (TTA-aggregated query + multi-embedding
gallery, max-cosine per identity, best/second-best margin) and sweeps
MATCH_THRESHOLD to report Precision / Recall / F1 and the false-match /
false-non-match rates that justify the chosen operating point.

Protocol (leave-transforms-out, closed + open set):
  * Enroll each tiger from a fixed set of views of its ORIGINAL image
    (original, hflip, crop90, crop80) — this is what the pipeline stores.
  * GENUINE queries: every in-scope transform of every tiger. The correct
    identity IS enrolled. A correct MATCH decision must pick that identity.
  * IMPOSTOR queries: to probe false matches we do leave-one-identity-out — for
    each transform query we also record the best score to a WRONG identity; a
    MATCH to a wrong identity is a false match (FMR).
  * OPEN-SET queries: hold out 2 identities entirely (not enrolled). Every one
    of their transform embeddings SHOULD be decided NEW. A MATCH here is a
    false accept of an unknown individual.

Reported per candidate MATCH_THRESHOLD (with fixed UNCERTAINTY_MARGIN):
  genuine:  MATCH (accept, correct id) / UNCERTAIN / NEW
  open-set: NEW (correct) / UNCERTAIN / MATCH (false accept)
  Precision = correct-genuine-MATCH / all-MATCH
  Recall    = correct-genuine-MATCH / genuine
  FMR       = (wrong-id MATCH on genuine + open-set MATCH) / non-target trials
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

QUERY_TRANSFORMS = [
    "flip_horizontal", "rotate_ccw15", "rotate_cw15",
    "crop90", "crop80", "crop70", "blur_light", "blur_heavy",
]
# Identities held out of the gallery to measure open-set (unknown) behaviour.
HELD_OUT = {"tiger_11", "tiger_12"}
UNCERTAINTY_MARGIN = 0.05


def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def hflip(model, img):
    return model.get_embedding(img.transpose(Image.FLIP_LEFT_RIGHT))


def rot(model, img, deg):
    return model.get_embedding(img.rotate(deg, resample=Image.BILINEAR, expand=False))


def crop(model, img, frac):
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return model.get_embedding(img.crop((left, top, left + cw, top + ch)))


def tta_query(model, path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    views = [
        l2(model.get_embedding(img)),
        l2(hflip(model, img)),
        l2(rot(model, img, 5.0)),
        l2(rot(model, img, -5.0)),
        l2(crop(model, img, 0.90)),
    ]
    return l2(np.mean(views, axis=0))


def decide(best, second, thr):
    margin = best - (second if second is not None else 0.0)
    if best >= thr and margin >= UNCERTAINTY_MARGIN:
        return "MATCH"
    if best >= 0.70:
        return "UNCERTAIN"
    return "NEW"


def main():
    tigers = sorted(d for d in IMPL.iterdir() if d.is_dir() and d.name.startswith("tiger_"))
    print("Loading MegaDescriptor...")
    model = MegaDescriptor()
    print("ready\n")

    gallery = {}
    for tdir in tigers:
        if tdir.name in HELD_OUT:
            continue
        op = tdir / "original.png"
        if not op.exists():
            continue
        img = Image.open(op).convert("RGB")
        gallery[tdir.name] = [
            l2(model.get_embedding(img)),
            l2(hflip(model, img)),
            l2(crop(model, img, 0.90)),
            l2(crop(model, img, 0.80)),
        ]
    names = list(gallery)

    # Precompute per-query best/second scores + whether target is enrolled.
    trials = []  # (true_name_or_None, best_name, best, second)
    for tdir in tigers:
        target = tdir.name if tdir.name in gallery else None
        for t in QUERY_TRANSFORMS:
            p = tdir / f"{t}.png"
            if not p.exists():
                continue
            q = tta_query(model, p)
            sims = {n: max(float(np.dot(q, g)) for g in gallery[n]) for n in names}
            ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
            best_name, best = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else None
            trials.append((target, best_name, best, second))

    genuine = [x for x in trials if x[0] is not None]
    openset = [x for x in trials if x[0] is None]

    print(f"genuine queries={len(genuine)}  open-set(unknown) queries={len(openset)}  "
          f"gallery identities={len(names)}  margin={UNCERTAINTY_MARGIN}\n")
    header = (f"{'thr':>5} | {'MATCH':>5} {'UNC':>4} {'NEW':>4} "
              f"{'TP':>4} {'FM':>3} | {'openMATCH':>9} {'openNEW':>7} | "
              f"{'Prec':>5} {'Rec':>5} {'F1':>5} {'FMR':>5}")
    print(header)
    print("-" * len(header))
    for thr in [0.70, 0.75, 0.80, 0.82, 0.85, 0.90]:
        gm = gu = gn = tp = fm = 0
        for target, best_name, best, second in genuine:
            d = decide(best, second, thr)
            if d == "MATCH":
                gm += 1
                if best_name == target:
                    tp += 1
                else:
                    fm += 1
            elif d == "UNCERTAIN":
                gu += 1
            else:
                gn += 1
        om = one = ou = 0
        for _, _, best, second in openset:
            d = decide(best, second, thr)
            if d == "MATCH":
                om += 1
            elif d == "NEW":
                one += 1
            else:
                ou += 1
        prec = tp / gm if gm else 0.0
        rec = tp / len(genuine) if genuine else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        # FMR: any MATCH to a non-target (wrong-id genuine + open-set accept).
        non_target = fm + om
        fmr = non_target / (len(genuine) + len(openset))
        print(f"{thr:>5.2f} | {gm:>5} {gu:>4} {gn:>4} {tp:>4} {fm:>3} | "
              f"{om:>9} {one:>7} | {prec:>5.2f} {rec:>5.2f} {f1:>5.2f} {fmr:>5.2f}")


if __name__ == "__main__":
    main()
