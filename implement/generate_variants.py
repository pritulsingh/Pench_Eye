"""Generate augmentation variants for each implement/tiger_NN original image.

Flat layout produced (no subfolders): all variants live directly inside the
tiger folder with clear, transformation-named filenames:

  tiger_NN/
    original.png
    flip_horizontal.png
    flip_vertical.png
    rotate_ccw15.png
    rotate_cw15.png
    rotate_cw30.png
    crop90.png
    crop80.png
    crop70.png
    blur_light.png
    blur_heavy.png
    bright_up.png
    bright_down.png
    contrast_up.png
    contrast_down.png

This script only writes local image files. It does NOT touch the database
and does NOT call the ML / Re-ID pipeline.

Run with the project venv python from the Pench_Eye root:
  .venv/bin/python implement/generate_variants.py
"""
import glob
import os
import shutil

from PIL import Image, ImageEnhance, ImageFilter

IMPLEMENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Subfolders from the previous layout that must be flattened/removed.
OLD_SUBDIRS = [
    "original",
    "flip_horizontal",
    "flip_vertical",
    "rotated",
    "cropped",
    "blurred",
    "contrast",
]


def center_crop(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def build_variants(img: Image.Image):
    """Return {filename_stem: image} of reasonable variants."""
    return {
        "flip_horizontal": img.transpose(Image.FLIP_LEFT_RIGHT),
        "flip_vertical": img.transpose(Image.FLIP_TOP_BOTTOM),
        "rotate_ccw15": img.rotate(15, expand=True, fillcolor=(0, 0, 0)),
        "rotate_cw15": img.rotate(-15, expand=True, fillcolor=(0, 0, 0)),
        "rotate_cw30": img.rotate(-30, expand=True, fillcolor=(0, 0, 0)),
        "crop90": center_crop(img, 0.90),
        "crop80": center_crop(img, 0.80),
        "crop70": center_crop(img, 0.70),
        "blur_light": img.filter(ImageFilter.GaussianBlur(radius=1.5)),
        "blur_heavy": img.filter(ImageFilter.GaussianBlur(radius=3.0)),
        "bright_up": ImageEnhance.Brightness(img).enhance(1.3),
        "bright_down": ImageEnhance.Brightness(img).enhance(0.75),
        "contrast_up": ImageEnhance.Contrast(img).enhance(1.4),
        "contrast_down": ImageEnhance.Contrast(img).enhance(0.7),
    }


def locate_original(folder: str):
    """Find the untouched source image, whether flat or in an original/ subdir."""
    candidates = (
        glob.glob(os.path.join(folder, "original", "*"))
        + glob.glob(os.path.join(folder, "original_*"))
        + glob.glob(os.path.join(folder, "original.*"))
    )
    return candidates[0] if candidates else None


def main():
    folders = sorted(glob.glob(os.path.join(IMPLEMENT_DIR, "tiger_*")))
    per_tiger = {}
    for folder in folders:
        original_path = locate_original(folder)
        if not original_path:
            print(f"[skip] no original in {folder}")
            continue

        ext = os.path.splitext(original_path)[1].lower() or ".png"
        img = Image.open(original_path).convert("RGB")

        # Preserve the untouched original as original.<ext> in the folder root.
        dst_original = os.path.join(folder, f"original{ext}")
        if os.path.abspath(original_path) != os.path.abspath(dst_original):
            shutil.copy2(original_path, dst_original)

        # Write flat variants.
        count = 0
        for stem, variant in build_variants(img).items():
            variant.convert("RGB").save(os.path.join(folder, f"{stem}.png"))
            count += 1

        # Remove old subfolders (and any files left in them).
        for sub in OLD_SUBDIRS:
            sub_path = os.path.join(folder, sub)
            if os.path.isdir(sub_path):
                shutil.rmtree(sub_path)

        # Remove any legacy flat original_*.png that isn't our canonical original.
        for legacy in glob.glob(os.path.join(folder, "original_*")):
            if os.path.abspath(legacy) != os.path.abspath(dst_original):
                os.remove(legacy)

        per_tiger[os.path.basename(folder)] = count
        print(f"[ok] {os.path.basename(folder)}: original{ext} + {count} variants")

    if per_tiger:
        vals = sorted(set(per_tiger.values()))
        print(
            f"Processed {len(per_tiger)} folders; per-tiger variant count(s): {vals} "
            f"(+1 original each)"
        )


if __name__ == "__main__":
    main()
