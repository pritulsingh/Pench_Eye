# Example Re-ID dataset layout

This folder documents the layout `ml/reid/train.py` expects. **It contains no
images.** Copying this structure does not give you a dataset, and training on
placeholder files cannot produce a working tiger identifier.

See [`docs/reid_training.md`](../../docs/reid_training.md) for data requirements.

## Directory layout (preferred)

```
data/reid/
├── train/
│   ├── TIGER_001/
│   │   ├── CAM03_20240117_0001.jpg
│   │   ├── CAM03_20240117_0002.jpg
│   │   └── CAM11_20240402_0001.jpg
│   └── TIGER_002/
│       └── CAM07_20240118_0001.jpg
├── val/
│   └── TIGER_001/
│       └── CAM05_20240311_0001.jpg
└── test/
    └── TIGER_001/
        └── CAM09_20240520_0001.jpg
```

* One directory per individual; the directory name is the identity label.
* The `train/val/test` level is optional — omit it and the splitter divides the
  data for you.
* Filenames sharing a prefix before the trailing frame counter are treated as one
  capture sequence (`CAM03_20240117_0001` and `_0002` group together), so burst
  frames never straddle a split.
* Add `_left` / `_right` to filenames when the flank is known.

## Flat CSV annotations

`annotations.csv` in this folder shows the alternative format. Only `image_path`
and `identity_id` are required; `split`, `sequence_id` and `flank` are used when
present. Relative paths resolve against the CSV's own directory.

Train from a CSV by pointing `--data` at the file:

```bash
python -m ml.reid.train --data data/reid_example/annotations.csv --output ml/weights/tiger_reid
```

## Adapting an external dataset

Public sets such as ATRW (Amur Tiger Re-identification in the Wild) can be used
if you obtain them legitimately. Either rename directories to
`IDENTITY/*.jpg`, or generate a CSV mapping each crop to its identity. Note that
ATRW is Amur tigers under different conditions to Pench, so expect domain shift;
treat it as pretraining rather than a replacement for local imagery.
