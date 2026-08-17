from pathlib import Path

from PIL import Image

from ml.evaluation.validate_amur_dataset import build_report


def test_amur_validation_reports_real_files_and_duplicates(tmp_path: Path):
    root = tmp_path / "amur_tiger"
    (root / "images").mkdir(parents=True)
    image = Image.new("RGB", (8, 8), color="black")
    image.save(root / "images" / "TIGER_001_a.jpg")
    image.save(root / "images" / "TIGER_001_b.jpg")
    (root / "metadata.csv").write_text(
        "image_path,individual_id,camera_id,timestamp,sequence_id\n"
        "images/TIGER_001_a.jpg,TIGER_001,CAM_01,2026-01-01T00:00:00Z,SEQ_1\n"
        "images/TIGER_001_b.jpg,TIGER_001,CAM_02,2026-01-01T01:00:00Z,SEQ_2\n",
        encoding="utf-8",
    )

    report = build_report(root)
    assert report["usable_images"] == 2
    assert report["individuals"] == 1
    assert len(report["duplicate_groups"]) == 1
    assert report["missing_sequence_metadata"] == 0