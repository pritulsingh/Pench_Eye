"""Tests for the inference abstraction and the simulated pipeline path."""
import pytest

from app.services.inference_service import DemoInference, pipeline_info


def test_demo_inference_is_deterministic():
    a, b = DemoInference(), DemoInference()
    h = "a" * 64
    assert a.triage(image_hash=h) == b.triage(image_hash=h)
    assert a.detect(image_hash=h) == b.detect(image_hash=h)


def test_demo_identity_returns_normalized_embedding():
    engine = DemoInference()
    out = engine.identify(image_hash="deadbeef", known_tiger_codes=["TIGER-001", "TIGER-002"])
    assert len(out.embedding) == 768
    norm = sum(v * v for v in out.embedding) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)
    assert out.suggested_tiger_code in {"TIGER-001", "TIGER-002"}
    assert out.candidates and out.candidates[0]["rank"] == 1


def test_pipeline_info_flags_demo_clearly():
    info = pipeline_info()
    assert info["ml_mode"] in {"demo", "production"}
    if info["is_demo"]:
        assert "not a scientifically validated" in info["disclaimer"]
    else:
        assert info["is_demo"] is False
        assert "detector" in info


def test_upload_validation_rules():
    from app.services.pipeline_service import ImageValidationError, sanitize_filename, validate_upload

    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename(None) == "upload.jpg"

    with pytest.raises(ImageValidationError):
        validate_upload("script.exe", b"MZ")
    with pytest.raises(ImageValidationError):
        validate_upload("empty.jpg", b"")
    assert validate_upload("photo.JPG", b"\xff\xd8\xff") == "photo.JPG"
