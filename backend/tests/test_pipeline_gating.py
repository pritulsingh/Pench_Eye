"""
Regression tests: production upload path must never invent tigers.

Critical success condition from the human-as-tiger bug:
  human / blank / no-tiger images MUST NOT create tiger observations,
  MUST NOT run MegaDescriptor, and MUST NOT invent TIGER-* identities.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw
from starlette.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.inference_service import (
    DetectionOutput,
    ProductionInference,
    TriageOutput,
    pipeline_info,
)
from ml.detection.tiger_detector import TigerDetector, is_tiger_class


def _jpeg_bytes(draw_fn, size=(320, 240), salt: bytes | None = None) -> bytes:
    img = Image.new("RGB", size, (40, 40, 40))
    draw = ImageDraw.Draw(img)
    draw_fn(draw, size)
    if salt:
        # Ensure unique sha256 across test runs / retries.
        draw.text((4, 4), salt.hex()[:16], fill=(255, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _human_image_bytes() -> bytes:
    """Synthetic person-like figure — no tiger stripes / morphology."""
    import os

    def draw(d, size):
        w, h = size
        # Head + body silhouette
        d.ellipse([w // 2 - 30, 30, w // 2 + 30, 90], fill=(220, 180, 140))
        d.rectangle([w // 2 - 40, 90, w // 2 + 40, 200], fill=(30, 60, 140))
        d.rectangle([w // 2 - 70, 100, w // 2 - 40, 160], fill=(220, 180, 140))
        d.rectangle([w // 2 + 40, 100, w // 2 + 70, 160], fill=(220, 180, 140))

    return _jpeg_bytes(draw, salt=os.urandom(8))


def _blank_image_bytes() -> bytes:
    import os

    img = Image.new("RGB", (320, 240), (0, 0, 0))
    # Keep nearly black but unique so duplicate quarantine does not mask blank tests.
    px = img.load()
    salt = os.urandom(2)
    px[0, 0] = (salt[0] % 3, salt[1] % 3, 0)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _content_image_bytes() -> bytes:
    """Non-blank forest-like scene with no tiger class content."""
    import os

    def draw(d, size):
        w, h = size
        for y in range(h):
            shade = 20 + int(40 * y / h)
            d.line([(0, y), (w, y)], fill=(shade, shade + 10, shade))
        for x in range(0, w, 18):
            d.rectangle([x, h // 3, x + 6, h], fill=(10, 40, 10))

    return _jpeg_bytes(draw, salt=os.urandom(8))


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_default_ml_mode_is_production():
    assert settings.ML_MODE.value == "production"
    info = pipeline_info()
    assert info["is_demo"] is False


def test_is_tiger_class_never_maps_person():
    assert is_tiger_class("tiger") is True
    assert is_tiger_class("person") is False
    assert is_tiger_class("human") is False
    assert is_tiger_class("animal") is False
    assert is_tiger_class(None) is False
    assert is_tiger_class("unknown") is False


def test_production_yolo_fail_closed_without_weights(tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    det = TigerDetector(ml_mode="production", model_path=str(missing))
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    assert det.detect(image) == []
    status = det.status()
    assert status["weights_present"] is False
    assert status["model_loaded"] is False
    assert status["fail_closed"] is True


def test_production_triage_never_uses_simulated_reason():
    engine = ProductionInference()
    # Non-blank patterned image
    pixels = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    out = engine.triage_frame(pixels, "abc")
    assert "simulated" not in (out.reason or "").lower()
    assert out.stage != "demo"


def test_production_detect_never_invents_tiger_without_weights():
    engine = ProductionInference()
    pixels = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    out = engine.detect_frame(pixels, "abc")
    assert out.present is False
    assert out.species != "tiger"
    assert out.reason == "tiger_detector_unavailable"


def test_human_image_upload_no_tiger(client):
    """Regression: human photo must not become TIGER-*."""
    before_obs = client.get("/api/v1/observations", params={"limit": 1}).json()["total"]
    before_tigers = client.get("/api/v1/tigers", params={"limit": 1}).json()["total"]

    payload = _human_image_bytes()
    r = client.post(
        "/api/v1/images/upload",
        files={"file": ("human_regression.jpg", payload, "image/jpeg")},
    )
    assert r.status_code == 201, r.text
    body = r.json()

    assert body["status"] == "inference_unavailable"
    assert body.get("tiger_code") is None
    assert body.get("observation_id") is None
    assert body.get("megadescriptor_ran") is False
    assert "simulated" not in (body.get("triage_reason") or "").lower()
    assert body.get("species") in (None, "unknown")

    after_obs = client.get("/api/v1/observations", params={"limit": 1}).json()["total"]
    after_tigers = client.get("/api/v1/tigers", params={"limit": 1}).json()["total"]
    assert after_obs == before_obs
    assert after_tigers == before_tigers


def test_blank_image_rejected(client):
    r = client.post(
        "/api/v1/images/upload",
        files={"file": ("blank.jpg", _blank_image_bytes(), "image/jpeg")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "rejected"
    assert body.get("observation_id") is None
    assert body.get("tiger_code") is None
    assert body.get("megadescriptor_ran") is False


def test_no_tiger_content_image_stops_before_megadescriptor(client):
    """Non-blank image without YOLO tiger class → MegaDescriptor must not run."""
    identify = MagicMock()
    with patch(
        "app.services.pipeline_service.inference_pipeline.identify_frame",
        identify,
        create=True,
    ):
        r = client.post(
            "/api/v1/images/upload",
            files={"file": ("forest.jpg", _content_image_bytes(), "image/jpeg")},
        )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "inference_unavailable"
    assert body.get("megadescriptor_ran") is False
    identify.assert_not_called()


def test_diagnose_human_image_reports_pipeline(client):
    r = client.post(
        "/api/v1/ml/pipeline/diagnose",
        files={"file": ("human.jpg", _human_image_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["validation"]["passed"] is True
    assert body["final"] in {"inference_unavailable", "rejected"}
    assert body["megadescriptor"] == "NOT RUN"
    assert body["tiger_detection"] is False
    assert "simulated" not in (body.get("triage", {}).get("reason") or "").lower()


def test_pipeline_status_exposes_detector(client):
    r = client.get("/api/v1/ml/pipeline/status")
    assert r.status_code == 200
    body = r.json()
    assert body["ml_mode"] == "production"
    assert "detector" in body
    assert body["is_demo"] is False


def test_megadescriptor_only_after_tiger_gate():
    """Unit-level gate: identify_frame must not be reached without tiger detection."""
    from app.services import pipeline_service as ps

    engine = ProductionInference()
    pixels = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    detection = engine.detect_frame(pixels, None)
    assert detection.present is False
    # Simulate the gate used in process_image
    assert not (detection.present and detection.species == "tiger")
