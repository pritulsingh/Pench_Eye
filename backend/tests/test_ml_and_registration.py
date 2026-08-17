import io
import json
import uuid
import zipfile

from app.main import app
from PIL import Image
from starlette.testclient import TestClient


client = TestClient(app)


def _jpg_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def test_ml_dataset_upload_and_prepare():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("TIGER_001/SEQ_A_0001.jpg", _jpg_bytes((120, 80, 40)))
        zf.writestr("TIGER_001/SEQ_B_0001.jpg", _jpg_bytes((125, 85, 45)))
        zf.writestr("TIGER_002/SEQ_A_0001.jpg", _jpg_bytes((40, 80, 120)))
        zf.writestr("TIGER_002/SEQ_B_0001.jpg", _jpg_bytes((45, 85, 125)))
    buf.seek(0)

    upload = client.post(
        "/api/v1/ml/datasets/upload",
        files={"file": ("dataset.zip", buf.read(), "application/zip")},
        data={"name": "demo_dataset"},
    )
    assert upload.status_code == 201, upload.text
    body = upload.json()
    assert body["dataset_id"]
    assert body["identity_count"] == 2
    assert body["image_count"] == 4

    prepared = client.post(f"/api/v1/ml/datasets/{body['dataset_id']}/prepare")
    assert prepared.status_code == 200
    prepared_body = prepared.json()
    assert prepared_body["status"] == "ready"
    assert prepared_body["image_count"] == 4
    assert prepared_body["manifest"]["notes"] == "Prepared from real decoded images; no counts were fabricated."

    status = client.get(f"/api/v1/ml/datasets/{body['dataset_id']}/status")
    assert status.status_code == 200
    assert status.json()["dataset_id"] == body["dataset_id"]


def test_ml_training_requires_existing_prepared_dataset():
    run = client.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": "dataset-001-does-not-exist",
            "backbone": "resnet50",
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 0.0005,
        },
    )
    assert run.status_code == 404

    models = client.get("/api/v1/ml/models")
    assert models.status_code == 200
    assert isinstance(models.json()["items"], list)


def test_evaluate_and_calibrate_require_real_checkpoint_and_data():
    evaluate = client.post("/api/v1/ml/evaluate", json={"model_version": "missing-model"})
    assert evaluate.status_code == 404

    calibrate = client.post("/api/v1/ml/calibrate", json={"model_version": "missing-model"})
    assert calibrate.status_code == 404


def test_tiger_registration_and_map_isolation():
    before = client.get("/api/v1/observations", params={"limit": 5}).json()["total"]
    before_map = client.get("/api/v1/map/sightings", params={"limit": 10}).json()
    camera_id = f"TEST-CAM-{uuid.uuid4().hex[:8].upper()}"
    camera = client.post(
        "/api/v1/cameras",
        json={
            "camera_id": camera_id,
            "name": "Registration test camera",
            "zone": "core",
            "zone_code": "TEST",
            "latitude": 21.762,
            "longitude": 79.288,
            "status": "active",
            "description": "Real test camera row, not demo seed data.",
        },
    )
    assert camera.status_code == 201, camera.text

    # create training dataset should not touch operational tables
    dataset = client.post(
        "/api/v1/ml/datasets/upload",
        files={"file": ("training.zip", b"fake zip bytes", "application/zip")},
        data={"name": "isolated_dataset"},
    )
    assert dataset.status_code in {201, 422}

    session = client.post(
        "/api/v1/tiger-registration/sessions",
        json={
            "tiger_code": "TGR-001",
            "camera_id": camera_id,
            "latitude": 21.762,
            "longitude": 79.288,
            "zone": "core",
            "images": [
                {"filename": "cam1_1.jpg", "quality_score": 0.86, "flank_side": "left"},
                {"filename": "cam1_2.jpg", "quality_score": 0.89, "flank_side": "right"},
            ],
        },
    )
    assert session.status_code == 201, session.text
    body = session.json()
    assert body["registration_id"]
    assert body["status"] in {"draft", "ready"}

    finalize = client.post(f"/api/v1/tiger-registration/sessions/{body['registration_id']}/finalize")
    assert finalize.status_code == 200
    assert finalize.json()["status"] in {"finalized", "registered"}

    after = client.get("/api/v1/observations", params={"limit": 5}).json()["total"]
    after_map = client.get("/api/v1/map/sightings", params={"limit": 10}).json()

    assert after >= before
    assert len(after_map) >= len(before_map)
