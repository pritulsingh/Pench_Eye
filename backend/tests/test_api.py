"""End-to-end API tests for Pench Eye (run against the ASGI app)."""
import pytest
from starlette.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_thresholds_configured():
    assert settings.AUTO_MATCH_THRESHOLD > settings.REVIEW_THRESHOLD
    assert settings.REVIEW_THRESHOLD > settings.NEW_INDIVIDUAL_THRESHOLD
    assert settings.BLANK_THRESHOLD > 0.5


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "ml_mode" in body


def test_dashboard_stats(client):
    r = client.get("/api/v1/dashboard/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_cameras"] >= 0
    assert data["camera_health"]["total"] == data["total_cameras"]
    assert isinstance(data["detection_trend"], list)


def test_map_overview_has_layers(client):
    data = client.get("/api/v1/map/overview").json()
    assert data["center"] and data["bounds"]
    assert isinstance(data["zones"], list)
    assert isinstance(data["cameras"], list)
    assert all("marker_state" in c for c in data["cameras"])


def test_cameras_list_and_detail(client):
    listing = client.get("/api/v1/cameras").json()
    if listing["total"] == 0:
        pytest.skip("No real camera stations in current database")
    camera_id = listing["items"][0]["camera_id"]

    detail = client.get(f"/api/v1/cameras/{camera_id}").json()
    assert detail["camera_id"] == camera_id
    assert "recent_detections" in detail
    assert "detection_timeline" in detail

    assert client.get("/api/v1/cameras/CAM-DOES-NOT-EXIST").status_code == 404


def test_tigers_profile_and_gallery(client):
    listing = client.get("/api/v1/tigers").json()
    if listing["total"] == 0:
        pytest.skip("No real tigers in current database")
    code = listing["items"][0]["tiger_id"]

    profile = client.get(f"/api/v1/tigers/{code}").json()
    assert profile["tiger_id"] == code
    assert "frequent_cameras" in profile

    assert client.get(f"/api/v1/tigers/{code}/observations").status_code == 200
    assert client.get(f"/api/v1/tigers/{code}/gallery").status_code == 200


def test_observation_filters(client):
    all_obs = client.get("/api/v1/observations", params={"limit": 5}).json()
    if not all_obs["items"]:
        pytest.skip("No real observations in current database")
    camera_id = next((o["camera_id"] for o in all_obs["items"] if o.get("camera_id")), None)
    if camera_id is None:
        pytest.skip("No observations with camera_id in current database")
    filtered = client.get(
        "/api/v1/observations", params={"camera_id": camera_id, "limit": 50}
    ).json()
    assert all(o["camera_id"] == camera_id for o in filtered["items"])


def test_alerts_and_analytics(client):
    assert client.post("/api/v1/alerts/evaluate").status_code == 200
    assert "items" in client.get("/api/v1/alerts").json()
    summary = client.get("/api/v1/alerts/summary").json()
    assert {"open", "acknowledged", "resolved"}.issubset(summary)

    analytics = client.get("/api/v1/analytics/overview", params={"days": 365}).json()
    assert isinstance(analytics["detections_by_hour"], list)
    assert len(analytics["detections_by_weekday"]) == 7


def test_reviews_listing(client):
    r = client.get("/api/v1/reviews")
    assert r.status_code == 200
    assert "items" in r.json()


def test_upload_rejects_non_image(client):
    r = client.post(
        "/api/v1/images/upload",
        files={"file": ("evil.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 422


def test_demo_api_is_not_mounted_in_production_app(client):
    r = client.post("/api/v1/demo/simulate", json={"count": 1})
    assert r.status_code == 404


def test_image_file_endpoint(client):
    images = client.get("/api/v1/images", params={"limit": 5}).json()["items"]
    if not images:
        pytest.skip("No real images in current database")
    r = client.get(f"/api/v1/images/{images[0]['image_id']}/file")
    assert r.status_code in {200, 404}
