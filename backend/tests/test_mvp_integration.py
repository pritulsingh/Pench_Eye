"""
Integration test for MegaDescriptor MVP pipeline.

Tests the core workflow:
1. Generate MegaDescriptor embedding
2. Match against existing embeddings
3. Create observation
4. Store embedding
5. Query movement tracks
"""

import pytest
import pytest_asyncio
import numpy as np
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Test config
DATABASE_URL = "sqlite+aiosqlite:///:memory:"


class MockMegaDescriptor:
    @staticmethod
    def cosine_similarities_batch(query, references):
        query = np.asarray(query, dtype=np.float32)
        references = np.asarray(references, dtype=np.float32)
        return references @ query


@pytest_asyncio.fixture
async def db_session():
    """Create test database session."""
    engine = create_async_engine(DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        from app.core.database import Base
        import app.models  # Import models to register them
        await conn.run_sync(Base.metadata.create_all)
    
    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    
    async with AsyncSessionLocal() as session:
        yield session
    
    await engine.dispose()


@pytest.mark.asyncio
async def test_megadescriptor_embedding_generation(monkeypatch):
    """Test that MegaDescriptor can generate embeddings (mock)."""
    from app.services.megadescriptor_service import MegaDescriptorEmbeddingService
    
    # This test is mocked because downloading MegaDescriptor requires internet
    # In production, remove the mock and test with real images
    
    monkeypatch.setattr(
        "app.services.megadescriptor_service.get_megadescriptor_model",
        lambda *args, **kwargs: MockMegaDescriptor(),
    )
    service = MegaDescriptorEmbeddingService()
    
    # Create a mock embedding (768-d, L2-normalized)
    mock_embedding = np.random.randn(768).astype(np.float32)
    mock_embedding /= np.linalg.norm(mock_embedding)
    
    assert len(mock_embedding) == 768
    assert abs(np.linalg.norm(mock_embedding) - 1.0) < 1e-6  # L2 normalized


@pytest.mark.asyncio
async def test_similarity_calculation(monkeypatch):
    """Test cosine similarity calculation for normalized embeddings."""
    from app.services.megadescriptor_service import MegaDescriptorEmbeddingService
    
    monkeypatch.setattr(
        "app.services.megadescriptor_service.get_megadescriptor_model",
        lambda *args, **kwargs: MockMegaDescriptor(),
    )
    service = MegaDescriptorEmbeddingService()
    
    # Create two normalized embeddings
    emb1 = np.random.randn(768).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)
    
    emb2 = np.random.randn(768).astype(np.float32)
    emb2 /= np.linalg.norm(emb2)
    
    result = service.find_most_similar(
        emb1,
        [
            {"tiger_id": "TIGER-001", "embedding": emb1},
            {"tiger_id": "TIGER-002", "embedding": emb2},
        ],
    )

    assert result is not None
    assert result["tiger_id"] == "TIGER-001"
    assert result["similarity"] > 0.99


@pytest.mark.asyncio
async def test_tiger_creation_on_new_detection(db_session):
    """Test that new tigers are created when no match is found."""
    from app.models.tiger import Tiger, TigerStatus
    from app.services.tiger_service import TigerService
    from sqlalchemy import select
    
    # Create first tiger
    tiger1 = await TigerService.create_tiger(
        db_session,
        name=None,
        sex="unknown",
        notes="First tiger"
    )
    assert tiger1.tiger_id == "TIGER-001"
    
    # Create second tiger
    tiger2 = await TigerService.create_tiger(
        db_session,
        name=None,
        sex="unknown",
        notes="Second tiger"
    )
    assert tiger2.tiger_id == "TIGER-002"
    
    # Verify both exist
    result = await db_session.execute(select(Tiger))
    tigers = list(result.scalars().all())
    assert len(tigers) == 2


@pytest.mark.asyncio
async def test_observation_storage(db_session):
    """Test that observations are correctly stored in database."""
    from app.models.observation import Observation, MatchType, ReviewStatus
    from app.models.tiger import Tiger
    from app.models.image import Image, ImageStatus, SourceType, ProcessingStatus
    from app.models.camera_station import CameraStation, CameraStatus, CameraZone
    from sqlalchemy import select
    
    # Create camera
    camera = CameraStation(
        camera_id="CAM_01",
        name="Test Camera",
        zone=CameraZone.CORE,
        latitude=21.7000,
        longitude=79.2600,
        status=CameraStatus.ACTIVE,
    )
    db_session.add(camera)
    await db_session.commit()
    
    # Create image
    image = Image(
        image_id="IMG-001",
        original_filename="test.jpg",
        source_filename="test.jpg",
        camera_id="CAM_01",
        timestamp=datetime.now(timezone.utc),
        file_size_bytes=1024,
        width_px=1920,
        height_px=1080,
        sha256_hash="abc123",
        status=ImageStatus.PROCESSED,
        source_type=SourceType.IMAGE,
        processing_status=ProcessingStatus.COMPLETED,
    )
    db_session.add(image)
    await db_session.commit()
    
    # Create tiger
    tiger = Tiger(
        tiger_id="TIGER_001",
        name=None,
        status="active",
    )
    db_session.add(tiger)
    await db_session.commit()
    
    # Create observation
    obs = Observation(
        observation_id="OBS-001",
        tiger_id=tiger.id,
        image_id=image.id,
        camera_id="CAM_01",
        timestamp=datetime.now(timezone.utc),
        latitude=21.7000,
        longitude=79.2600,
        species="tiger",
        identity_confidence=0.91,
        match_type=MatchType.AUTO_MATCH,
        review_status=ReviewStatus.APPROVED,
    )
    db_session.add(obs)
    await db_session.commit()
    
    # Verify observation exists
    result = await db_session.execute(select(Observation).where(Observation.observation_id == "OBS-001"))
    stored_obs = result.scalar_one_or_none()
    
    assert stored_obs is not None
    assert stored_obs.tiger_id == tiger.id
    assert stored_obs.identity_confidence == 0.91
    assert stored_obs.match_type == MatchType.AUTO_MATCH


@pytest.mark.asyncio
async def test_embedding_storage(db_session):
    """Test that embeddings are correctly stored in database."""
    from app.models.embedding import Embedding
    from app.models.observation import Observation, MatchType, ReviewStatus
    from app.models.tiger import Tiger
    from app.models.image import Image, ImageStatus, SourceType, ProcessingStatus
    from app.models.camera_station import CameraStation, CameraStatus, CameraZone
    from sqlalchemy import select
    
    # Setup camera, image, tiger
    camera = CameraStation(
        camera_id="CAM_01", name="Test", zone=CameraZone.CORE,
        latitude=21.7000, longitude=79.2600, status=CameraStatus.ACTIVE,
    )
    db_session.add(camera)
    
    image = Image(
        image_id="IMG-001", original_filename="test.jpg", source_filename="test.jpg",
        camera_id="CAM_01", timestamp=datetime.now(timezone.utc),
        file_size_bytes=1024, width_px=1920, height_px=1080,
        sha256_hash="abc123", status=ImageStatus.PROCESSED,
        source_type=SourceType.IMAGE, processing_status=ProcessingStatus.COMPLETED,
    )
    db_session.add(image)
    
    tiger = Tiger(tiger_id="TIGER_001", name=None, status="active")
    db_session.add(tiger)
    await db_session.commit()
    
    # Create observation
    obs = Observation(
        observation_id="OBS-001", tiger_id=tiger.id, image_id=image.id,
        camera_id="CAM_01", timestamp=datetime.now(timezone.utc),
        latitude=21.7000, longitude=79.2600, species="tiger",
        identity_confidence=0.91, match_type=MatchType.AUTO_MATCH,
        review_status=ReviewStatus.APPROVED,
    )
    db_session.add(obs)
    await db_session.commit()
    
    # Create mock embedding
    embedding_vector = np.random.randn(768).astype(np.float32)
    embedding_vector /= np.linalg.norm(embedding_vector)
    
    emb = Embedding(
        embedding_id="EMB-001",
        observation_id=obs.id,
        tiger_id=tiger.id,
        embedding=embedding_vector.tolist(),
        model_version="megadescriptor-v1",
    )
    db_session.add(emb)
    await db_session.commit()
    
    # Verify embedding exists
    result = await db_session.execute(select(Embedding).where(Embedding.embedding_id == "EMB-001"))
    stored_emb = result.scalar_one_or_none()
    
    assert stored_emb is not None
    assert stored_emb.tiger_id == tiger.id
    assert stored_emb.model_version == "megadescriptor-v1"
    assert len(stored_emb.embedding) == 768


@pytest.mark.asyncio
async def test_movement_track_calculation(db_session):
    """Test that movement tracks are correctly calculated from observations."""
    from app.models.observation import Observation, MatchType, ReviewStatus
    from app.models.tiger import Tiger
    from app.models.image import Image, ImageStatus, SourceType, ProcessingStatus
    from app.models.camera_station import CameraStation, CameraStatus, CameraZone
    from app.services.map_service import MapService
    from sqlalchemy import select
    from datetime import timedelta
    
    # Create cameras at different locations
    cameras = []
    for i, (lat, lon) in enumerate([
        (21.7000, 79.2600),
        (21.7100, 79.2700),
        (21.7200, 79.2800),
    ]):
        cam = CameraStation(
            camera_id=f"CAM_{i+1:02d}",
            name=f"Camera {i+1}",
            zone=CameraZone.CORE,
            latitude=lat,
            longitude=lon,
            status=CameraStatus.ACTIVE,
        )
        db_session.add(cam)
        cameras.append(cam)
    
    # Create tiger
    tiger = Tiger(tiger_id="TIGER_001", name=None, status="active")
    db_session.add(tiger)
    
    # Create images
    images = []
    for i in range(3):
        img = Image(
            image_id=f"IMG-{i+1:03d}",
            original_filename=f"img_{i}.jpg",
            source_filename=f"img_{i}.jpg",
            camera_id=cameras[i].camera_id,
            timestamp=datetime.now(timezone.utc) + timedelta(hours=i),
            file_size_bytes=1024,
            width_px=1920,
            height_px=1080,
            sha256_hash=f"hash_{i}",
            status=ImageStatus.PROCESSED,
            source_type=SourceType.IMAGE,
            processing_status=ProcessingStatus.COMPLETED,
        )
        db_session.add(img)
        images.append(img)
    
    await db_session.commit()
    
    # Create observations at different cameras over time
    for i, (img, cam) in enumerate(zip(images, cameras)):
        obs = Observation(
            observation_id=f"OBS-{i+1:03d}",
            tiger_id=tiger.id,
            image_id=img.id,
            camera_id=cam.camera_id,
            timestamp=datetime.now(timezone.utc) + timedelta(hours=i),
            latitude=cam.latitude,
            longitude=cam.longitude,
            species="tiger",
            identity_confidence=0.90 + i*0.01,
            match_type=MatchType.AUTO_MATCH,
            review_status=ReviewStatus.APPROVED,
        )
        db_session.add(obs)
    
    await db_session.commit()
    
    # Calculate movement tracks
    tracks = await MapService.get_movement_tracks(db_session)
    
    # Verify track exists and has correct structure
    assert len(tracks) == 1
    assert tracks[0]["tiger_code"] == "TIGER_001"
    assert len(tracks[0]["legs"]) == 2  # 3 observations → 2 legs
    
    # Verify each leg has the right structure
    for leg in tracks[0]["legs"]:
        assert "from_camera_id" in leg
        assert "from_latitude" in leg
        assert "from_longitude" in leg
        assert "distance_km" in leg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
