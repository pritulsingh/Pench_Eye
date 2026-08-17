"""
Demo / simulation mode.

Generates a synthetic camera-trap capture for a real camera row and pushes it
through the *same* pipeline used by uploads, so a hackathon demo produces a
genuine end-to-end trace:

    simulated capture → triage → detection → identification → observation
    → alert rules → map / analytics / dashboard

All rows created here carry `is_demo=True`. Nothing in this module claims to be
live camera data.
"""
from __future__ import annotations

import io
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.camera_station import CameraStation, CameraStatus
from app.services.pipeline_service import PipelineService

SIMULATION_DISCLAIMER = (
    "Simulated capture — synthetic frame generated locally for demonstration. "
    "Not a live camera-trap image."
)


def synth_frame(seed: int, label: str) -> bytes:
    """Procedurally generate a small JPEG that looks like a night IR capture."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    width, height = 640, 400
    img = Image.new("RGB", (width, height), (18, 20, 18))
    draw = ImageDraw.Draw(img)

    # Ground / canopy bands
    for y in range(height):
        shade = 24 + int(26 * (y / height)) + rng.randint(-4, 4)
        draw.line([(0, y), (width, y)], fill=(shade, shade - 2, shade - 6))

    # Vegetation silhouettes
    for _ in range(26):
        x = rng.randint(0, width)
        h = rng.randint(40, 190)
        w = rng.randint(2, 6)
        draw.rectangle([x, height - h, x + w, height], fill=(12, 16, 12))

    # Animal-ish shape with stripes so detection crops are not uniform noise
    bx, by = rng.randint(120, 380), rng.randint(180, 250)
    bw, bh = rng.randint(150, 220), rng.randint(70, 110)
    draw.ellipse([bx, by, bx + bw, by + bh], fill=(196, 142, 62))
    for i in range(0, bw, 16):
        draw.line([(bx + i, by), (bx + i + 6, by + bh)], fill=(30, 24, 18), width=4)
    draw.ellipse([bx + bw - 30, by - 26, bx + bw + 24, by + 28], fill=(202, 150, 70))

    draw.rectangle([0, height - 22, width, height], fill=(0, 0, 0))
    draw.text((8, height - 16), label[:78], fill=(220, 220, 220))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


class SimulationService:
    @staticmethod
    async def pick_camera(db: AsyncSession, camera_id: Optional[str] = None) -> Optional[CameraStation]:
        if camera_id:
            return (
                await db.execute(select(CameraStation).where(CameraStation.camera_id == camera_id))
            ).scalar_one_or_none()

        cameras = list(
            (
                await db.execute(
                    select(CameraStation).where(CameraStation.status == CameraStatus.ACTIVE)
                )
            )
            .scalars()
            .all()
        )
        if not cameras:
            cameras = list((await db.execute(select(CameraStation))).scalars().all())
        return random.choice(cameras) if cameras else None

    @staticmethod
    async def simulate_capture(
        db: AsyncSession, camera_id: Optional[str] = None
    ) -> Dict[str, Any]:
        camera = await SimulationService.pick_camera(db, camera_id)
        if camera is None:
            return {"error": "No camera stations available. Seed demo data first."}

        now = datetime.now(timezone.utc)
        seed = int(now.timestamp() * 1000) % (2**31)
        label = f"DEMO {camera.camera_id} {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        content = synth_frame(seed, label)

        result = await PipelineService.process_image(
            db,
            content=content,
            filename=f"sim_{camera.camera_id}_{seed}.jpg",
            camera_id=camera.camera_id,
            captured_at=now,
            is_demo=True,
        )
        result.update(
            {
                "camera_id": camera.camera_id,
                "camera_name": camera.name,
                "zone": camera.zone.value if camera.zone else None,
                "latitude": camera.latitude,
                "longitude": camera.longitude,
                "captured_at": now,
                "disclaimer": SIMULATION_DISCLAIMER,
            }
        )
        return result

    @staticmethod
    async def simulate_batch(
        db: AsyncSession, count: int = 3, camera_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        count = max(1, min(count, 10))
        events = []
        for _ in range(count):
            events.append(await SimulationService.simulate_capture(db, camera_id))
        return events


simulation_service = SimulationService()
