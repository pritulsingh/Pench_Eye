from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.services.inference_service import pipeline_info
from app.services.simulation_service import SIMULATION_DISCLAIMER, SimulationService

router = APIRouter()


class SimulationRequest(BaseModel):
    camera_id: Optional[str] = None
    count: int = 1


class SimulationStatus(BaseModel):
    demo_mode: bool
    ml_mode: str
    model_version: str
    is_demo_inference: bool
    disclaimer: str
    simulation_disclaimer: str
    geo_data_source: str
    # Individual Re-ID state — the UI must not imply real identification when
    # this is a demo encoder or is unavailable.
    reid_available: bool = True
    reid_model_version: Optional[str] = None
    reid_is_demo: bool = True
    reid_validated: Optional[bool] = None
    reid_known_identities: Optional[int] = None
    reid_error: Optional[str] = None


@router.get("/status", response_model=SimulationStatus)
async def demo_status():
    info = pipeline_info()
    reid = info.get("reid", {})
    return SimulationStatus(
        demo_mode=settings.is_demo_mode,
        ml_mode=info["ml_mode"],
        model_version=info["model_version"],
        is_demo_inference=info["is_demo"],
        disclaimer=info["disclaimer"],
        simulation_disclaimer=SIMULATION_DISCLAIMER,
        geo_data_source=settings.GEO_DATA_SOURCE,
        reid_available=bool(reid.get("available")),
        reid_model_version=reid.get("model_version"),
        reid_is_demo=bool(reid.get("is_demo", True)),
        reid_validated=reid.get("validated"),
        reid_known_identities=reid.get("known_identities"),
        reid_error=reid.get("error"),
    )


@router.post("/simulate")
async def simulate(data: SimulationRequest, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Generate synthetic camera-trap capture(s) and run the real pipeline on them.
    Rows created are flagged is_demo=True.
    """
    events: List[Dict[str, Any]] = await SimulationService.simulate_batch(
        db, count=data.count, camera_id=data.camera_id
    )
    return {"events": events, "disclaimer": SIMULATION_DISCLAIMER}
