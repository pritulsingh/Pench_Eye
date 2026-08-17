from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from pydantic import BaseModel
from app.core.database import get_db
from app.schemas.observation import CandidateMatch
from app.services.reid_service import reid_service
from app.models.embedding import Embedding
from sqlalchemy import select

router = APIRouter()

class SearchRequest(BaseModel):
    image_id: str
    top_k: int = 10

@router.post("/similar", response_model=List[CandidateMatch])
async def search_similar(data: SearchRequest, db: AsyncSession = Depends(get_db)):
    # Find embedding for this image observation
    from app.models.observation import Observation
    result = await db.execute(
        select(Embedding).join(Observation, Observation.id == Embedding.observation_id)
        .where(Observation.image_id == data.image_id).limit(1)
    )
    embedding = result.scalar_one_or_none()
    
    if not embedding:
        raise HTTPException(status_code=404, detail="Embedding not found for this image")
        
    similar = await reid_service.search_similar(db, embedding.embedding, top_k=data.top_k)
    return [
        CandidateMatch(tiger_id=s['tiger_id'], score=s['score'], rank=s['rank'])
        for s in similar
    ]
