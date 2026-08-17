"""
Camera-trap image pipeline.

    upload/simulated capture
      → validation
      → preprocessing + hashing
      → triage (blank / duplicate)
      → detection
      → identification (individual tiger)
      → observation row
      → alert rules
      → map / analytics / dashboard read from the same rows

The ML stages are all behind `inference_service.InferencePipeline`, so the demo
and production paths differ only in which object is injected.
"""
from __future__ import annotations

import hashlib
import io
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.camera_station import CameraStation
from app.models.image import Image, ImageStatus, ProcessingStatus, SourceType
from app.models.observation import DetectionType, MatchType, Observation, ReviewStatus, FlankSide
from app.models.review_queue import QueueStatus, ReviewQueue
from app.models.tiger import Tiger, TigerStatus
from app.services.alert_service import AlertService
from app.services.inference_service import inference_pipeline
from app.services.storage_service import storage_service
from app.services.tiger_service import TigerService


class ImageValidationError(ValueError):
    """Raised for rejected uploads (bad extension, too large, unreadable)."""


def _megadescriptor_input(pixels: Any, bbox: Optional[List[int]]) -> Any:
    """Return an RGB tiger crop when the detector supplied a valid box."""
    if pixels is None or not bbox or len(bbox) != 4:
        return None
    height, width = pixels.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = pixels[y1:y2, x1:x2]
    if getattr(crop, "size", 0) == 0:
        return None
    # decode_image returns BGR for detector compatibility; MegaDescriptor's
    # ndarray interface expects RGB.
    return crop[:, :, ::-1].copy() if getattr(crop, "ndim", 0) == 3 else crop


def sanitize_filename(filename: Optional[str]) -> str:
    """Keep only the basename and a safe character set; never trust client paths."""
    if not filename:
        return "upload.jpg"
    base = os.path.basename(filename.replace("\\", "/"))
    safe = "".join(c for c in base if c.isalnum() or c in ("-", "_", ".", " ")).strip()
    return safe[:180] or "upload.jpg"


def validate_upload(filename: Optional[str], content: bytes) -> str:
    if not content:
        raise ImageValidationError("Empty file upload.")
    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            f"File exceeds the {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )
    safe_name = sanitize_filename(filename)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in settings.allowed_upload_extensions:
        raise ImageValidationError(
            f"Unsupported file type '{ext or 'unknown'}'. Allowed: "
            f"{', '.join(settings.allowed_upload_extensions)}"
        )
    return safe_name


def decode_image(content: bytes) -> Tuple[Optional[Any], int, int]:
    """Verify the bytes really are an image and return (pixels, width, height)."""
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(content)) as img:
            img.verify()
        with PILImage.open(io.BytesIO(content)) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            try:
                import numpy as np

                pixels = np.array(rgb)[:, :, ::-1].copy()
            except Exception:
                pixels = None
        return pixels, width, height
    except Exception as exc:
        raise ImageValidationError("File is not a readable image.") from exc


class PipelineService:
    @staticmethod
    async def known_tiger_codes(db: AsyncSession, *, include_demo: bool = False) -> List[str]:
        query = select(Tiger.tiger_id).order_by(Tiger.tiger_id)
        if not include_demo:
            query = query.where(Tiger.is_demo.is_(False))
        rows = (await db.execute(query)).all()
        return [r[0] for r in rows]

    @staticmethod
    async def process_image(
        db: AsyncSession,
        *,
        content: bytes,
        filename: Optional[str],
        camera_id: Optional[str] = None,
        captured_at: Optional[datetime] = None,
        is_demo: Optional[bool] = None,
    ) -> Dict[str, Any]:
        safe_name = validate_upload(filename, content)
        pixels, width, height = decode_image(content)

        sha256 = hashlib.sha256(content).hexdigest()
        # Real uploads must never ride the hash-seeded DemoInference path.
        if is_demo is None:
            demo_flag = False
        else:
            demo_flag = bool(is_demo)

        if not demo_flag and inference_pipeline.is_demo:
            raise ImageValidationError(
                "Real image uploads require ML_MODE=production. "
                "Demo inference invents tiger identities from file hashes and is "
                "disabled for /images/upload. Use /api/v1/demo/simulate for demos."
            )

        camera: Optional[CameraStation] = None
        if camera_id:
            camera = (
                await db.execute(
                    select(CameraStation).where(CameraStation.camera_id == camera_id)
                )
            ).scalar_one_or_none()
            if camera is None:
                raise ImageValidationError(f"Unknown camera '{camera_id}'.")
            if not demo_flag and camera.is_demo:
                raise ImageValidationError(
                    f"Camera '{camera_id}' is synthetic demo metadata and cannot be used "
                    "for a production upload."
                )

        # Deduplicate per camera only: the identical frame arriving on a
        # different camera is a legitimate separate sighting (same tiger seen at
        # another trap) and must flow through to an observation + map marker.
        dup_query = select(Image).where(Image.sha256_hash == sha256)
        if camera:
            dup_query = dup_query.where(Image.camera_id == camera.camera_id)
        else:
            dup_query = dup_query.where(Image.camera_id.is_(None))
        duplicate = (await db.execute(dup_query.limit(1))).scalar_one_or_none()

        if demo_flag and not hasattr(inference_pipeline, "triage_frame"):
            triage = inference_pipeline.triage(image_hash=sha256, width=width, height=height)
        elif hasattr(inference_pipeline, "triage_frame"):
            triage = inference_pipeline.triage_frame(pixels, sha256)
        else:
            raise ImageValidationError("Triage pipeline is not available.")

        if duplicate is not None:
            status = ImageStatus.QUARANTINED
            reason = f"duplicate_of_{duplicate.image_id}"
        elif triage.is_blank:
            status = ImageStatus.QUARANTINED
            reason = triage.reason or "blank"
        else:
            status = ImageStatus.TRIAGED
            reason = triage.reason or "subject_detected"

        timestamp = captured_at or datetime.now(timezone.utc)
        image = Image(
            image_id=f"IMG-{uuid.uuid4().hex[:12].upper()}",
            original_filename=safe_name,
            source_filename=safe_name,
            camera_id=camera.camera_id if camera else None,
            timestamp=timestamp,
            latitude=camera.latitude if camera else None,
            longitude=camera.longitude if camera else None,
            file_size_bytes=len(content),
            width_px=width,
            height_px=height,
            sha256_hash=sha256,
            quality_score=triage.quality_score,
            blank_probability=triage.blank_probability,
            blank_threshold_used=settings.BLANK_THRESHOLD,
            triage_reason=reason,
            status=status,
            source_type=SourceType.IMAGE,
            processing_status=ProcessingStatus.PROCESSING,
            is_demo=demo_flag,
        )
        db.add(image)
        await db.commit()
        await db.refresh(image)

        prefix = "quarantine" if status == ImageStatus.QUARANTINED else "active"
        key = f"{prefix}/{image.image_id}.jpg"
        try:
            await storage_service.upload_image(content, key)
            if status == ImageStatus.QUARANTINED:
                image.quarantine_key = key
            else:
                image.storage_key = key
        except Exception as exc:
            image.error_message = f"storage_failed: {exc}"

        result: Dict[str, Any] = {
            "image_id": image.image_id,
            "status": status.value,
            "blank_probability": triage.blank_probability,
            "is_blank": bool(triage.is_blank),
            "triage_reason": reason,
            "triage_stage": getattr(triage, "stage", None),
            "observation_id": None,
            "tiger_code": None,
            "similarity": None,
            "identity_confidence": None,  # legacy alias of similarity; not a calibrated probability
            "decision": None,
            "candidate_tiger": None,
            "alerts_created": 0,
            "megadescriptor_ran": False,
            "message": "Image quarantined" if status == ImageStatus.QUARANTINED else "Image processed",
        }

        if status == ImageStatus.QUARANTINED:
            result.update(
                status="rejected",
                reason="duplicate_image" if duplicate is not None else "blank_or_irrelevant_image",
                message="Image rejected by triage",
            )
            image.processing_status = ProcessingStatus.COMPLETED
            await db.commit()
            return result

        if demo_flag and not hasattr(inference_pipeline, "detect_frame"):
            detection = inference_pipeline.detect(image_hash=sha256)
        elif hasattr(inference_pipeline, "detect_frame"):
            detection = inference_pipeline.detect_frame(pixels, sha256)
        else:
            raise ImageValidationError("Detection pipeline is not available.")

        if not detection.present or detection.species != "tiger":
            image.processing_status = ProcessingStatus.COMPLETED
            det_reason = getattr(detection, "reason", None) or "No tiger detected in image"
            detector_unavailable = det_reason == "tiger_detector_unavailable"
            result.update(
                status="inference_unavailable" if detector_unavailable else "no_tiger_detected",
                reason=det_reason,
                message=(
                    "Tiger detector unavailable; no downstream inference or database observation ran"
                    if detector_unavailable
                    else "No tiger detected"
                ),
                species=detection.species if detection.present else None,
                detection_confidence=detection.confidence,
                raw_detections=getattr(detection, "raw_detections", []),
                megadescriptor_ran=False,
            )
            await db.commit()
            return result

        tiger_crop = _megadescriptor_input(pixels, detection.bbox)
        if tiger_crop is None:
            image.processing_status = ProcessingStatus.COMPLETED
            result.update(
                status="tiger_crop_invalid",
                reason="valid_tiger_crop_required",
                species=None,
                detection_confidence=detection.confidence,
                raw_detections=getattr(detection, "raw_detections", []),
                message="Tiger detection did not produce a valid crop",
            )
            await db.commit()
            return result

        pipeline_result = await PipelineService._create_observation_from_detection(
            db,
            image=image,
            camera=camera,
            detection=detection,
            image_hash=sha256,
            timestamp=timestamp,
            demo_flag=demo_flag,
            pixels=tiger_crop,
        )
        image.processing_status = ProcessingStatus.COMPLETED
        image.status = ImageStatus.PROCESSED
        await db.commit()

        result.update(pipeline_result)
        return result

    @staticmethod
    async def _identify(
        db: AsyncSession, *, pixels: Any, image_hash: str, demo_flag: bool = False
    ) -> tuple[Any, Optional[Dict[str, Any]], Optional[str]]:
        """
        Run MegaDescriptor identification (MVP).

        Only called after a confirmed tiger detection + crop.
        """
        if hasattr(inference_pipeline, "identify_frame") and not inference_pipeline.is_demo:
            from app.services.inference_service import MegaDescriptorUnavailable

            try:
                identity = inference_pipeline.identify_frame(
                    pixels,
                    image_hash,
                    await PipelineService.known_tiger_codes(db, include_demo=demo_flag),
                )
            except MegaDescriptorUnavailable as exc:
                return None, None, str(exc)

            from app.models.embedding import Embedding
            import numpy as np

            # Gallery search excludes demo embeddings for real uploads.
            emb_query = (
                select(Embedding.embedding, Embedding.tiger_id, Tiger.is_demo, Observation.is_demo)
                .join(Tiger, Tiger.id == Embedding.tiger_id, isouter=True)
                .join(Observation, Observation.id == Embedding.observation_id)
                .where(Embedding.tiger_id.isnot(None))
            )
            existing = (await db.execute(emb_query)).all()

            similarity_result = None
            if existing and identity.embedding:
                query_embedding = np.asarray(identity.embedding, dtype=np.float32)
                query_norm = float(np.linalg.norm(query_embedding))
                if query_embedding.shape != (settings.EMBEDDING_DIM,) or not np.isfinite(query_norm):
                    return None, None, "MegaDescriptor returned an invalid embedding."
                if query_norm == 0:
                    return None, None, "MegaDescriptor returned a zero embedding."
                query_embedding = query_embedding / query_norm
                best_similarity = -1.0
                best_tiger_id = None

                for emb_vector, tiger_id, tiger_is_demo, observation_is_demo in existing:
                    if emb_vector is None:
                        continue
                    if not demo_flag and (bool(tiger_is_demo) or bool(observation_is_demo)):
                        continue
                    try:
                        ref_embedding = np.asarray(emb_vector, dtype=np.float32)
                        if ref_embedding.shape != query_embedding.shape:
                            continue
                        ref_norm = float(np.linalg.norm(ref_embedding))
                        if not np.isfinite(ref_norm) or ref_norm == 0:
                            continue
                        similarity = float(np.dot(query_embedding, ref_embedding / ref_norm))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_tiger_id = tiger_id
                    except Exception:
                        continue

                tiger = await db.get(Tiger, best_tiger_id) if best_tiger_id else None
                best_tiger_code = tiger.tiger_id if tiger else None
                candidate = {
                    "tiger_code": best_tiger_code,
                    "score": best_similarity,
                    "rank": 1,
                }
                if tiger and best_similarity >= settings.HIGH_MATCH_THRESHOLD:
                    similarity_result = {
                        "decision": "high_confidence_match",
                        "tiger_id": best_tiger_id,
                        "tiger_code": best_tiger_code,
                        "similarity": best_similarity,
                        "candidates": [candidate],
                    }
                elif tiger and best_similarity >= settings.REVIEW_THRESHOLD:
                    similarity_result = {
                        "decision": "review",
                        "tiger_id": best_tiger_id,
                        "tiger_code": best_tiger_code,
                        "similarity": best_similarity,
                        "candidates": [candidate],
                    }
                else:
                    similarity_result = {
                        "decision": "new_tiger",
                        "similarity": best_similarity if best_similarity > 0 else 0.0,
                        "candidates": [candidate] if tiger else [],
                    }
            else:
                similarity_result = {
                    "decision": "new_tiger",
                    "similarity": 0.0,
                    "candidates": [],
                }

            return identity, similarity_result, None

        # Explicit demo path only.
        codes = await PipelineService.known_tiger_codes(db, include_demo=True)
        return inference_pipeline.identify(image_hash=image_hash, known_tiger_codes=codes), None, None

    @staticmethod
    async def _create_observation_from_detection(
        db: AsyncSession,
        *,
        image: Image,
        camera: Optional[CameraStation],
        detection,
        image_hash: str,
        timestamp: datetime,
        demo_flag: bool,
        pixels: Any = None,
    ) -> Dict[str, Any]:
        is_tiger = detection.species == "tiger"
        identity = None
        tiger: Optional[Tiger] = None
        decision = "not_applicable"
        # Cosine similarity against gallery (NOT a calibrated probability).
        similarity: Optional[float] = None
        match_type = MatchType.DEMO if demo_flag else MatchType.NEW_INDIVIDUAL
        review_status = None
        flank = FlankSide.UNKNOWN
        gallery_decision: Optional[Dict[str, Any]] = None
        identity_error: Optional[str] = None
        megadescriptor_ran = False

        if is_tiger:
            identity, gallery_decision, identity_error = await PipelineService._identify(
                db, pixels=pixels, image_hash=image_hash, demo_flag=demo_flag
            )
            megadescriptor_ran = identity is not None and not getattr(identity, "is_demo", True)

            if identity_error:
                return {
                    "observation_id": None,
                    "tiger_code": None,
                    "similarity": None,
                    "decision": "identity_unavailable",
                    "candidate_tiger": None,
                    "species": None,
                    "alerts_created": 0,
                    "megadescriptor_ran": False,
                    "identity_error": identity_error,
                    "status": "embedding_failed",
                    "reason": "megadescriptor_failed",
                    "message": "Tiger embedding generation failed; no observation was created",
                }
            elif identity is not None:
                try:
                    flank = FlankSide(identity.flank_side)
                except ValueError:
                    flank = FlankSide.UNKNOWN

                if gallery_decision is not None:
                    decision = gallery_decision["decision"]
                    similarity = gallery_decision["similarity"]

                    if decision == "high_confidence_match" and gallery_decision.get("tiger_id"):
                        tiger = (
                            await db.execute(
                                select(Tiger).where(Tiger.id == gallery_decision["tiger_id"])
                            )
                        ).scalar_one_or_none()
                        if tiger:
                            match_type = MatchType.AUTO_MATCH
                            review_status = ReviewStatus.APPROVED
                    elif decision == "review":
                        review_status = ReviewStatus.PENDING_REVIEW
                        match_type = None
                    else:
                        decision = "new_tiger"
                        tiger = await TigerService.create_tiger(
                            db,
                            name=None,
                            sex="unknown",
                            notes="Auto-created by MegaDescriptor pipeline (no confident match).",
                        )
                        match_type = MatchType.NEW_INDIVIDUAL
                        review_status = ReviewStatus.APPROVED
                else:
                    # Demo identity carries a simulated similarity score.
                    similarity = identity.similarity
                    if identity.similarity >= settings.AUTO_MATCH_THRESHOLD and identity.suggested_tiger_code:
                        decision = "auto_match"
                        tiger = await TigerService.get_tiger(db, identity.suggested_tiger_code)
                        match_type = MatchType.AUTO_MATCH
                        review_status = ReviewStatus.APPROVED
                    elif identity.similarity >= settings.REVIEW_THRESHOLD:
                        decision = "human_review"
                        review_status = ReviewStatus.PENDING_REVIEW
                    else:
                        decision = "new_individual"
                        tiger = await TigerService.create_tiger(
                            db,
                            name=None,
                            sex="unknown",
                            notes="Auto-created by the identification pipeline (no confident match).",
                        )
                        match_type = MatchType.NEW_INDIVIDUAL
                        review_status = ReviewStatus.APPROVED

        observation = Observation(
            observation_id=f"OBS-{uuid.uuid4().hex[:10].upper()}",
            tiger_id=tiger.id if tiger else None,
            image_id=image.id,
            camera_id=camera.camera_id if camera else None,
            timestamp=timestamp,
            latitude=camera.latitude if camera else image.latitude,
            longitude=camera.longitude if camera else image.longitude,
            zone=camera.zone if camera else None,
            species=detection.species,
            detection_type=DetectionType(detection.detection_type)
            if detection.detection_type in DetectionType._value2member_map_
            else DetectionType.UNKNOWN,
            detection_confidence=detection.confidence,
            # Column name is historical; value is cosine similarity, not probability.
            identity_confidence=similarity,
            match_type=match_type if is_tiger else None,
            review_status=review_status,
            flank_side=flank,
            bounding_box_json={"bbox": detection.bbox} if detection.bbox else None,
            model_version=inference_pipeline.model_version,
            is_demo=demo_flag,
        )
        db.add(observation)
        await db.commit()
        await db.refresh(observation)

        if identity and identity.embedding and (demo_flag or not getattr(identity, "is_demo", True)):
            from app.models.embedding import Embedding

            db.add(
                Embedding(
                    embedding_id=f"EMB-{uuid.uuid4().hex[:12].upper()}",
                    observation_id=observation.id,
                    tiger_id=tiger.id if tiger else None,
                    embedding=identity.embedding,
                    model_version=identity.model_version,
                    flank_side=identity.flank_side,
                    is_demo=demo_flag,
                )
            )
            await db.commit()

        if decision in {"human_review", "review"} and identity is not None:
            candidates = (
                gallery_decision["candidates"]
                if gallery_decision is not None
                else identity.candidates
            )
            codes = [c.get("tiger_code") for c in candidates if c.get("tiger_code")]
            db.add(
                ReviewQueue(
                    review_id=f"REV-{uuid.uuid4().hex[:10].upper()}",
                    observation_id=observation.id,
                    candidate_tiger_ids=codes,
                    candidate_scores={
                        c["tiger_code"]: c.get("score") for c in candidates if c.get("tiger_code")
                    },
                    alternative_candidates_json=candidates,
                    status=QueueStatus.PENDING,
                )
            )
            await db.commit()

        if camera is not None:
            camera.last_active_at = timestamp
            camera.last_detection_at = timestamp
            await db.commit()

        if tiger is not None:
            await TigerService.update_tiger_stats(db, tiger.id)

        alerts = await AlertService.evaluate_detection(db, observation, camera, tiger)

        outcome = {
            "observation_id": observation.observation_id,
            "tiger_code": tiger.tiger_id if tiger else None,
            "similarity": similarity,
            "identity_confidence": similarity,  # legacy alias — value is cosine similarity
            "decision": decision,
            "candidate_tiger": gallery_decision.get("tiger_code") if gallery_decision else None,
            "species": detection.species,
            "alerts_created": len(alerts),
            "megadescriptor_ran": megadescriptor_ran,
            "message": f"Detection recorded ({detection.species})",
        }
        if identity_error:
            outcome["identity_error"] = identity_error
            outcome["message"] = (
                f"Detection recorded ({detection.species}) — individual identification "
                "unavailable, queued for human review."
            )
        if gallery_decision is not None:
            outcome["identity_reliability"] = gallery_decision.get("reliability")
            outcome["gallery_total"] = gallery_decision.get("gallery_total")
        if identity is not None and getattr(identity, "quality_warnings", None):
            outcome["identity_quality_warnings"] = list(identity.quality_warnings)
        return outcome


pipeline_service = PipelineService()
