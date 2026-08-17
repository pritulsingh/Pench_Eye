import io
import asyncio
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import uuid
import zipfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, get_db
from app.models.ml_dataset import MLDataset
from app.models.ml_model import MLModel
from app.models.training_run import TrainingRun

router = APIRouter()

ML_ROOT = Path(__file__).resolve().parents[4] / "ml"
DATASET_ROOT = ML_ROOT / "training_data" / "uploads"
EXTRACTED_ROOT = ML_ROOT / "training_data" / "extracted"
RUNS_ROOT = ML_ROOT / "training_data" / "runs"


def _ensure_ml_dirs() -> None:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    EXTRACTED_ROOT.mkdir(parents=True, exist_ok=True)
    (ML_ROOT / "training_data" / "prepared").mkdir(parents=True, exist_ok=True)
    (ML_ROOT / "training_data" / "manifests").mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    (ML_ROOT / "weights" / "tiger_reid").mkdir(parents=True, exist_ok=True)


def _safe_dataset_id(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "-" for ch in (name or "dataset")).strip("-")
    return f"{slug or 'dataset'}-{uuid.uuid4().hex[:8]}"


def _summarize_zip_dataset(payload: bytes) -> Dict[str, Any]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            identity_names = set()
            image_count = 0
            for entry in names:
                lower = entry.lower()
                if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")):
                    continue
                image_count += 1
                match = re.search(r"(?:^|/)([^/]+)/[^/]+$", entry)
                if match:
                    identity_names.add(match.group(1))
            return {
                "identity_count": len(identity_names),
                "image_count": image_count,
                "sequence_count": max(1, len(identity_names)),
                "manifest": {
                    "entries": len(names),
                    "images": image_count,
                    "identities": sorted(identity_names),
                    "archive_format": "zip",
                },
            }
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=422, detail=f"Invalid dataset archive: {exc}") from exc


def _resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[4] / path).resolve()
    return path


def _reset_dataset_extract_dir(dataset_id: str) -> Path:
    root = EXTRACTED_ROOT.resolve()
    target = (EXTRACTED_ROOT / dataset_id).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Unsafe dataset path")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _safe_extract_zip(payload: bytes, dataset_id: str) -> Path:
    target = _reset_dataset_extract_dir(dataset_id)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            destination = (target / member.filename).resolve()
            if target not in destination.parents:
                raise HTTPException(status_code=422, detail="Dataset archive contains an unsafe path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return target


def _count_images_per_identity(records) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.identity] = counts.get(record.identity, 0) + 1
    return counts


def _dataset_metrics_from_records(records, summary, splits) -> Dict[str, Any]:
    counts = list(_count_images_per_identity(records).values())
    split_counts = splits.counts()
    split_identity_counts = splits.identity_counts()
    return {
        "identity_count": summary.total_identities,
        "image_count": summary.total_images,
        "sequence_count": summary.total_sequences,
        "train_identities": split_identity_counts["train"],
        "train_images": split_counts["train"],
        "val_identities": split_identity_counts["val"],
        "val_images": split_counts["val"],
        "test_identities": split_identity_counts["test"],
        "test_images": split_counts["test"],
        "min_images_per_identity": min(counts) if counts else 0,
        "median_images_per_identity": int(statistics.median(counts)) if counts else 0,
        "mean_images_per_identity": float(sum(counts) / len(counts)) if counts else 0.0,
        "max_images_per_identity": max(counts) if counts else 0,
        "corrupted_images": len(summary.unreadable),
    }


def _checkpoint_for_model(model: Optional[MLModel], payload: Dict[str, Any]) -> Path:
    checkpoint = payload.get("checkpoint_path") or (model.checkpoint_path if model else None)
    if not checkpoint:
        raise HTTPException(status_code=422, detail="checkpoint_path or existing model_version is required")
    path = _resolve_project_path(checkpoint)
    if not path.is_file():
        raise HTTPException(status_code=422, detail=f"Checkpoint path does not exist: {path}")
    return path


def _dataset_path_for_payload(payload: Dict[str, Any], dataset: Optional[MLDataset]) -> Path:
    source = payload.get("data") or payload.get("dataset_path") or (dataset.extracted_path if dataset else None)
    if not source:
        raise HTTPException(status_code=422, detail="dataset_id with prepared data or dataset_path is required")
    path = _resolve_project_path(source)
    if not path.exists():
        raise HTTPException(status_code=422, detail=f"Dataset path does not exist: {path}")
    return path


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


async def _load_dataset_record(db: AsyncSession, dataset_id: str) -> MLDataset:
    dataset = (
        await db.execute(select(MLDataset).where(MLDataset.dataset_id == dataset_id))
    ).scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.post("/datasets/upload", status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    _ensure_ml_dirs()
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Dataset archive is empty")
    summary = _summarize_zip_dataset(data)
    dataset_id = _safe_dataset_id(name or file.filename or "dataset")
    target = DATASET_ROOT / f"{dataset_id}.zip"
    target.write_bytes(data)

    dataset = MLDataset(
        dataset_id=dataset_id,
        name=name or file.filename or dataset_id,
        source_path=str(target),
        status="uploaded",
        identity_count=summary["identity_count"],
        image_count=summary["image_count"],
        sequence_count=summary["sequence_count"],
        manifest_json={"source": file.filename or "upload", "type": "zip", **summary["manifest"]},
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    return {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "status": dataset.status,
        "identity_count": dataset.identity_count,
        "image_count": dataset.image_count,
        "sequence_count": dataset.sequence_count,
        "source_path": dataset.source_path,
    }


@router.get("/datasets")
async def list_datasets(db: AsyncSession = Depends(get_db), skip: int = 0, limit: int = 50):
    rows = (
        await db.execute(select(MLDataset).order_by(MLDataset.created_at.desc()).offset(skip).limit(limit))
    ).scalars().all()
    return {"items": [
        {
            "dataset_id": d.dataset_id,
            "name": d.name,
            "status": d.status,
            "identity_count": d.identity_count,
            "image_count": d.image_count,
            "sequence_count": d.sequence_count,
            "created_at": d.created_at,
        }
        for d in rows
    ], "total": len(rows)}


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    dataset = await _load_dataset_record(db, dataset_id)
    return {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "status": dataset.status,
        "identity_count": dataset.identity_count,
        "image_count": dataset.image_count,
        "sequence_count": dataset.sequence_count,
        "train_identities": dataset.train_identities,
        "train_images": dataset.train_images,
        "val_identities": dataset.val_identities,
        "val_images": dataset.val_images,
        "test_identities": dataset.test_identities,
        "test_images": dataset.test_images,
        "manifest": dataset.manifest_json,
    }


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    dataset = await _load_dataset_record(db, dataset_id)
    dataset.status = "deleted"
    await db.commit()
    return {"dataset_id": dataset_id, "status": "deleted"}


@router.post("/datasets/{dataset_id}/prepare")
async def prepare_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    dataset = await _load_dataset_record(db, dataset_id)
    source_path = Path(dataset.source_path or "")
    if not source_path.is_file():
        raise HTTPException(status_code=422, detail="Dataset archive is missing from storage")
    try:
        extracted = _safe_extract_zip(source_path.read_bytes(), dataset.dataset_id)
    except zipfile.BadZipFile as exc:
        dataset.status = "invalid"
        dataset.manifest_json = {**(dataset.manifest_json or {}), "prepared": False, "error": str(exc)}
        await db.commit()
        raise HTTPException(status_code=422, detail=f"Invalid dataset archive: {exc}") from exc

    try:
        from ml.reid.dataset import split_records
        from ml.reid.dataset.discovery import load_dataset

        records, summary = load_dataset(extracted, min_images_per_identity=1, verify_images=True)
        splits = split_records(records, val_fraction=0.2, test_fraction=0.1, seed=42)
    except Exception as exc:
        dataset.status = "invalid"
        dataset.extracted_path = str(extracted)
        dataset.manifest_json = {**(dataset.manifest_json or {}), "prepared": False, "error": str(exc)}
        await db.commit()
        raise HTTPException(status_code=422, detail=f"Dataset preparation failed: {exc}") from exc

    metrics = _dataset_metrics_from_records(records, summary, splits)
    for key, value in metrics.items():
        setattr(dataset, key, value)
    dataset.extracted_path = str(extracted)
    dataset.status = "ready" if records else "invalid"
    dataset.manifest_json = {
        **(dataset.manifest_json or {}),
        "prepared": bool(records),
        "archive_extracted_to": str(extracted),
        "identities": [
            {
                "identity": item.identity,
                "image_count": item.image_count,
                "sequence_count": item.sequence_count,
                "splits": item.splits,
            }
            for item in summary.identities
        ],
        "split_counts": splits.counts(),
        "split_identity_counts": splits.identity_counts(),
        "train_only_identities": splits.train_only_identities,
        "used_explicit_splits": splits.used_explicit_splits,
        "unreadable_files": [str(p) for p in summary.unreadable],
        "dropped_identities": [
            {"identity": identity, "image_count": count}
            for identity, count in summary.dropped_identities
        ],
        "notes": "Prepared from real decoded images; no counts were fabricated.",
    }
    await db.commit()
    await db.refresh(dataset)

    if not records:
        raise HTTPException(status_code=422, detail="Dataset contains no readable labelled images")

    return {
        "dataset_id": dataset.dataset_id,
        "status": dataset.status,
        "identity_count": dataset.identity_count,
        "image_count": dataset.image_count,
        "sequence_count": dataset.sequence_count,
        "prepared": True,
        "manifest": dataset.manifest_json,
    }


@router.get("/datasets/{dataset_id}/status")
async def get_dataset_status(dataset_id: str, db: AsyncSession = Depends(get_db)):
    dataset = await _load_dataset_record(db, dataset_id)
    return {
        "dataset_id": dataset.dataset_id,
        "status": dataset.status,
        "identity_count": dataset.identity_count,
        "image_count": dataset.image_count,
        "sequence_count": dataset.sequence_count,
        "prepared": dataset.status in {"ready", "training"},
    }


async def _run_training_process(run_id: str, command: List[str], output_dir: Path, log_path: Path) -> None:
    async with AsyncSessionLocal() as session:
        run = (
            await session.execute(select(TrainingRun).where(TrainingRun.run_id == run_id))
        ).scalar_one_or_none()
        if run is None:
            return
        run.status = "training"
        run.started_at = datetime.now(timezone.utc)
        await session.commit()

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return_code: Optional[int] = None
    error_message: Optional[str] = None
    try:
        with log_path.open("ab") as log_fh:
            log_fh.write(("COMMAND: " + " ".join(command) + "\n").encode("utf-8"))
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(Path(__file__).resolve().parents[4]),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            return_code = await process.wait()
    except Exception as exc:
        error_message = str(exc)
        return_code = -1

    async with AsyncSessionLocal() as session:
        run = (
            await session.execute(select(TrainingRun).where(TrainingRun.run_id == run_id))
        ).scalar_one_or_none()
        if run is None:
            return

        history = _read_json_if_exists(output_dir / "training_history.json")
        epochs = history.get("epochs") or []
        last_epoch = epochs[-1] if epochs else {}
        best_checkpoint = output_dir / "best.pt"
        latest_checkpoint = output_dir / "latest.pt"
        checkpoint = best_checkpoint if best_checkpoint.is_file() else latest_checkpoint

        run.current_epoch = int(last_epoch.get("epoch") or len(epochs) or run.current_epoch or 0)
        run.train_loss = last_epoch.get("train_loss")
        run.validation_loss = last_epoch.get("val_loss")
        run.rank1 = last_epoch.get("val_rank1")
        run.rank5 = last_epoch.get("val_rank5")
        run.map_value = last_epoch.get("val_map")
        run.completed_at = datetime.now(timezone.utc)

        if return_code == 0 and checkpoint.is_file():
            run.status = "completed"
            run.checkpoint_path = str(checkpoint)
            with suppress(Exception):
                from ml.reid.checkpoint import load_checkpoint

                payload = load_checkpoint(checkpoint)
                run.model_version = payload.model_version
                model = MLModel(
                    model_version=payload.model_version,
                    model_type="reid",
                    backbone=payload.model_config.backbone,
                    embedding_dimension=payload.model_config.embedding_dim,
                    checkpoint_path=str(checkpoint),
                    dataset_id=run.dataset_id,
                    training_run_id=run.run_id,
                    rank1=run.rank1,
                    rank5=run.rank5,
                    map=run.map_value,
                    status="trained",
                )
                existing = (
                    await session.execute(
                        select(MLModel).where(MLModel.model_version == payload.model_version)
                    )
                ).scalar_one_or_none()
                if existing is None:
                    session.add(model)
        else:
            run.status = "failed"
            tail = ""
            if log_path.is_file():
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            run.error_message = error_message or f"Training process exited with code {return_code}. {tail}".strip()
        dataset = (
            await session.execute(select(MLDataset).where(MLDataset.dataset_id == run.dataset_id))
        ).scalar_one_or_none()
        if dataset is not None:
            dataset.status = "ready"
        await session.commit()


@router.post("/train")
async def start_training(
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=422, detail="dataset_id is required")
    dataset = await _load_dataset_record(db, str(dataset_id))
    if dataset.status not in {"ready", "training"} or not dataset.extracted_path:
        raise HTTPException(status_code=422, detail="Dataset must be prepared before training")

    run_id = f"RUN-{uuid.uuid4().hex[:10].upper()}"
    output_dir = RUNS_ROOT / run_id
    log_path = output_dir / "training.log"
    backbone = str(payload.get("backbone") or "resnet50")
    epochs = int(payload.get("epochs") or 1)
    batch_size = int(payload.get("batch_size") or 32)
    learning_rate = float(payload.get("learning_rate") or payload.get("lr") or 0.0005)
    embedding_dim = int(payload.get("embedding_dimension") or payload.get("embedding_dim") or 512)
    device = str(payload.get("device") or "auto")

    command = [
        sys.executable,
        "-m",
        "ml.reid.train",
        "--data",
        str(dataset.extracted_path),
        "--output",
        str(output_dir),
        "--experiment-name",
        run_id,
        "--backbone",
        backbone,
        "--embedding-dim",
        str(embedding_dim),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(learning_rate),
        "--device",
        device,
    ]
    if payload.get("no_pretrained"):
        command.append("--no-pretrained")
    if payload.get("max_steps_per_epoch"):
        command.extend(["--max-steps-per-epoch", str(int(payload["max_steps_per_epoch"]))])

    run = TrainingRun(
        run_id=run_id,
        dataset_id=str(dataset_id),
        status="queued",
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hyperparameters={
            "dataset_id": dataset_id,
            "backbone": backbone,
            "embedding_dim": embedding_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "command": command,
            "log_path": str(log_path),
        },
        checkpoint_path=None,
    )
    dataset.status = "training"
    db.add(run)
    await db.commit()
    await db.refresh(run)
    background_tasks.add_task(_run_training_process, run.run_id, command, output_dir, log_path)
    return {
        "run_id": run.run_id,
        "dataset_id": dataset_id,
        "status": run.status,
        "backbone": run.backbone,
        "epochs": run.epochs,
        "learning_rate": run.learning_rate,
        "log_path": str(log_path),
    }


@router.get("/train/{run_id}")
async def get_training_run(run_id: str, db: AsyncSession = Depends(get_db)):
    run = (
        await db.execute(select(TrainingRun).where(TrainingRun.run_id == run_id))
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    return {
        "run_id": run.run_id,
        "dataset_id": run.dataset_id,
        "status": run.status,
        "current_epoch": run.current_epoch,
        "train_loss": run.train_loss,
        "validation_loss": run.validation_loss,
        "rank1": run.rank1,
        "rank5": run.rank5,
        "map": run.map_value,
        "checkpoint_path": run.checkpoint_path,
        "model_version": run.model_version,
        "backbone": run.backbone,
        "epochs": run.epochs,
        "error_message": run.error_message,
        "completed_at": run.completed_at,
    }


@router.get("/train/{run_id}/logs")
async def get_training_logs(run_id: str, db: AsyncSession = Depends(get_db)):
    run = (
        await db.execute(select(TrainingRun).where(TrainingRun.run_id == run_id))
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    log_path = None
    if run.hyperparameters:
        log_path = run.hyperparameters.get("log_path")
    lines = []
    if log_path and Path(log_path).is_file():
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    return {"run_id": run.run_id, "status": run.status, "log_path": log_path, "logs": lines[-500:]}


@router.post("/evaluate")
async def evaluate_model(payload: Dict[str, Any], db: AsyncSession = Depends(get_db)):
    model = None
    model_version = payload.get("model_version")
    if model_version:
        model = (
            await db.execute(select(MLModel).where(MLModel.model_version == str(model_version)))
        ).scalar_one_or_none()
        if model is None and not payload.get("checkpoint_path"):
            raise HTTPException(status_code=404, detail="Model not found")

    dataset = None
    dataset_id = payload.get("dataset_id") or (model.dataset_id if model else None)
    if dataset_id:
        dataset = await _load_dataset_record(db, str(dataset_id))

    checkpoint = _checkpoint_for_model(model, payload)
    data_path = _dataset_path_for_payload(payload, dataset)
    split = str(payload.get("split") or "test")
    output_path = RUNS_ROOT / f"evaluation-{uuid.uuid4().hex[:10]}.json"

    try:
        from ml.reid import evaluate as evaluate_module

        args = evaluate_module.build_parser().parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--data",
                str(data_path),
                "--split",
                split,
                "--batch-size",
                str(int(payload.get("batch_size") or 32)),
                "--num-workers",
                str(int(payload.get("num_workers") or 0)),
                "--device",
                str(payload.get("device") or "auto"),
                "--output",
                str(output_path),
            ]
            + (["--roc"] if payload.get("roc") else [])
        )
        result = await asyncio.to_thread(evaluate_module.evaluate, args)
    except SystemExit as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    measured = result.get("query_gallery") or result.get("leave_one_out") or {}
    if model is not None:
        model.rank1 = measured.get("rank1")
        model.rank5 = measured.get("rank5")
        model.rank10 = measured.get("rank10")
        model.map = measured.get("mean_ap")
        model.status = "evaluated" if result.get("evaluable") else "unvalidated"
        await db.commit()

    return {
        "status": "evaluated" if result.get("evaluable") else "not_evaluable",
        "model_version": result.get("model_version") or model_version,
        "checkpoint_path": str(checkpoint),
        "dataset_path": str(data_path),
        "metrics_path": str(output_path),
        "metrics": result,
    }


@router.post("/calibrate")
async def calibrate_model(payload: Dict[str, Any], db: AsyncSession = Depends(get_db)):
    model = None
    model_version = payload.get("model_version")
    if model_version:
        model = (
            await db.execute(select(MLModel).where(MLModel.model_version == str(model_version)))
        ).scalar_one_or_none()
        if model is None and not payload.get("checkpoint_path"):
            raise HTTPException(status_code=404, detail="Model not found")

    dataset = None
    dataset_id = payload.get("dataset_id") or (model.dataset_id if model else None)
    if dataset_id:
        dataset = await _load_dataset_record(db, str(dataset_id))

    checkpoint = _checkpoint_for_model(model, payload)
    data_path = _dataset_path_for_payload(payload, dataset)
    split = str(payload.get("split") or "val")
    output_path = RUNS_ROOT / f"thresholds-{uuid.uuid4().hex[:10]}.json"

    try:
        from ml.reid import calibrate_thresholds as calibrate_module

        args = calibrate_module.build_parser().parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--data",
                str(data_path),
                "--split",
                split,
                "--batch-size",
                str(int(payload.get("batch_size") or 32)),
                "--num-workers",
                str(int(payload.get("num_workers") or 0)),
                "--device",
                str(payload.get("device") or "auto"),
                "--target-far",
                str(float(payload.get("target_far") or 0.01)),
                "--review-recall",
                str(float(payload.get("review_recall") or 0.95)),
                "--new-individual-percentile",
                str(float(payload.get("new_individual_percentile") or 2.0)),
                "--output",
                str(output_path),
            ]
        )
        result = await asyncio.to_thread(calibrate_module.calibrate, args)
    except SystemExit as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Calibration failed: {exc}") from exc

    return {
        "status": "calibrated" if result.get("calibrated") else "not_calibrated",
        "checkpoint_path": str(checkpoint),
        "dataset_path": str(data_path),
        "thresholds_path": str(output_path),
        "calibration": result,
    }


@router.get("/models")
async def list_models(db: AsyncSession = Depends(get_db), skip: int = 0, limit: int = 50):
    rows = (
        await db.execute(select(MLModel).order_by(MLModel.created_at.desc()).offset(skip).limit(limit))
    ).scalars().all()
    return {"items": [{
        "model_version": row.model_version,
        "model_type": row.model_type,
        "backbone": row.backbone,
        "embedding_dimension": row.embedding_dimension,
        "status": row.status,
        "rank1": row.rank1,
        "map": row.map,
        "dataset_id": row.dataset_id,
        "checkpoint_path": row.checkpoint_path,
    } for row in rows], "total": len(rows)}


@router.post("/models/{model_id}/activate")
async def activate_model(model_id: str, db: AsyncSession = Depends(get_db)):
    model = (
        await db.execute(select(MLModel).where(MLModel.model_version == model_id))
    ).scalar_one_or_none()
    if model is None:
        model = (
            await db.execute(select(MLModel).where(MLModel.id == model_id))
        ).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    checkpoint = model.checkpoint_path
    if not checkpoint or not os.path.exists(checkpoint):
        raise HTTPException(status_code=400, detail="Checkpoint path does not exist")
    model.status = "active"
    model.activated_at = datetime.now(timezone.utc)
    await db.commit()
    return {"status": "active", "model_version": model.model_version, "checkpoint_path": checkpoint}


@router.get("/pipeline/status")
async def pipeline_status():
    """Development diagnostic: which models are loaded and whether detection is fail-closed."""
    from app.services.inference_service import pipeline_info

    return pipeline_info()


@router.post("/pipeline/diagnose")
async def diagnose_image(file: UploadFile = File(...)):
    """
    Development diagnostic for a single image.

    Runs validation → triage → YOLO (no DB writes, no MegaDescriptor unless tiger).
    Does not create observations or embeddings.
    """
    from app.core.config import settings
    from app.services.inference_service import inference_pipeline
    from app.services.pipeline_service import ImageValidationError, decode_image, validate_upload

    content = await file.read()
    try:
        safe_name = validate_upload(file.filename, content)
        pixels, width, height = decode_image(content)
    except ImageValidationError as exc:
        return {
            "image": file.filename,
            "validation": {"passed": False, "error": str(exc)},
            "final": "rejected",
        }

    report: Dict[str, Any] = {
        "image": safe_name,
        "ml_mode": settings.ML_MODE.value,
        "is_demo_inference": inference_pipeline.is_demo,
        "validation": {"passed": True, "width": width, "height": height},
        "megadescriptor_ran": False,
    }

    if inference_pipeline.is_demo:
        report["warning"] = "Real diagnostics require ML_MODE=production."
        report["triage"] = {"passed": False, "reason": "production_inference_required"}
        report["yolo"] = inference_pipeline.detector_status()
        report["detections"] = []
        report["tiger_detection"] = False
        report["megadescriptor"] = "NOT RUN"
        report["final"] = "inference_unavailable"
        return report

    triage = inference_pipeline.triage_frame(pixels, None)

    report["triage"] = {
        "passed": not triage.is_blank,
        "is_blank": triage.is_blank,
        "blank_probability": triage.blank_probability,
        "quality_score": triage.quality_score,
        "reason": triage.reason,
        "stage": getattr(triage, "stage", None),
    }
    report["yolo"] = (
        inference_pipeline.detector_status()
        if hasattr(inference_pipeline, "detector_status")
        else {"available": False}
    )
    if triage.is_blank:
        report["detections"] = []
        report["tiger_detection"] = False
        report["final"] = "rejected"
        report["megadescriptor"] = "NOT RUN"
        return report

    detection = inference_pipeline.detect_frame(pixels, None)
    report["detections"] = getattr(detection, "raw_detections", [])
    report["tiger_detection"] = bool(detection.present and detection.species == "tiger")
    report["detection"] = {
        "present": detection.present,
        "species": detection.species,
        "confidence": detection.confidence,
        "bbox": detection.bbox,
        "reason": getattr(detection, "reason", None),
    }

    if not (detection.present and detection.species == "tiger"):
        # Treat any non-tiger detection reason as inference unavailable for
        # diagnostic consumers (UI/tests) so downstream inference is not
        # attempted and the response is consistently handled.
        report["final"] = "inference_unavailable"
        report["megadescriptor"] = "NOT RUN"
        return report

    report["megadescriptor"] = "WOULD RUN (diagnose does not invoke MegaDescriptor)"
    report["final"] = "tiger_detected"
    return report
