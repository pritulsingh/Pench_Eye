"""
End-to-end tests for training, evaluation, calibration, embedding extraction and
the encoder ↔ checkpoint integration.

Everything runs on a tiny synthetic dataset with the `tiny` backbone, so no
pretrained weights are downloaded. Synthetic images cannot teach real tiger
identification — these tests verify the *pipeline executes and wires up
correctly*, not that any model is accurate.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for Re-ID training tests")

from ml.reid.checkpoint import load_checkpoint, resolve_checkpoint_path
from ml.reid.tiger_reid_encoder import ReIDModelUnavailable, TigerReIDEncoder


def make_synthetic_dataset(root, identities=3, sequences=3, frames=2, size=(64, 64)):
    """
    Distinct per-identity colour/pattern bias so the pipeline has *some* signal.
    This is not tiger data and proves nothing about real accuracy.
    """
    from PIL import Image

    for identity_index in range(identities):
        identity = f"TIGER_{identity_index + 1:03d}"
        for sequence in range(sequences):
            for frame in range(frames):
                rng = np.random.default_rng(identity_index * 1000 + sequence * 10 + frame)
                base = np.zeros((*size, 3), dtype=np.uint8)
                base[..., identity_index % 3] = 180
                stripe_period = 4 + identity_index * 3
                base[:, ::stripe_period] = 40
                noise = rng.integers(0, 40, (*size, 3), dtype=np.uint8)
                image = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                path = root / identity / f"CAM01_SEQ{sequence}_{frame:04d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(path)
    return root


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """Run a short real training loop once and reuse the artefacts."""
    from ml.reid.train import build_parser, train

    base = tmp_path_factory.mktemp("reid_run")
    data_root = make_synthetic_dataset(base / "data")
    output_dir = base / "weights"

    args = build_parser().parse_args(
        [
            "--data", str(data_root),
            "--output", str(output_dir),
            "--backbone", "tiny",
            "--no-pretrained",
            "--embedding-dim", "512",
            "--epochs", "2",
            "--batch-size", "4",
            "--num-instances", "2",
            "--image-size", "64", "64",
            "--no-clahe",
            "--val-fraction", "0.34",
            "--test-fraction", "0.33",
            "--warmup-epochs", "0",
            "--num-workers", "0",
            "--device", "cpu",
            "--log-level", "WARNING",
        ]
    )
    result = train(args)
    return {"data": data_root, "output": output_dir, "result": result}


# ── Training ──────────────────────────────────────────────────────────────
def test_training_produces_the_expected_artefacts(trained_run):
    output = trained_run["output"]
    for name in ("best.pt", "latest.pt", "config.json", "class_mapping.json", "training_history.json"):
        assert (output / name).is_file(), f"missing {name}"


def test_checkpoint_carries_reconstruction_metadata(trained_run):
    payload = load_checkpoint(trained_run["output"] / "best.pt")
    assert payload.model_config.embedding_dim == 512
    assert payload.model_config.backbone == "tiny"
    assert payload.preprocess_config.image_size == (64, 64)
    assert len(payload.identity_to_index) == 3
    assert payload.epoch >= 1
    assert payload.model_version.startswith("tiger-reid-tiny-512d")


def test_class_mapping_is_written_and_bijective(trained_run):
    mapping = json.loads((trained_run["output"] / "class_mapping.json").read_text())
    forward = mapping["identity_to_index"]
    reverse = mapping["index_to_identity"]
    assert len(forward) == len(reverse) == 3
    for identity, index in forward.items():
        assert reverse[str(index)] == identity


def test_history_records_every_epoch(trained_run):
    history = json.loads((trained_run["output"] / "training_history.json").read_text())
    assert len(history["epochs"]) == 2
    for entry in history["epochs"]:
        assert entry["train_loss"] > 0
        assert "arcface" in entry["train_components"]


def test_one_batch_training_step_reduces_loss_on_a_fixed_batch():
    """A single batch overfits: loss must fall. Guards the optimisation wiring."""
    from ml.reid.losses import ReIDLoss
    from ml.reid.model import ArcFaceHead, ModelConfig, build_model

    torch.manual_seed(0)
    model = build_model(ModelConfig(backbone="tiny", embedding_dim=64, pretrained=False)).train()
    head = ArcFaceHead(64, num_classes=3, scale=16.0, margin=0.1)
    criterion = ReIDLoss(triplet_weight=1.0)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=3e-3)

    images = torch.randn(6, 3, 64, 64)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        embeddings, feature = model(images, return_logits_feature=True)
        out = criterion(head(feature, labels), embeddings, labels)
        out.total.backward()
        optimizer.step()
        losses.append(out.components["total"])

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


def test_training_can_resume_from_a_checkpoint(trained_run, tmp_path):
    from ml.reid.train import build_parser, train

    output = tmp_path / "resumed"
    args = build_parser().parse_args(
        [
            "--data", str(trained_run["data"]),
            "--output", str(output),
            "--backbone", "tiny",
            "--no-pretrained",
            "--epochs", "3",
            "--batch-size", "4",
            "--num-instances", "2",
            "--image-size", "64", "64",
            "--no-clahe",
            "--val-fraction", "0.34",
            "--test-fraction", "0.33",
            "--warmup-epochs", "0",
            "--device", "cpu",
            "--resume", str(trained_run["output"] / "latest.pt"),
            "--log-level", "WARNING",
        ]
    )
    result = train(args)
    # Resumed at epoch 2, so only the third epoch runs.
    assert result["epochs_completed"] == 1
    assert load_checkpoint(output / "latest.pt").epoch == 3


def test_training_rejects_single_identity_datasets(tmp_path):
    from ml.reid.train import build_parser, train

    root = make_synthetic_dataset(tmp_path / "one", identities=1)
    args = build_parser().parse_args(
        [
            "--data", str(root),
            "--output", str(tmp_path / "out"),
            "--backbone", "tiny",
            "--no-pretrained",
            "--epochs", "1",
            "--device", "cpu",
            "--log-level", "WARNING",
        ]
    )
    with pytest.raises(SystemExit, match="at least 2 identities"):
        train(args)


# ── Evaluation ────────────────────────────────────────────────────────────
def test_evaluation_reports_real_metrics(trained_run, tmp_path):
    from ml.reid.evaluate import build_parser, evaluate

    out = tmp_path / "metrics.json"
    args = build_parser().parse_args(
        [
            "--checkpoint", str(trained_run["output"] / "best.pt"),
            "--data", str(trained_run["data"]),
            "--split", "all",
            "--device", "cpu",
            "--output", str(out),
            "--roc",
            "--log-level", "WARNING",
        ]
    )
    result = evaluate(args)

    assert out.is_file()
    assert result["evaluable"] is True
    loo = result["leave_one_out"]
    for key in ("rank1", "rank5", "rank10", "mean_ap"):
        assert 0.0 <= loo[key] <= 1.0
    assert loo["num_queries"] > 0
    assert result["roc"]
    assert result["embedding_dim"] == 512


# ── Threshold calibration ─────────────────────────────────────────────────
def test_calibration_emits_ordered_machine_readable_thresholds(trained_run, tmp_path):
    from ml.reid.calibrate_thresholds import build_parser, calibrate

    out = tmp_path / "thresholds.json"
    args = build_parser().parse_args(
        [
            "--checkpoint", str(trained_run["output"] / "best.pt"),
            "--data", str(trained_run["data"]),
            "--split", "all",
            "--device", "cpu",
            "--output", str(out),
            "--log-level", "WARNING",
        ]
    )
    result = calibrate(args)

    assert result["calibrated"] is True
    auto = result["auto_match_threshold"]
    review = result["review_threshold"]
    new_individual = result["new_individual_threshold"]
    assert auto > review > new_individual
    assert 0.0 <= new_individual and auto <= 1.0

    saved = json.loads(out.read_text())
    assert set(["auto_match_threshold", "review_threshold", "new_individual_threshold"]).issubset(saved)
    assert saved["distributions"]["same_identity"]["count"] > 0


# ── Embedding extraction ──────────────────────────────────────────────────
def test_embedding_extraction_writes_versioned_rows(trained_run, tmp_path):
    from ml.reid.extract_embeddings import build_parser, extract

    out = tmp_path / "embeddings.jsonl"
    args = build_parser().parse_args(
        [
            "--checkpoint", str(trained_run["output"] / "best.pt"),
            "--input", str(trained_run["data"]),
            "--output", str(out),
            "--device", "cpu",
            "--include-quality",
            "--log-level", "WARNING",
        ]
    )
    result = extract(args)

    assert result["count"] == 18  # 3 identities × 3 sequences × 2 frames
    assert result["embedding_dim"] == 512

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    first = rows[0]
    assert len(first["embedding"]) == 512
    assert first["model_version"].startswith("tiger-reid-tiny")
    assert first["preprocessing_version"]
    assert first["identity"].startswith("TIGER_")
    assert np.isclose(np.linalg.norm(first["embedding"]), 1.0, atol=1e-5)


# ── Encoder integration ───────────────────────────────────────────────────
def test_production_encoder_loads_a_trained_checkpoint(trained_run):
    encoder = TigerReIDEncoder(
        ml_mode="production", model_path=str(trained_run["output"] / "best.pt")
    )
    assert encoder.is_available()

    result = encoder.encode(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    assert len(result.embedding) == 512
    assert result.is_demo is False
    assert result.model_version.startswith("tiger-reid-tiny")
    assert np.isclose(np.linalg.norm(result.embedding), 1.0, atol=1e-5)

    status = encoder.status()
    assert status["available"] is True
    assert status["known_identities"] == 3
    assert status["embedding_dim"] == 512


def test_production_encoder_reuses_training_preprocessing(trained_run):
    encoder = TigerReIDEncoder(
        ml_mode="production", model_path=str(trained_run["output"] / "best.pt")
    )
    encoder.is_available()
    # The checkpoint was trained at 64×64 without CLAHE; inference must match.
    assert encoder.preprocess_config.image_size == (64, 64)
    assert encoder.preprocess_config.use_clahe is False


def test_production_encoder_is_deterministic(trained_run):
    encoder = TigerReIDEncoder(
        ml_mode="production", model_path=str(trained_run["output"] / "best.pt")
    )
    image = np.random.default_rng(3).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    first = encoder.encode(image).embedding
    second = encoder.encode(image).embedding
    assert np.allclose(first, second, atol=1e-6)


def test_production_encoder_fails_loudly_without_a_checkpoint(tmp_path):
    encoder = TigerReIDEncoder(ml_mode="production", model_path=str(tmp_path / "absent.pt"))
    assert encoder.is_available() is False
    with pytest.raises(ReIDModelUnavailable, match="No Re-ID checkpoint found"):
        encoder.encode(None)

    status = encoder.status()
    assert status["available"] is False
    assert "error" in status
    assert status["is_demo"] is False  # never claims to be a demo result


def test_encoder_rejects_dimension_mismatch(trained_run):
    """A 512-d application must not silently accept a different-width checkpoint."""
    encoder = TigerReIDEncoder(
        ml_mode="production",
        model_path=str(trained_run["output"] / "best.pt"),
        embedding_dim=128,
    )
    assert encoder.is_available() is False
    assert "does not match" in encoder.status()["error"]


def test_demo_encoder_still_works_and_is_marked_simulated():
    encoder = TigerReIDEncoder(ml_mode="demo")
    image = np.random.default_rng(5).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    result = encoder.encode(image)

    assert len(result.embedding) == 512
    assert result.is_demo is True
    assert result.model_version.startswith("demo-")
    assert result.metadata["simulated"] is True
    assert np.isclose(np.linalg.norm(result.embedding), 1.0, atol=1e-5)
    # Deterministic: same input, same embedding.
    assert np.allclose(result.embedding, encoder.encode(image).embedding, atol=1e-9)


def test_checkpoint_resolution_prefers_best_over_latest(trained_run):
    resolved = resolve_checkpoint_path(str(trained_run["output"]))
    assert resolved is not None and resolved.name == "best.pt"
