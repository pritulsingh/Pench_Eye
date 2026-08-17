"""
Tests for the Re-ID model, losses, metrics, checkpointing and the encoder
integration. Uses the `tiny` backbone throughout so nothing is downloaded.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for Re-ID model tests")

from ml.reid.checkpoint import load_checkpoint, load_model_for_inference, save_checkpoint
from ml.reid.losses import BatchHardTripletLoss, ReIDLoss
from ml.reid.metrics import (
    compute_roc,
    compute_similarity_stats,
    evaluate_query_gallery,
    evaluate_reid,
    split_query_gallery,
)
from ml.reid.model import ArcFaceHead, ModelConfig, build_model
from ml.reid.preprocessing import PreprocessConfig, preprocess_rgb

TINY = ModelConfig(backbone="tiny", embedding_dim=512, pretrained=False)


# ── Model ─────────────────────────────────────────────────────────────────
def test_model_outputs_512d_embeddings():
    model = build_model(TINY).eval()
    out = model(torch.randn(4, 3, 224, 224))
    assert out.shape == (4, 512)


def test_embeddings_are_l2_normalized():
    model = build_model(TINY).eval()
    out = model(torch.randn(6, 3, 224, 224))
    norms = out.norm(dim=1).detach().numpy()
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_model_returns_bnneck_feature_for_arcface():
    model = build_model(TINY).train()
    embedding, feature = model(torch.randn(4, 3, 224, 224), return_logits_feature=True)
    assert embedding.shape == feature.shape == (4, 512)
    # BNNeck output is not normalised — that is the point of the neck.
    assert not np.allclose(feature.norm(dim=1).detach().numpy(), 1.0, atol=1e-3)


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError, match="Unsupported backbone"):
        build_model(ModelConfig(backbone="not-a-model", pretrained=False))


def test_arcface_head_shape_and_margin_effect():
    head = ArcFaceHead(512, num_classes=5)
    features = torch.randn(4, 512)
    labels = torch.tensor([0, 1, 2, 3])
    logits = head(features, labels)
    assert logits.shape == (4, 5)
    # The target-class logit is penalised by the angular margin.
    plain = torch.nn.functional.linear(
        torch.nn.functional.normalize(features), torch.nn.functional.normalize(head.weight)
    ) * head.scale
    for i, label in enumerate(labels):
        assert logits[i, label] <= plain[i, label] + 1e-4


# ── Losses ────────────────────────────────────────────────────────────────
def test_triplet_loss_is_zero_when_clusters_are_separated():
    loss_fn = BatchHardTripletLoss(margin=0.3)
    a = torch.tensor([[1.0, 0.0], [0.99, 0.01]])
    b = torch.tensor([[0.0, 1.0], [0.01, 0.99]])
    embeddings = torch.cat([a, b])
    labels = torch.tensor([0, 0, 1, 1])
    assert float(loss_fn(embeddings, labels)) == pytest.approx(0.0, abs=1e-4)


def test_triplet_loss_positive_when_identities_overlap():
    loss_fn = BatchHardTripletLoss(margin=0.5)
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.99, 0.01], [0.01, 0.99]])
    labels = torch.tensor([0, 0, 1, 1])
    assert float(loss_fn(embeddings, labels)) > 0.0


def test_triplet_term_absent_when_weight_is_zero():
    criterion = ReIDLoss(triplet_weight=0.0)
    out = criterion(torch.randn(4, 3), torch.randn(4, 8), torch.tensor([0, 1, 2, 0]))
    assert "triplet" not in out.components
    assert out.components["total"] == pytest.approx(out.components["arcface"])


def test_combined_loss_includes_weighted_triplet():
    criterion = ReIDLoss(triplet_weight=2.0, triplet_margin=1.0)
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.99, 0.01], [0.01, 0.99]])
    labels = torch.tensor([0, 0, 1, 1])
    out = criterion(torch.randn(4, 2), embeddings, labels)
    assert out.components["total"] > out.components["arcface"]


# ── Metrics ───────────────────────────────────────────────────────────────
def clustered_embeddings(num_identities=4, per_identity=3, dim=8, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(num_identities, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    embeddings, labels = [], []
    for identity in range(num_identities):
        for _ in range(per_identity):
            vector = centres[identity] + rng.normal(scale=noise, size=dim)
            embeddings.append(vector / np.linalg.norm(vector))
            labels.append(f"TIGER_{identity:03d}")
    return np.array(embeddings), labels


def test_rank1_is_perfect_for_tight_clusters():
    embeddings, labels = clustered_embeddings(noise=0.01)
    metrics = evaluate_reid(embeddings, labels)
    assert metrics.rank1 == pytest.approx(1.0)
    assert metrics.rank5 == pytest.approx(1.0)
    assert metrics.mean_ap == pytest.approx(1.0)
    assert metrics.num_queries == len(labels)


def test_rank5_is_at_least_rank1():
    embeddings, labels = clustered_embeddings(num_identities=8, per_identity=4, noise=0.6, seed=3)
    metrics = evaluate_reid(embeddings, labels)
    assert metrics.rank5 >= metrics.rank1
    assert metrics.rank10 >= metrics.rank5


def test_random_embeddings_do_not_score_perfectly():
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(40, 16))
    labels = [f"T{i % 8}" for i in range(40)]
    metrics = evaluate_reid(embeddings, labels)
    assert metrics.rank1 < 0.9  # chance-level, not perfect


def test_map_penalises_relevant_items_ranked_late():
    # Two identities; one has a distant outlier that must rank late.
    embeddings = np.array(
        [[1.0, 0.0], [0.98, 0.19], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float64
    )
    labels = ["A", "A", "A", "B"]
    metrics = evaluate_reid(embeddings, labels)
    assert 0.0 < metrics.mean_ap < 1.0


def test_metrics_report_zero_queries_when_no_gallery_mate():
    embeddings, labels = clustered_embeddings(num_identities=3, per_identity=1)
    metrics = evaluate_reid(embeddings, labels)
    assert metrics.num_queries == 0
    assert metrics.rank1 == 0.0
    assert "not computable" in metrics.format()


def test_same_identity_similarity_exceeds_different_identity():
    embeddings, labels = clustered_embeddings(noise=0.05)
    stats = compute_similarity_stats(embeddings, labels)
    assert stats.same_identity_mean > stats.different_identity_mean
    assert stats.separation > 0
    assert stats.same_pairs > 0 and stats.different_pairs > 0


def test_query_gallery_split_excludes_query_from_gallery():
    embeddings, labels = clustered_embeddings(num_identities=4, per_identity=3)
    query_idx, gallery_idx = split_query_gallery(labels, seed=1)
    assert set(query_idx).isdisjoint(gallery_idx)
    assert len(query_idx) == 4  # one query per identity

    metrics = evaluate_query_gallery(
        embeddings[query_idx],
        [labels[i] for i in query_idx],
        embeddings[gallery_idx],
        [labels[i] for i in gallery_idx],
    )
    assert metrics.num_queries == 4
    assert metrics.rank1 == pytest.approx(1.0)


def test_sequence_exclusion_reduces_trivial_matches():
    embeddings, labels = clustered_embeddings(num_identities=3, per_identity=4, noise=0.01)
    # Every image of an identity in one sequence → no cross-sequence query exists.
    sequences = [f"{label}_SEQ0" for label in labels]
    strict = evaluate_reid(
        embeddings, labels, sequence_ids=sequences, allow_same_sequence_fallback=False
    )
    assert strict.num_queries == 0
    assert strict.protocol == "cross_sequence"

    # With the fallback the weaker protocol is used and clearly labelled.
    lenient = evaluate_reid(embeddings, labels, sequence_ids=sequences)
    assert lenient.num_queries > 0
    assert lenient.protocol == "self_excluded_only"
    assert "optimistic" in lenient.format()


def test_roc_is_monotonic_in_threshold():
    embeddings, labels = clustered_embeddings(noise=0.1)
    curve = compute_roc(embeddings, labels)
    assert curve
    tars = [p["tar"] for p in curve]
    fars = [p["far"] for p in curve]
    assert tars == sorted(tars, reverse=True)
    assert fars == sorted(fars, reverse=True)


# ── Checkpoints ───────────────────────────────────────────────────────────
def test_checkpoint_roundtrip_preserves_config_and_weights(tmp_path):
    model = build_model(TINY).eval()
    preprocess = PreprocessConfig(image_size=(224, 224), use_clahe=False)
    mapping = {"TIGER_001": 0, "TIGER_002": 1}
    path = tmp_path / "best.pt"

    save_checkpoint(
        path,
        model=model,
        identity_to_index=mapping,
        preprocess_config=preprocess,
        epoch=3,
        metrics={"val_rank1": 0.5},
        train_config={"note": "unit test"},
    )
    assert path.is_file()

    payload = load_checkpoint(path)
    assert payload.model_config.embedding_dim == 512
    assert payload.model_config.backbone == "tiny"
    assert payload.identity_to_index == mapping
    assert payload.epoch == 3
    assert payload.preprocess_config.use_clahe is False
    assert payload.index_to_identity[1] == "TIGER_002"

    restored, _ = load_model_for_inference(path, device="cpu")
    batch = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(batch), restored(batch), atol=1e-6)


def test_loading_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


# ── Preprocessing parity ──────────────────────────────────────────────────
def test_preprocessing_is_deterministic_and_correct_shape():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (300, 200, 3), dtype=np.uint8)
    config = PreprocessConfig(image_size=(224, 224))
    first = preprocess_rgb(image, config)
    second = preprocess_rgb(image, config)
    assert first.shape == (3, 224, 224)
    assert np.array_equal(first, second)


def test_stripe_processor_matches_shared_preprocessing():
    from ml.reid.stripe_processor import StripeProcessor

    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 255, (180, 240, 3), dtype=np.uint8)
    config = PreprocessConfig(image_size=(224, 224), use_clahe=False)
    processed = StripeProcessor(config=config).process(bgr)

    assert processed.tensor.shape == (3, 224, 224)
    assert processed.preprocessing_version == config.version
    direct = preprocess_rgb(bgr[:, :, ::-1], config)
    assert np.allclose(processed.tensor, direct, atol=1e-6)
