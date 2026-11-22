"""H19: loop-unrolled SVD bi-projection Lancet tests.

Covers the depth-recurrent (looped) Transformer forward pass, loop-unrolled
activation hooks keyed by ``(layer, tick)``, per-tick contrastive directions,
the decaying-weight SVD bi-projected basis construction, the K-loop-aware SCR
layer profile, and the `run-h19` benchmark comparing SVD-compressed bases
against the full-rank unrolled concat with optional H18 quant snap.
"""

import copy
import json
import math
from dataclasses import replace

import pytest
import torch

from sml.model.transformer import ModelConfig, Transformer
from sml.surgery.pareto import (
    compute_loop_unrolled_scr_profile,
    run_h19_loop_unrolled_lancet_benchmark,
)
from sml.surgery.subspace import (
    build_loop_unrolled_biprojected_basis,
    collect_loop_unrolled_activations,
    extract_loop_unrolled_contrastive_directions,
)


def _looped_config(**overrides) -> ModelConfig:
    kwargs = dict(
        name="tiny-looped",
        vocab_size=64,
        d_model=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        max_seq_len=64,
        num_loop_repeats=3,
        loop_layer_range=(1, 2),
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _build_looped_model(**overrides) -> Transformer:
    torch.manual_seed(11)
    model = Transformer(_looped_config(**overrides))
    model.eval()
    return model


def _ids(seed: int, batch: int = 2, seq_len: int = 16, vocab: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq_len), generator=generator)


def _unit_basis(i: int, d: int = 32) -> torch.Tensor:
    vec = torch.zeros(d, dtype=torch.float32)
    vec[i] = 1.0
    return vec


# ---------------------------------------------------------------------------
# 1. Looped (depth-recurrent) Transformer forward pass & single-pass compat
# ---------------------------------------------------------------------------


def test_looped_transformer_forward_and_compat():
    torch.manual_seed(12)
    model_k1 = _build_looped_model(num_loop_repeats=1)
    ids = _ids(1)

    # k=1 must match the standard single pass: manually running each layer once.
    with torch.no_grad():
        out_k1 = model_k1(ids).logits
        h = model_k1.embed_tokens(ids)
        for block in model_k1.layers:
            h = block(h, rotary_emb=model_k1.rotary_emb)
        reference = model_k1.lm_head(model_k1.norm(h))
    assert out_k1.shape == (2, 16, 64)
    assert torch.allclose(out_k1, reference, atol=1e-6, rtol=0.0)

    # Same weights, k=3 executions of layers (1, 2) between prefix and suffix.
    model_k3 = copy.deepcopy(model_k1)
    model_k3.config = replace(model_k3.config, num_loop_repeats=3, loop_layer_range=(1, 2))
    with torch.no_grad():
        out_k3 = model_k3(ids).logits
    assert out_k3.shape == out_k1.shape
    assert torch.isfinite(out_k3).all()
    assert not torch.allclose(out_k3, out_k1)
    assert not torch.allclose(out_k3, reference)

    # Invalid loop configurations must be rejected.
    with pytest.raises(ValueError):
        _looped_config(num_loop_repeats=0)
    with pytest.raises(ValueError):
        _looped_config(loop_layer_range=(2, 1))
    with pytest.raises(ValueError):
        _looped_config(loop_layer_range=(0, 4))  # end must satisfy end < n_layers


# ---------------------------------------------------------------------------
# 2. Loop-unrolled activation hooks & per-(layer, tick) directions
# ---------------------------------------------------------------------------


def test_collect_loop_unrolled_activations_and_directions():
    model = _build_looped_model()
    layer_indices = [0, 1, 2, 3]
    # Layers 1-2 are looped K=3 times; layers 0 and 3 run once (t=0 only).
    expected_keys = {
        (0, 0),
        (3, 0),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    }

    acts = collect_loop_unrolled_activations(model, _ids(2), layer_indices=layer_indices)
    assert set(acts.keys()) == expected_keys
    for key, act in acts.items():
        assert key[1] in ({0, 1, 2} if key[0] in (1, 2) else {0})
        assert act.shape == (2, 16, 32)
        assert act.dtype == torch.float32
        assert torch.isfinite(act).all()

    pos_ids = _ids(3)
    neg_ids = _ids(4)
    directions = extract_loop_unrolled_contrastive_directions(
        model, pos_ids, neg_ids, layer_indices=layer_indices
    )
    assert set(directions.keys()) == expected_keys
    for key, direction in directions.items():
        assert key[1] in ({0, 1, 2} if key[0] in (1, 2) else {0})
        assert direction.shape == (32,)
        assert direction.dtype == torch.float32
        assert abs(float(direction.norm(p=2)) - 1.0) <= 1e-5


# ---------------------------------------------------------------------------
# 3. Decaying-weight SVD bi-projected basis: orthogonality & rank bound
# ---------------------------------------------------------------------------


def test_build_loop_unrolled_biprojected_basis_orthogonality_and_rank():
    d_model = 32
    anchors = [[_unit_basis(0, d_model), _unit_basis(1, d_model)] for _ in range(3)]
    targets = [
        [_unit_basis(10, d_model), _unit_basis(16, d_model)],
        [_unit_basis(11, d_model), _unit_basis(17, d_model)],
        [_unit_basis(12, d_model), _unit_basis(18, d_model)],
    ]

    for k_loop in (1, 2):
        u_anchor, u_target = build_loop_unrolled_biprojected_basis(
            anchors, targets, k_loop=k_loop
        )
        assert u_anchor.dtype == torch.float32
        assert u_target.dtype == torch.float32
        # Anchor basis spans {e0, e1}; the compressed target basis honors k_loop.
        assert u_anchor.shape == (2, d_model)
        assert u_target.shape[1] == d_model
        assert u_target.shape[0] <= k_loop

        # Targets avoid the anchor subspace and stay orthonormal.
        assert float((u_target @ u_anchor.T).abs().max()) < 1e-5
        eye_t = torch.eye(u_target.shape[0], dtype=torch.float32)
        assert float((u_target @ u_target.T - eye_t).abs().max()) < 1e-5
        eye_a = torch.eye(u_anchor.shape[0], dtype=torch.float32)
        assert float((u_anchor @ u_anchor.T - eye_a).abs().max()) < 1e-5

    # omega_t = K - t + 1 weighting prioritizes earlier loop executions:
    # with K=2 ticks, tick 0 target = e1 and tick 1 target = e0, k_loop=1
    # must keep e1 (weight 2) over e0 (weight 1).
    anchors_2 = [[_unit_basis(2, d_model)] for _ in range(2)]
    targets_2 = [[_unit_basis(1, d_model)], [_unit_basis(0, d_model)]]
    u_anchor_2, u_target_2 = build_loop_unrolled_biprojected_basis(
        anchors_2, targets_2, k_loop=1
    )
    assert u_target_2.shape == (1, d_model)
    assert abs(float(u_target_2[0] @ _unit_basis(1, d_model))) > 0.99
    assert abs(float(u_target_2[0] @ _unit_basis(2, d_model))) < 1e-5


# ---------------------------------------------------------------------------
# 4. K-loop-aware SCR profile with exact weight restoration
# ---------------------------------------------------------------------------


def test_compute_loop_unrolled_scr_profile_restores_weights():
    model = _build_looped_model()
    d_model = model.config.d_model

    torch.manual_seed(14)
    target_dirs = {
        1: [torch.randn(d_model, dtype=torch.float32) for _ in range(2)],
        2: [torch.randn(d_model, dtype=torch.float32)],
    }

    ids = {
        "refusal_pos": _ids(5),
        "refusal_neg": _ids(6),
        "slop_pos": _ids(7),
        "slop_neg": _ids(8),
        "anchor": _ids(9),
        "cot": _ids(10),
    }
    watched = {}
    for layer_idx in target_dirs:
        watched[layer_idx] = {
            "o_proj": model.layers[layer_idx].self_attn.o_proj.weight.detach().clone(),
            "down_proj": model.layers[layer_idx].mlp.down_proj.weight.detach().clone(),
        }

    profile = compute_loop_unrolled_scr_profile(
        model,
        target_dirs_per_layer=target_dirs,
        refusal_pos_ids=ids["refusal_pos"],
        refusal_neg_ids=ids["refusal_neg"],
        slop_pos_ids=ids["slop_pos"],
        slop_neg_ids=ids["slop_neg"],
        anchor_ids=ids["anchor"],
        cot_ids=ids["cot"],
        lambda_cot=0.5,
    )

    assert set(profile.keys()) == set(target_dirs.keys())
    for layer_idx, row in profile.items():
        assert set(row.keys()) == {
            "scr",
            "signal",
            "anchor_kl",
            "cot_kl",
            "collateral_kl",
            "rank",
        }
        assert all(math.isfinite(float(value)) for value in row.values())
        assert row["rank"] == float(len(target_dirs[layer_idx]))
        assert row["signal"] >= 0.0
        assert row["anchor_kl"] >= 0.0
        assert row["cot_kl"] >= 0.0
        assert row["collateral_kl"] >= 0.0
        assert row["scr"] >= 0.0
        # scr = (refusal KL + slop KL) / (anchor KL + lambda_cot * cot KL + eps)
        expected_scr = row["signal"] / (row["collateral_kl"] + 1e-6)
        assert abs(row["scr"] - expected_scr) <= 1e-9 * max(1.0, abs(row["scr"]))
        assert abs(row["collateral_kl"] - (row["anchor_kl"] + 0.5 * row["cot_kl"])) <= 1e-9

    # The profile must leave the model untouched.
    for layer_idx, weights in watched.items():
        assert torch.equal(
            model.layers[layer_idx].self_attn.o_proj.weight, weights["o_proj"]
        )
        assert torch.equal(
            model.layers[layer_idx].mlp.down_proj.weight, weights["down_proj"]
        )


# ---------------------------------------------------------------------------
# 5. H19 benchmark: SVD bi-projection vs full-rank unrolled concat (+ quant snap)
# ---------------------------------------------------------------------------


def test_h19_benchmark_and_quant_snap():
    res = run_h19_loop_unrolled_lancet_benchmark(
        num_loop_repeats=3,
        k_loop=2,
        top_k_layers=2,
        quant_snap=True,
    )
    assert isinstance(res, dict)

    # The CLI dumps this structure verbatim, so it must be JSON-serializable.
    json.dumps(res)

    h19 = res["methods"]["h19_loop_unrolled_svd_biprojection"]
    full_rank = res["methods"]["full_rank_unrolled_concat"]
    assert set(res["methods"].keys()) >= {
        "naive_single_pass",
        "full_rank_unrolled_concat",
        "h19_loop_unrolled_svd_biprojection",
    }

    # SVD compression keeps the per-layer rank at k_loop=2, while the
    # full-rank unrolled concat carries all 2*K = 6 tick targets.
    h19_ranks = h19["effective_rank_per_layer"]
    full_ranks = full_rank["effective_rank_per_layer"]
    assert h19_ranks and full_ranks
    assert all(value <= 2.0 for value in h19_ranks.values())
    assert min(full_ranks.values()) == 6.0

    # The compressed basis is strictly gentler on anchor/CoT collateral.
    assert h19["total_collateral_kl"] < full_rank["total_collateral_kl"]

    # Orthogonality budgets survive the K-looped construction.
    assert h19["anchor_orthogonality_error"] < 1e-5
    assert h19["target_orthonormality_error"] < 1e-5
