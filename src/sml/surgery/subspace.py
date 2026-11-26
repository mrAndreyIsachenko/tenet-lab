"""Subspace manipulation utilities for model surgery."""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import re
import shutil
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file

from sml.model.transformer import Transformer
from sml.surgery.cka import (
    collect_layer_activations,
    get_moe_layer_components,
    get_transformer_layers,
    is_router_weight_key,
)
from sml.surgery.probes import (
    COT_REASONING_ANCHOR_PAIRS,
    REASONING_ANCHOR_PAIRS,
    TOOL_SQL_ANCHOR_PAIRS,
    PhaseSegmenter,
    encode_contrast_pairs,
)
from sml.surgery.quantization import (
    quantize_nvfp4_e2m1,
    quantize_tensor_groupwise,
    regularize_microblock_crest_factor,
)


def _resolve_pool_mask(
    input_ids: torch.Tensor,
    pool_mode: Literal["all", "deliberation_only", "answer_only"] = "all",
    phase_segmenter: PhaseSegmenter | None = None,
    tokenizer: Any = None,
    explicit_mask: torch.Tensor | None = None,
    texts: list[str] | None = None,
) -> torch.Tensor | None:
    """Resolve a (B, T) pooling mask for one side of a contrastive pair.

    Precedence: an explicit mask wins; ``pool_mode="all"`` returns ``None``
    (pool everything); otherwise the phase segmenter computes the masks, using
    ``deliberation_mask`` for ``"deliberation_only"`` and ``answer_mask`` for
    ``"answer_only"``.
    """
    if explicit_mask is not None:
        return explicit_mask
    if pool_mode == "all":
        return None
    if phase_segmenter is None:
        phase_segmenter = PhaseSegmenter()
    delib_mask, answer_mask = phase_segmenter.compute_phase_masks(
        input_ids, tokenizer=tokenizer, texts=texts
    )
    return delib_mask if pool_mode == "deliberation_only" else answer_mask


def _masked_pooled_mean(activations: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool a flattened (B*T, D) activation tensor, optionally masked.

    ``mask`` is a (B, T) (or already flattened) boolean tensor aligned with the
    activation rows; it is flattened to (B*T,). When no row is selected the
    unmasked mean over ``dim=0`` is used as a fallback.
    """
    if mask is None:
        return activations.mean(dim=0)
    flat = mask.detach().reshape(-1).to(device=activations.device, dtype=torch.bool)
    if bool(flat.any()):
        return activations[flat].mean(dim=0)
    return activations.mean(dim=0)


def extract_contrastive_direction(
    model: Transformer,
    positive_ids: torch.Tensor,
    negative_ids: torch.Tensor,
    layer_idx: int = -1,
    *,
    pool_mode: Literal["all", "deliberation_only", "answer_only"] = "all",
    phase_segmenter: PhaseSegmenter | None = None,
    tokenizer: Any = None,
    pos_mask: torch.Tensor | None = None,
    neg_mask: torch.Tensor | None = None,
    texts: tuple[list[str], list[str]] | None = None,
) -> torch.Tensor:
    """
    Extract a contrastive direction between positive and negative examples.

    Args:
        model: Transformer model instance
        positive_ids: Positive input tensor of shape (B, T)
        negative_ids: Negative input tensor of shape (B, T)
        layer_idx: Index of layer to extract activations from
        pool_mode: Which phase to pool - ``"all"`` (every token),
            ``"deliberation_only"`` (inside deliberation tags) or
            ``"answer_only"`` (outside deliberation tags).
        phase_segmenter: Segmenter used to derive phase masks when
            ``pool_mode`` is not ``"all"``.
        tokenizer: Optional tokenizer passed through to phase-mask computation.
        pos_mask: Optional explicit (B, T) boolean mask for the positive pool;
            overrides ``pool_mode`` when provided.
        neg_mask: Optional explicit (B, T) boolean mask for the negative pool;
            overrides ``pool_mode`` when provided.
        texts: Optional ``(positive_texts, negative_texts)`` backing the id
            batches, used for the character-ratio phase-mask fallback.

    Returns:
        Unit-normalized contrastive direction vector of shape (D,)
    """
    pos_texts, neg_texts = (None, None) if texts is None else (texts[0], texts[1])

    # Collect activations for positive examples
    pos_activations = collect_layer_activations(model, positive_ids)
    pos_mask_resolved = _resolve_pool_mask(
        positive_ids,
        pool_mode,
        phase_segmenter,
        tokenizer,
        explicit_mask=pos_mask,
        texts=pos_texts,
    )
    pos_mean = _masked_pooled_mean(pos_activations[layer_idx], pos_mask_resolved)

    # Collect activations for negative examples
    neg_activations = collect_layer_activations(model, negative_ids)
    neg_mask_resolved = _resolve_pool_mask(
        negative_ids,
        pool_mode,
        phase_segmenter,
        tokenizer,
        explicit_mask=neg_mask,
        texts=neg_texts,
    )
    neg_mean = _masked_pooled_mean(neg_activations[layer_idx], neg_mask_resolved)

    # Compute difference and normalize
    diff = pos_mean - neg_mean
    direction = diff / (diff.norm(p=2) + 1e-8)

    return direction


def compute_probe_spectral_stats(diff_matrix: torch.Tensor) -> dict[str, float]:
    """
    Compute spectral statistics of a per-example difference matrix.

    Given ``Delta_H`` of shape (B, D) holding per-example activation
    differences, compute the singular values of the (float32) matrix and
    summarize how concentrated the differences are along a single direction:

    - ``explained_variance_ratio_1``: share of total squared singular-value
      energy captured by the top singular value.
    - ``singular_value_1``: the top singular value itself.
    - ``spectral_rank_effective``: participation-ratio style effective rank,
      ``(sum(s^2))^2 / sum(s^4)``; 1.0 for a rank-1 matrix, up to B for a
      flat spectrum.

    Args:
        diff_matrix: Difference matrix of shape (B, D).

    Returns:
        Dict with the three statistics above; zeros when the matrix is
        empty or has no energy.
    """
    if diff_matrix.numel() == 0:
        return {
            "explained_variance_ratio_1": 0.0,
            "singular_value_1": 0.0,
            "spectral_rank_effective": 0.0,
        }

    s = torch.linalg.svdvals(diff_matrix.float())
    if s.numel() == 0:
        return {
            "explained_variance_ratio_1": 0.0,
            "singular_value_1": 0.0,
            "spectral_rank_effective": 0.0,
        }

    return {
        "explained_variance_ratio_1": float((s[0] ** 2) / (torch.sum(s**2) + 1e-8)),
        "singular_value_1": float(s[0]),
        "spectral_rank_effective": float((torch.sum(s**2) ** 2) / (torch.sum(s**4) + 1e-8)),
    }


def extract_contrastive_direction_with_stats(
    model: Transformer,
    positive_ids: torch.Tensor,
    negative_ids: torch.Tensor,
    layer_idx: int = -1,
    *,
    pool_mode: Literal["all", "deliberation_only", "answer_only"] = "all",
    phase_segmenter: PhaseSegmenter | None = None,
    tokenizer: Any = None,
    pos_mask: torch.Tensor | None = None,
    neg_mask: torch.Tensor | None = None,
    texts: tuple[list[str], list[str]] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Extract a contrastive direction plus spectral stats of the differences.

    Like ``extract_contrastive_direction``, but instead of averaging
    activations across examples first, it keeps a per-example mean over
    sequence tokens so ``Delta_H = mean_pos - mean_neg`` has shape (B, D).
    The returned direction is the unit-normalized difference of the pooled
    means, matching ``extract_contrastive_direction`` (including any phase
    pooling); the statistics describe how concentrated the per-example
    differences are around that direction. When a phase mask is active, each
    example is pooled over its own masked tokens, falling back to that
    example's full token mean when its mask is empty.

    Args:
        model: Transformer model instance
        positive_ids: Positive input tensor of shape (B, T)
        negative_ids: Negative input tensor of shape (B, T)
        layer_idx: Index of layer to extract activations from
        pool_mode: Which phase to pool - ``"all"`` (every token),
            ``"deliberation_only"`` (inside deliberation tags) or
            ``"answer_only"`` (outside deliberation tags).
        phase_segmenter: Segmenter used to derive phase masks when
            ``pool_mode`` is not ``"all"``.
        tokenizer: Optional tokenizer passed through to phase-mask computation.
        pos_mask: Optional explicit (B, T) boolean mask for the positive pool;
            overrides ``pool_mode`` when provided.
        neg_mask: Optional explicit (B, T) boolean mask for the negative pool;
            overrides ``pool_mode`` when provided.
        texts: Optional ``(positive_texts, negative_texts)`` backing the id
            batches, used for the character-ratio phase-mask fallback.

    Returns:
        Tuple ``(direction, stats)`` where ``direction`` is a unit-norm
        vector of shape (D,) and ``stats`` comes from
        ``compute_probe_spectral_stats``.
    """
    pos_texts, neg_texts = (None, None) if texts is None else (texts[0], texts[1])
    batch_size = int(positive_ids.shape[0])
    seq_len = int(positive_ids.shape[1]) if positive_ids.dim() > 1 else 1

    def _per_example_means(activations: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        acts = activations[layer_idx].reshape(batch_size, seq_len, -1)
        if mask is None:
            return acts.mean(dim=1)  # (B, D)
        flat = mask.detach().to(device=acts.device, dtype=torch.bool).reshape(batch_size, seq_len)
        rows = []
        for b in range(batch_size):
            if bool(flat[b].any()):
                rows.append(acts[b][flat[b]].mean(dim=0))
            else:
                rows.append(acts[b].mean(dim=0))
        return torch.stack(rows, dim=0)

    # Collect activations for positive examples. `collect_layer_activations`
    # returns flattened (B*T, D) tensors, so restore the (B, T, D) layout from
    # the input's shape; the masked mean over sequence tokens gives (B, D).
    pos_activations = collect_layer_activations(model, positive_ids)
    pos_mask_resolved = _resolve_pool_mask(
        positive_ids,
        pool_mode,
        phase_segmenter,
        tokenizer,
        explicit_mask=pos_mask,
        texts=pos_texts,
    )
    pos_mean = _per_example_means(pos_activations, pos_mask_resolved)  # (B, D)

    # Collect activations for negative examples
    neg_activations = collect_layer_activations(model, negative_ids)
    neg_mask_resolved = _resolve_pool_mask(
        negative_ids,
        pool_mode,
        phase_segmenter,
        tokenizer,
        explicit_mask=neg_mask,
        texts=neg_texts,
    )
    neg_mean = _per_example_means(neg_activations, neg_mask_resolved)  # (B, D)

    # Per-example difference matrix
    delta_h = pos_mean - neg_mean
    stats = compute_probe_spectral_stats(delta_h)

    # Mean direction, normalized to unit L2 norm (matching
    # extract_contrastive_direction, including its phase pooling).
    pos_pooled = _masked_pooled_mean(pos_activations[layer_idx], pos_mask_resolved)
    neg_pooled = _masked_pooled_mean(neg_activations[layer_idx], neg_mask_resolved)
    mean_dir = pos_pooled - neg_pooled
    direction = mean_dir / (mean_dir.norm(p=2) + 1e-8)

    return direction, stats


def build_tri_anchor_basis(
    cot_dirs: list[torch.Tensor] | torch.Tensor,
    math_logic_dirs: list[torch.Tensor] | torch.Tensor,
    tool_sql_dirs: list[torch.Tensor] | torch.Tensor,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """
    Combine Tri-Anchor direction families into one orthonormal basis.

    Concatenates the CoT-reasoning, math/logic and tool/SQL direction sets
    (in that order) and orthonormalizes them with ``_orthonormalize_basis``
    (two-pass modified Gram-Schmidt in float32), so dependent or near-duplicate
    directions across families are dropped.

    Args:
        cot_dirs: Deliberation/CoT anchor directions.
        math_logic_dirs: Math/logic anchor directions.
        tool_sql_dirs: Tool/SQL anchor directions.
        device: Optional device to orthonormalize on.

    Returns:
        List of orthonormal float32 basis vectors.
    """
    combined = (
        _as_direction_list(cot_dirs)
        + _as_direction_list(math_logic_dirs)
        + _as_direction_list(tool_sql_dirs)
    )
    return _orthonormalize_basis(combined, device=device)


def extract_tri_anchor_directions(
    model: torch.nn.Module,
    *,
    cot_pairs: list[tuple[str, str]] | None = None,
    math_logic_pairs: list[tuple[str, str]] | None = None,
    tool_sql_pairs: list[tuple[str, str]] | None = None,
    layer_indices: list[int] | None = None,
    phase_segmenter: PhaseSegmenter | None = None,
    vocab_size: int = 32768,
    seq_len: int = 48,
    tokenizer: Any = None,
) -> dict[int, list[torch.Tensor]]:
    """
    Extract per-layer Tri-Anchor basis directions from a model.

    For every candidate layer, three contrastive directions are extracted:

    - ``a_cot``: from the CoT reasoning pairs, pooled over the deliberation
      phase only (``pool_mode="deliberation_only"``).
    - ``a_math``: from the math/logic anchor pairs (all tokens).
    - ``a_tool``: from the tool/SQL anchor pairs (all tokens).

    The three directions are combined per layer with
    ``build_tri_anchor_basis`` into an orthonormal Tri-Anchor basis.

    Args:
        model: Transformer model instance (HF or sml-nano).
        cot_pairs: CoT reasoning pairs; defaults to ``COT_REASONING_ANCHOR_PAIRS``.
        math_logic_pairs: Math/logic pairs; defaults to ``REASONING_ANCHOR_PAIRS``.
        tool_sql_pairs: Tool/SQL pairs; defaults to ``TOOL_SQL_ANCHOR_PAIRS``.
        layer_indices: Layers to extract (None means all layers).
        phase_segmenter: Segmenter used to isolate deliberation tokens for
            ``a_cot``; defaults to the ``"think"`` format.
        vocab_size: Vocabulary size for the fallback hash encoder.
        seq_len: Sequence length used when encoding probe texts.
        tokenizer: Optional HF-style tokenizer for encoding probe texts.

    Returns:
        Dict mapping layer index -> list of orthonormal Tri-Anchor directions.
    """
    cot_pairs = COT_REASONING_ANCHOR_PAIRS if cot_pairs is None else cot_pairs
    math_logic_pairs = REASONING_ANCHOR_PAIRS if math_logic_pairs is None else math_logic_pairs
    tool_sql_pairs = TOOL_SQL_ANCHOR_PAIRS if tool_sql_pairs is None else tool_sql_pairs
    if phase_segmenter is None:
        phase_segmenter = PhaseSegmenter()

    model_device = next(model.parameters()).device
    cot_pos_ids, cot_neg_ids = encode_contrast_pairs(
        cot_pairs, vocab_size=vocab_size, seq_len=seq_len, device=model_device, tokenizer=tokenizer
    )
    cot_texts = ([pos for pos, _ in cot_pairs], [neg for _, neg in cot_pairs])
    math_pos_ids, math_neg_ids = encode_contrast_pairs(
        math_logic_pairs,
        vocab_size=vocab_size,
        seq_len=seq_len,
        device=model_device,
        tokenizer=tokenizer,
    )
    tool_pos_ids, tool_neg_ids = encode_contrast_pairs(
        tool_sql_pairs,
        vocab_size=vocab_size,
        seq_len=seq_len,
        device=model_device,
        tokenizer=tokenizer,
    )

    if layer_indices is None:
        layer_indices = list(range(len(get_transformer_layers(model))))

    tri_basis: dict[int, list[torch.Tensor]] = {}
    for idx in layer_indices:
        a_cot = extract_contrastive_direction(
            model,
            cot_pos_ids,
            cot_neg_ids,
            layer_idx=idx,
            pool_mode="deliberation_only",
            phase_segmenter=phase_segmenter,
            tokenizer=tokenizer,
            texts=cot_texts,
        )
        a_math = extract_contrastive_direction(
            model, math_pos_ids, math_neg_ids, layer_idx=idx, pool_mode="all"
        )
        a_tool = extract_contrastive_direction(
            model, tool_pos_ids, tool_neg_ids, layer_idx=idx, pool_mode="all"
        )
        tri_basis[int(idx)] = build_tri_anchor_basis(
            [a_cot], [a_math], [a_tool], device=model_device
        )
    return tri_basis


def project_out_rank1_subspace(
    weight: torch.Tensor, direction: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """
    Project out a rank-1 subspace component from a weight matrix.

    Args:
        weight: Weight tensor of shape (out_features, in_features)
        direction: Direction vector of shape (out_features,)
        scale: Scaling factor for projection

    Returns:
        Modified weight tensor of same shape as input
    """
    # Normalize direction to unit norm
    r = direction.to(weight.device, dtype=torch.float32)
    r = r / (r.norm(p=2) + 1e-8)

    # Compute projection matrix
    proj = torch.outer(r, r @ weight.float())

    # Apply projection and return in original dtype and device
    return (weight.float() - scale * proj).to(device=weight.device, dtype=weight.dtype)


def _orthonormalize_basis(
    vectors: list[torch.Tensor] | torch.Tensor, device: torch.device | None = None
) -> list[torch.Tensor]:
    """
    Orthonormalize a set of vectors with modified Gram-Schmidt in float32.

    Args:
        vectors: List of vectors (any shape, flattened) or a tensor whose
            first dimension enumerates the vectors.
        device: Optional device to move vectors to before orthonormalization.

    Returns:
        List of orthonormal float32 vectors. Linearly dependent or
        near-zero vectors are dropped.
    """
    if isinstance(vectors, torch.Tensor):
        if vectors.ndim == 1:
            candidates: list[torch.Tensor] = [vectors]
        else:
            candidates = [vectors[i] for i in range(vectors.shape[0])]
    else:
        candidates = [torch.as_tensor(v) for v in vectors]

    basis: list[torch.Tensor] = []
    for vec in candidates:
        if device is not None:
            v = vec.detach().to(device=device, dtype=torch.float32)
        else:
            v = vec.detach().to(dtype=torch.float32)
        v = v.reshape(-1)

        # Modified Gram-Schmidt with re-orthogonalization ("twice is enough")
        # to prevent numerical drift when candidate vectors are nearly collinear.
        for _ in range(2):
            for b in basis:
                v = v - (v @ b) * b

        norm = float(v.norm(p=2))
        if norm > 1e-6:
            basis.append(v / norm)

    return basis


def biproject_direction(
    target_dir: torch.Tensor, anchor_dirs: list[torch.Tensor] | torch.Tensor
) -> torch.Tensor:
    """
    Remove all anchor components from a direction, then renormalize.

    The result is the residual of ``target_dir`` after projecting out the
    subspace spanned by ``anchor_dirs``, normalized to unit L2 norm.

    Args:
        target_dir: Direction vector to biproject.
        anchor_dirs: Anchor directions whose span should be removed.

    Returns:
        Unit-norm float32 residual tensor on ``target_dir.device``.
    """
    device = target_dir.device
    basis = _orthonormalize_basis(anchor_dirs, device=device)

    t = target_dir.detach().to(dtype=torch.float32).reshape(-1)
    for _ in range(2):
        for b in basis:
            t = t - (t @ b) * b

    norm = float(t.norm(p=2))
    if norm > 1e-8:
        t = t / norm
        for b in basis:
            t = t - (t @ b) * b
        norm2 = float(t.norm(p=2))
        if norm2 > 1e-8:
            t = t / norm2

    return t


def collect_loop_unrolled_activations(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    layer_indices: list[int] | None = None,
) -> dict[tuple[int, int], torch.Tensor]:
    """
    Collect activations from every execution of (possibly loop-unrolled) layers.

    When a model loop-unrolls a block, the same layer module runs several
    times per forward pass and a plain per-layer hook would only keep the
    last output. This registers one hook per requested layer and keys each
    captured hidden state by ``(layer_idx, tick)``, where ``tick`` counts
    how many times that layer has executed so far in the pass.

    Args:
        model: Transformer model instance (HF or sml-nano).
        input_ids: Input token ids of shape (B, T).
        layer_indices: Indices of layers to hook (None means all layers).

    Returns:
        Mapping of ``(layer_idx, tick)`` -> detached hidden tensor captured
        from that layer execution.
    """
    layers = get_transformer_layers(model)
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    activations: dict[tuple[int, int], torch.Tensor] = {}
    tick_counts = {layer_idx: 0 for layer_idx in layer_indices}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(layer_idx: int):
        def _hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            tick = tick_counts[layer_idx]
            activations[(layer_idx, tick)] = out.detach()
            tick_counts[layer_idx] += 1

        return _hook

    try:
        for layer_idx in layer_indices:
            hooks.append(layers[layer_idx].register_forward_hook(_make_hook(layer_idx)))
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in hooks:
            handle.remove()

    return activations


def extract_loop_unrolled_contrastive_directions(
    model: torch.nn.Module,
    positive_ids: torch.Tensor,
    negative_ids: torch.Tensor,
    *,
    layer_indices: list[int] | None = None,
    pool_mode: Literal["all", "deliberation_only", "answer_only"] = "all",
    phase_segmenter: PhaseSegmenter | None = None,
    tokenizer: Any = None,
    pos_mask: torch.Tensor | None = None,
    neg_mask: torch.Tensor | None = None,
    texts: tuple[list[str], list[str]] | None = None,
) -> dict[tuple[int, int], torch.Tensor]:
    """
    Extract one contrastive direction per loop-unrolled layer execution.

    Like :func:`extract_contrastive_direction`, but activations are keyed by
    ``(layer_idx, tick)`` (see :func:`collect_loop_unrolled_activations`) and
    a pooled ``pos_mean - neg_mean`` direction is produced for every key.

    Args:
        model: Transformer model instance (HF or sml-nano).
        positive_ids: Positive input tensor of shape (B, T).
        negative_ids: Negative input tensor of shape (B, T).
        layer_indices: Indices of layers to hook (None means all layers).
        pool_mode: Which phase to pool - ``"all"`` (every token),
            ``"deliberation_only"`` (inside deliberation tags) or
            ``"answer_only"`` (outside deliberation tags).
        phase_segmenter: Segmenter used to derive phase masks when
            ``pool_mode`` is not ``"all"``.
        tokenizer: Optional tokenizer passed through to phase-mask computation.
        pos_mask: Optional explicit (B, T) boolean mask for the positive pool;
            overrides ``pool_mode`` when provided.
        neg_mask: Optional explicit (B, T) boolean mask for the negative pool;
            overrides ``pool_mode`` when provided.
        texts: Optional ``(positive_texts, negative_texts)`` backing the id
            batches, used for the character-ratio phase-mask fallback.

    Returns:
        Mapping of ``(layer_idx, tick)`` -> unit-norm float32 direction of
        shape (D,). Near-zero differences map to zero vectors.
    """
    pos_texts, neg_texts = (None, None) if texts is None else (texts[0], texts[1])

    pos_acts = collect_loop_unrolled_activations(model, positive_ids, layer_indices=layer_indices)
    neg_acts = collect_loop_unrolled_activations(model, negative_ids, layer_indices=layer_indices)

    p_mask = _resolve_pool_mask(
        positive_ids,
        pool_mode,
        phase_segmenter,
        tokenizer,
        explicit_mask=pos_mask,
        texts=pos_texts,
    )
    n_mask = _resolve_pool_mask(
        negative_ids,
        pool_mode,
        phase_segmenter,
        tokenizer,
        explicit_mask=neg_mask,
        texts=neg_texts,
    )

    directions: dict[tuple[int, int], torch.Tensor] = {}
    for key, pos_flat in pos_acts.items():
        neg_flat = neg_acts.get(key)
        if neg_flat is None:
            continue
        pos_mean = _masked_pooled_mean(
            pos_flat.reshape(-1, pos_flat.shape[-1]).to(dtype=torch.float32), p_mask
        )
        neg_mean = _masked_pooled_mean(
            neg_flat.reshape(-1, neg_flat.shape[-1]).to(dtype=torch.float32), n_mask
        )
        diff = pos_mean - neg_mean
        norm = float(diff.norm(p=2))
        directions[key] = diff / norm if norm > 1e-12 else torch.zeros_like(diff)

    return directions


def build_loop_unrolled_biprojected_basis(
    anchor_dirs_by_tick: list[list[torch.Tensor] | torch.Tensor],
    target_dirs_by_tick: list[list[torch.Tensor] | torch.Tensor],
    *,
    k_loop: int = 2,
    anchor_sv_tol: float = 1e-6,
    target_sv_tol: float = 1e-6,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build anchor and target bases from per-tick loop-unrolled directions.

    Anchor directions are normalized, weighted by a decaying per-tick weight
    (``omega_t = K - t`` for ``K`` ticks, so earlier executions matter more),
    stacked and SVD-reduced to an orthonormal anchor basis. Target directions
    are normalized, weighted the same way, bi-projected orthogonally to the
    anchor basis, and the residuals are SVD-reduced to at most ``k_loop``
    orthonormal target directions that avoid the anchor subspace.

    Args:
        anchor_dirs_by_tick: One list (or tensor) of anchor directions per
            tick, ordered by execution tick.
        target_dirs_by_tick: One list (or tensor) of target directions per
            tick; must have the same length as ``anchor_dirs_by_tick``.
        k_loop: Maximum number of target basis directions to keep.
        anchor_sv_tol: Relative singular-value threshold for the anchor SVD.
        target_sv_tol: Relative singular-value threshold for the residual SVD.
        device: Optional device to move directions to before computation.

    Returns:
        ``(u_anchor, u_target)``: float32 orthonormal bases of shapes
        ``(r_anchor, D)`` and ``(k_eff, D)`` with ``k_eff <= k_loop``.
    """
    if len(anchor_dirs_by_tick) != len(target_dirs_by_tick):
        raise ValueError(
            "anchor_dirs_by_tick and target_dirs_by_tick must have the same "
            f"length ({len(anchor_dirs_by_tick)} != {len(target_dirs_by_tick)})"
        )
    if len(anchor_dirs_by_tick) < 1:
        raise ValueError("anchor_dirs_by_tick must contain at least one tick")
    if k_loop < 1:
        raise ValueError("k_loop must be >= 1")

    num_ticks = len(anchor_dirs_by_tick)

    def _normalize(row: torch.Tensor) -> torch.Tensor | None:
        v = torch.as_tensor(row).detach().reshape(-1)
        if device is not None:
            v = v.to(device=device, dtype=torch.float32)
        else:
            v = v.to(dtype=torch.float32)
        norm = float(v.norm(p=2))
        if norm <= 1e-12:
            return None
        return v / norm

    # Weighted anchor matrix: earlier ticks (loop-unrolled executions closer
    # to the input) receive larger weight.
    anchor_rows: list[torch.Tensor] = []
    for tick in range(num_ticks):
        sqrt_w = math.sqrt(float(num_ticks - tick))
        for row in anchor_dirs_by_tick[tick]:
            v = _normalize(row)
            if v is not None:
                anchor_rows.append(v * sqrt_w)

    if not anchor_rows:
        raise ValueError("no valid (non-zero) anchor directions found")

    A = torch.stack(anchor_rows)
    _, s_a, vh_a = torch.linalg.svd(A, full_matrices=False)
    keep_mask = s_a > anchor_sv_tol * s_a[0]
    anchor_basis = _orthonormalize_basis(vh_a[keep_mask], device=device)
    if anchor_basis:
        u_anchor = torch.stack(anchor_basis)
    else:
        u_anchor = torch.zeros(0, A.shape[1], dtype=torch.float32, device=A.device)
    anchor_rows_basis = [u_anchor[i] for i in range(u_anchor.shape[0])]

    # Residuals of weighted, normalized target directions after removing the
    # anchor subspace via two-pass modified Gram-Schmidt.
    residuals: list[torch.Tensor] = []
    for tick in range(num_ticks):
        sqrt_w = math.sqrt(float(num_ticks - tick))
        for row in target_dirs_by_tick[tick]:
            v = _normalize(row)
            if v is None:
                continue
            for _ in range(2):
                for b in anchor_rows_basis:
                    v = v - (v @ b) * b
            norm = float(v.norm(p=2))
            if norm > 1e-6:
                residuals.append((v / norm) * sqrt_w)

    if not residuals:
        d_model = u_anchor.shape[1]
        return u_anchor, torch.zeros(0, d_model, dtype=torch.float32, device=u_anchor.device)

    R_perp = torch.stack(residuals)
    _, s_t, vh_t = torch.linalg.svd(R_perp, full_matrices=False)
    n_keep = min(k_loop, int((s_t > target_sv_tol * s_t[0]).sum()))

    # Re-orthogonalize the kept residual basis rows against the anchor basis
    # and among themselves (two-pass MGS, mirroring biproject_direction).
    u_target_rows: list[torch.Tensor] = []
    for i in range(min(n_keep, int(vh_t.shape[0]))):
        v = vh_t[i].detach()
        if device is not None:
            v = v.to(device=device, dtype=torch.float32)
        else:
            v = v.to(dtype=torch.float32)
        v = v.reshape(-1)
        for _ in range(2):
            for b in anchor_rows_basis:
                v = v - (v @ b) * b
            for b in u_target_rows:
                v = v - (v @ b) * b
        norm = float(v.norm(p=2))
        if norm > 1e-6:
            u_target_rows.append(v / norm)

    if u_target_rows:
        u_target = torch.stack(u_target_rows)
    else:
        u_target = torch.zeros(0, u_anchor.shape[1], dtype=torch.float32, device=u_anchor.device)

    return u_anchor, u_target


def project_out_subspace_norm_preserving(
    weight: torch.Tensor,
    directions: list[torch.Tensor] | torch.Tensor,
    scale: float = 1.0,
    preserve_row_norm: bool = True,
) -> torch.Tensor:
    """
    Project a multi-dimensional subspace out of a weight matrix, keeping row norms.

    Args:
        weight: Weight tensor of shape (out_features, in_features).
        directions: Directions spanning the subspace to remove.
        scale: Scaling factor for the projection.
        preserve_row_norm: If True, rescale each row of the projected weight
            so its L2 norm matches the corresponding original row.

    Returns:
        Modified weight tensor in the original device and dtype.
    """
    orig_device = weight.device
    orig_dtype = weight.dtype

    w_f = weight.detach().to(dtype=torch.float32)
    basis = _orthonormalize_basis(directions, device=orig_device)

    w_proj = w_f.clone()
    for r in basis:
        w_proj = w_proj - scale * torch.outer(r, r @ w_f)

    if preserve_row_norm:
        orig_row_norms = w_f.norm(p=2, dim=-1)
        new_row_norms = w_proj.norm(p=2, dim=-1)
        w_proj = w_proj * (orig_row_norms / (new_row_norms + 1e-8)).unsqueeze(-1)

    return w_proj.to(device=orig_device, dtype=orig_dtype)


def project_and_snap_subspace_norm_preserving(
    weight: torch.Tensor,
    directions: list[torch.Tensor] | torch.Tensor,
    *,
    scale: float = 1.0,
    snap_mode: str = "hybrid_nvfp4_int4",
    num_iters: int = 4,
    step_size: float = 0.75,
    block_size: int = 16,
    max_block_cf: float | None = 3.2,
    preserve_bf16_nullspace: bool = False,
    preserve_row_norm: bool = True,
    max_rel_drift: float = 0.02,
) -> torch.Tensor:
    """
    Lancet projection followed by quantization-aware subspace snapping (H18).

    The pipeline is:

    0. Optionally apply micro-block crest-factor regularization to the raw
       weight (hygiene pass, controlled by ``max_block_cf``).
    1. Compute the norm-preserving Lancet projection ``w_lancet`` of that
       hygiene weight.
    2. Iteratively reduce the quantization error that leaks back into the
       target subspace: simulate the destination quantization of the current
       weight, subtract the leaked component, clamp drift against
       ``w_lancet``, and keep the candidate with the lowest subspace leak.

    Args:
        weight: Weight tensor of shape (out_features, in_features).
        directions: Directions spanning the subspace to remove.
        scale: Scaling factor for the Lancet projection.
        snap_mode: Destination quantization simulator; one of
            ``"nvfp4_b16"``, ``"int4_b32"`` or ``"hybrid_nvfp4_int4"``
            (the average of both simulators).
        num_iters: Number of leak-reduction iterations.
        step_size: Step size for the leak-reduction update.
        block_size: Micro-block/group size for the quantization simulators
            and the crest-factor regularization.
        max_block_cf: Micro-block crest-factor cap applied to the raw weight
            before projection; ``None`` or non-positive disables the pass.
        preserve_bf16_nullspace: If True, re-project each candidate onto the
            complement of the target subspace after the update.
        preserve_row_norm: If True, rescale rows of each candidate to match
            the ``w_lancet`` row L2 norms.
        max_rel_drift: Maximum relative Frobenius drift of a candidate
            relative to ``w_lancet``; candidates beyond the budget (plus a
            small tolerance) are never accepted.

    Returns:
        The best (lowest-leak) weight tensor in the original device and dtype.

    Raises:
        ValueError: If ``snap_mode`` is not a supported mode.
    """
    allowed_modes = ("nvfp4_b16", "int4_b32", "hybrid_nvfp4_int4")
    if snap_mode not in allowed_modes:
        raise ValueError(f"Unsupported snap_mode: {snap_mode!r}; expected one of {allowed_modes}")

    orig_device = weight.device
    orig_dtype = weight.dtype

    # Run the snap loop on the GPU when one is available: the 4-iteration
    # NVFP4/INT4 quantization simulation dominates runtime on CPU.
    work_device = orig_device
    if orig_device.type == "cpu" and torch.cuda.is_available():
        weight = weight.to(device="cuda")
        work_device = weight.device

    # Step 0: optional micro-block crest-factor hygiene on the raw weight.
    w_hygiene = weight
    if max_block_cf is not None and max_block_cf > 0:
        w_hygiene = regularize_microblock_crest_factor(
            weight, block_size=block_size, max_block_cf=max_block_cf
        )

    # Step 1: norm-preserving Lancet projection, computed in float32.
    w_lancet = project_out_subspace_norm_preserving(
        w_hygiene, directions, scale=scale, preserve_row_norm=preserve_row_norm
    ).to(torch.float32)

    basis = _orthonormalize_basis(directions, device=work_device)
    if not basis:
        return w_lancet.to(device=orig_device, dtype=orig_dtype)

    U_mat = torch.stack(basis, dim=0)  # (r, d_out)

    def _simulate_q4(w_f32: torch.Tensor) -> torch.Tensor:
        """Simulate the destination quantization of ``w_f32``."""
        if snap_mode == "nvfp4_b16":
            return quantize_nvfp4_e2m1(w_f32, block_size=block_size)
        if snap_mode == "int4_b32":
            return quantize_tensor_groupwise(w_f32, bits=4, group_size=32)
        nv = quantize_nvfp4_e2m1(w_f32, block_size=block_size)
        i4 = quantize_tensor_groupwise(w_f32, bits=4, group_size=32)
        return 0.5 * nv + 0.5 * i4

    best_w = w_lancet
    best_leak = float((U_mat @ _simulate_q4(w_lancet)).norm().item())

    w_curr = w_lancet.clone()
    target_row_norms = w_lancet.norm(p=2, dim=-1, keepdim=True)
    w_lancet_norm = float(w_lancet.norm().item())

    for _ in range(num_iters):
        q_k = _simulate_q4(w_curr)
        E_k = U_mat.T @ (U_mat @ q_k)
        w_next = w_curr - step_size * E_k

        if preserve_bf16_nullspace:
            w_next = w_next - U_mat.T @ (U_mat @ w_next)

        # Clamp drift relative to the Lancet projection.
        diff = w_next - w_lancet
        rel_drift = float(diff.norm().item()) / (w_lancet_norm + 1e-8)
        if rel_drift > max_rel_drift:
            w_next = w_lancet + diff * (max_rel_drift / rel_drift)

        if preserve_row_norm:
            new_row_norms = w_next.norm(p=2, dim=-1, keepdim=True)
            w_next = w_next * (target_row_norms / (new_row_norms + 1e-8))

        leak_next = float((U_mat @ _simulate_q4(w_next)).norm().item())

        # Re-check drift on the final candidate before accepting it.
        rel_drift_next = float((w_next - w_lancet).norm().item()) / (w_lancet_norm + 1e-8)
        if rel_drift_next <= max_rel_drift + 1e-5 and leak_next < best_leak:
            best_leak = leak_next
            best_w = w_next.clone()

        w_curr = w_next

    return best_w.to(device=orig_device, dtype=orig_dtype)


def _as_direction_list(value: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
    """Normalize a single per-layer direction spec into a list of vectors."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            return [value]
        return [value[i] for i in range(value.shape[0])]
    return [torch.as_tensor(v) for v in value]


def _resolve_per_layer_directions(
    directions: dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    num_layers: int,
) -> dict[int, list[torch.Tensor]]:
    """
    Resolve a direction spec into a per-layer mapping ``{layer_idx: [dirs]}``.

    Accepted forms:
        - dict mapping layer index -> single vector, list of vectors, or
          tensor whose rows are vectors.
        - tensor of shape (D,): one direction applied to every layer.
        - tensor of shape (L, D): row ``i`` is the direction for layer ``i``.
        - list of length L: element ``i`` holds the directions for layer ``i``.
    """
    if isinstance(directions, dict):
        return {int(k): _as_direction_list(v) for k, v in directions.items()}

    if isinstance(directions, torch.Tensor):
        if directions.ndim == 1:
            return {i: [directions] for i in range(num_layers)}
        return {i: [directions[i]] for i in range(min(directions.shape[0], num_layers))}

    resolved = {}
    for i, entry in enumerate(directions):
        if i >= num_layers:
            break
        resolved[i] = _as_direction_list(entry)
    return resolved


def _forward_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run a forward pass and return raw logits (handles HF and sml-nano)."""
    out = model(input_ids)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    return out


def compute_layer_scr_profile(
    model: torch.nn.Module,
    target_dirs_per_layer: (
        dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor
    ),
    anchor_input_ids: torch.Tensor,
    candidate_layers: list[int] | None = None,
    scale: float = 1.0,
    preserve_row_norm: bool = True,
    *,
    cot_input_ids: torch.Tensor | None = None,
    cot_tag_token_ids: list[int] | None = None,
    lambda_cot: float = 0.5,
) -> list[dict[str, float | int]]:
    """
    Score candidate layers by signal-to-collateral ratio (SCR) for surgery.

    For each candidate layer, temporarily apply norm-preserving subspace
    projection to ``self_attn.o_proj`` and ``mlp.down_proj`` (weights are
    restored afterward, so no per-layer deepcopy is needed), then measure:

    - ``signal``: sum of squared removed target-projection energy on both
      weights.
    - ``kl_anchor``: forward KL divergence between original and modified
      model logits on ``anchor_input_ids``.
    - ``kl_cot``: when ``cot_input_ids`` is provided, forward KL on the CoT
      batch; if ``cot_tag_token_ids`` is non-empty this is the blend
      ``0.5 * full-sequence KL + 0.5 * KL over the tag-logit slice``
      (``log_softmax(logits[..., cot_tag_token_ids])``).
    - ``kl_collateral``: the collateral penalty - ``lambda_cot * kl_cot +
      (1 - lambda_cot) * kl_anchor`` when CoT inputs are provided, otherwise
      just ``kl_anchor``.
    - ``scr``: ``signal / (kl_collateral + 1e-6)``.

    Args:
        model: Transformer model instance (HF or sml-nano).
        target_dirs_per_layer: Target directions per layer.
        anchor_input_ids: Token ids used to measure collateral damage.
        candidate_layers: Layers to score (None means all layers).
        scale: Projection scale passed through to the projection.
        preserve_row_norm: Passed through to the projection.
        cot_input_ids: Optional CoT token ids used to measure reasoning-side
            collateral damage alongside ``anchor_input_ids``.
        cot_tag_token_ids: Optional deliberation tag token ids; when
            non-empty (and CoT inputs are provided), half of ``kl_cot`` comes
            from the KL over those tag logits only.
        lambda_cot: Weight of ``kl_cot`` in the blended ``kl_collateral``.

    Returns:
        List of dicts with keys ``layer_idx``, ``signal``, ``kl_anchor``,
        ``kl_cot``, ``kl_collateral``, and ``scr``, sorted descending by
        ``scr``.
    """
    layers = get_transformer_layers(model)
    num_layers = len(layers)
    if candidate_layers is None:
        candidate_layers = list(range(num_layers))

    target_map = _resolve_per_layer_directions(target_dirs_per_layer, num_layers)
    model_device = next(model.parameters()).device
    anchor_input_ids = anchor_input_ids.to(model_device)

    use_cot = cot_input_ids is not None
    if use_cot:
        cot_input_ids = cot_input_ids.to(model_device)
    tag_ids = [int(t) for t in (cot_tag_token_ids or [])] if use_cot else []
    has_tag_kl = use_cot and len(tag_ids) > 0

    was_training = model.training
    model.eval()
    with torch.no_grad():
        orig_logits = _forward_logits(model, anchor_input_ids)
        orig_log_probs = torch.log_softmax(orig_logits.float(), dim=-1)
        orig_probs = orig_log_probs.exp()

        orig_cot_log_probs = None
        orig_cot_probs = None
        if use_cot:
            orig_cot_logits = _forward_logits(model, cot_input_ids)
            orig_cot_log_probs = torch.log_softmax(orig_cot_logits.float(), dim=-1)
            orig_cot_probs = orig_cot_log_probs.exp()

        profile: list[dict[str, float | int]] = []
        try:
            for idx in candidate_layers:
                layer = layers[idx]
                dirs = target_map.get(idx, [])
                basis = _orthonormalize_basis(dirs, device=model_device)

                # Signal: removed target projection energy on both weights.
                signal = 0.0
                for weight in (layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight):
                    w_f = weight.detach().to(dtype=torch.float32)
                    for r in basis:
                        signal += float((scale * (r @ w_f)).pow(2).sum())

                # Temporarily apply the projection, measure KL, then restore.
                orig_attn_weight = layer.self_attn.o_proj.weight
                orig_mlp_weight = layer.mlp.down_proj.weight
                layer.self_attn.o_proj.weight = torch.nn.Parameter(
                    project_out_subspace_norm_preserving(
                        orig_attn_weight, dirs, scale=scale, preserve_row_norm=preserve_row_norm
                    ),
                    requires_grad=False,
                )
                layer.mlp.down_proj.weight = torch.nn.Parameter(
                    project_out_subspace_norm_preserving(
                        orig_mlp_weight, dirs, scale=scale, preserve_row_norm=preserve_row_norm
                    ),
                    requires_grad=False,
                )
                try:
                    mod_logits = _forward_logits(model, anchor_input_ids)
                    mod_log_probs = torch.log_softmax(mod_logits.float(), dim=-1)
                    kl_anchor = float(
                        (orig_probs * (orig_log_probs - mod_log_probs)).sum(dim=-1).mean()
                    )

                    if use_cot:
                        mod_cot_logits = _forward_logits(model, cot_input_ids)
                        mod_cot_log_probs = torch.log_softmax(mod_cot_logits.float(), dim=-1)
                        kl_cot_full = float(
                            (orig_cot_probs * (orig_cot_log_probs - mod_cot_log_probs))
                            .sum(dim=-1)
                            .mean()
                        )
                        if has_tag_kl:
                            # Full-vocab binary Bernoulli KL per tag token id, so a
                            # uniform shift in tag logits is not normalized away.
                            mod_cot_probs = mod_cot_log_probs.exp()
                            eps = 1e-8
                            p_ref = orig_cot_probs[..., tag_ids].clamp(eps, 1.0 - eps)
                            p_mod = mod_cot_probs[..., tag_ids].clamp(eps, 1.0 - eps)
                            kl_cot_tag = float(
                                (
                                    p_ref * (p_ref.log() - p_mod.log())
                                    + (1.0 - p_ref) * ((1.0 - p_ref).log() - (1.0 - p_mod).log())
                                ).mean()
                            )
                            kl_cot = 0.5 * kl_cot_full + 0.5 * kl_cot_tag
                        else:
                            kl_cot = kl_cot_full
                        kl = lambda_cot * kl_cot + (1.0 - lambda_cot) * kl_anchor
                    else:
                        kl_cot = 0.0
                        kl = kl_anchor
                finally:
                    layer.self_attn.o_proj.weight = orig_attn_weight
                    layer.mlp.down_proj.weight = orig_mlp_weight

                scr = signal / (kl + 1e-6)
                entry: dict[str, float | int] = {
                    "layer_idx": int(idx),
                    "signal": float(signal),
                    "kl_collateral": float(kl),
                    "scr": float(scr),
                }
                if use_cot:
                    entry["kl_anchor"] = float(kl_anchor)
                    entry["kl_cot"] = float(kl_cot)
                profile.append(entry)
        finally:
            if was_training:
                model.train()

    return sorted(profile, key=lambda entry: entry["scr"], reverse=True)


def apply_subspace_ablation(
    model: torch.nn.Module,
    direction: torch.Tensor,
    layer_indices: list[int] | None = None,
    scale: float = 1.0,
) -> torch.nn.Module:
    """
    Apply subspace ablation to specific layers of a model.

    Args:
        model: Transformer model instance (HF or sml-nano)
        direction: Direction vector to project out
        layer_indices: Indices of layers to modify (None means all layers)
        scale: Scaling factor for projection

    Returns:
        Modified model with subspace ablation applied
    """
    # Create deep copy of model
    modified_model = copy.deepcopy(model)
    layers = get_transformer_layers(modified_model)

    # Determine which layers to modify
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    # Apply projection to selected layers
    with torch.no_grad():
        for i in layer_indices:
            layer = layers[i]

            # Update attention output projection
            layer.self_attn.o_proj.weight = torch.nn.Parameter(
                project_out_rank1_subspace(layer.self_attn.o_proj.weight, direction, scale)
            )

            # Update MLP down projection
            layer.mlp.down_proj.weight = torch.nn.Parameter(
                project_out_rank1_subspace(layer.mlp.down_proj.weight, direction, scale)
            )

    return modified_model


def apply_lancet_biprojected_surgery(
    model: torch.nn.Module,
    target_dirs: dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    anchor_dirs: (
        dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor | None
    ) = None,
    layer_indices: list[int] | None = None,
    scale: float = 1.0,
    preserve_row_norm: bool = True,
    layer_weights: dict[int, float] | None = None,
    *,
    quant_snap: bool = False,
    quant_snap_iters: int = 4,
    max_block16_crest_factor: float | None = None,
) -> torch.nn.Module:
    """
    Apply Lancet-style biprojected subspace surgery to selected layers.

    Each target direction is first biprojected against the anchor directions
    for that layer (removing components that overlap with the anchors, e.g.
    reasoning directions we want to preserve), then the resulting residual
    directions are projected out of ``self_attn.o_proj`` and ``mlp.down_proj``
    with row-norm preservation.

    Args:
        model: Transformer model instance (HF or sml-nano).
        target_dirs: Target directions per layer.
        anchor_dirs: Optional anchor directions per layer to protect.
        layer_indices: Indices of layers to modify (None means all layers).
        scale: Scaling factor for the projection.
        preserve_row_norm: Whether to preserve row L2 norms of the weights.
        layer_weights: Optional mapping of layer index -> multiplier applied
            on top of ``scale`` for that layer; layers absent from the dict
            use the base ``scale`` (multiplier 1.0).
        quant_snap: If True, use ``project_and_snap_subspace_norm_preserving``
            (Lancet projection plus quantization-aware subspace snapping)
            instead of the plain projection.
        quant_snap_iters: Number of snap iterations when ``quant_snap`` is on.
        max_block16_crest_factor: Optional micro-block crest-factor cap. With
            ``quant_snap`` it is passed as ``max_block_cf`` (falling back to
            3.2 when None); without ``quant_snap`` the raw weight is first
            regularized with it (``block_size=16``) before the projection.

    Returns:
        Modified (deep-copied) model with biprojected ablation applied.
    """
    modified_model = copy.deepcopy(model)
    layers = get_transformer_layers(modified_model)
    num_layers = len(layers)

    if layer_indices is None:
        layer_indices = list(range(num_layers))

    target_map = _resolve_per_layer_directions(target_dirs, num_layers)
    anchor_map = (
        _resolve_per_layer_directions(anchor_dirs, num_layers) if anchor_dirs is not None else {}
    )

    with torch.no_grad():
        for idx in layer_indices:
            layer = layers[idx]

            target_layer_dirs = target_map.get(idx, [])
            anchor_layer_dirs = anchor_map.get(idx, [])

            # Biproject each target direction against the anchors so the
            # ablated subspace does not overlap the protected one.
            if anchor_layer_dirs:
                resolved_dirs = [
                    biproject_direction(d, anchor_layer_dirs) for d in target_layer_dirs
                ]
            else:
                resolved_dirs = list(target_layer_dirs)

            # Per-layer scaling multiplier on top of the base ``scale``.
            layer_scale = scale
            if layer_weights is not None:
                layer_scale = scale * float(layer_weights.get(idx, 1.0))

            orig_attn_weight = layer.self_attn.o_proj.weight
            orig_mlp_weight = layer.mlp.down_proj.weight

            if quant_snap:
                # H18: Lancet projection plus quantization-aware snapping.
                snap_block_cf = (
                    max_block16_crest_factor if max_block16_crest_factor is not None else 3.2
                )
                layer.self_attn.o_proj.weight = torch.nn.Parameter(
                    project_and_snap_subspace_norm_preserving(
                        orig_attn_weight,
                        resolved_dirs,
                        scale=layer_scale,
                        num_iters=quant_snap_iters,
                        max_block_cf=snap_block_cf,
                        preserve_row_norm=preserve_row_norm,
                    ),
                    requires_grad=orig_attn_weight.requires_grad,
                )
                layer.mlp.down_proj.weight = torch.nn.Parameter(
                    project_and_snap_subspace_norm_preserving(
                        orig_mlp_weight,
                        resolved_dirs,
                        scale=layer_scale,
                        num_iters=quant_snap_iters,
                        max_block_cf=snap_block_cf,
                        preserve_row_norm=preserve_row_norm,
                    ),
                    requires_grad=orig_mlp_weight.requires_grad,
                )
            else:
                attn_input = orig_attn_weight
                mlp_input = orig_mlp_weight
                if max_block16_crest_factor is not None:
                    attn_input = regularize_microblock_crest_factor(
                        orig_attn_weight, block_size=16, max_block_cf=max_block16_crest_factor
                    )
                    mlp_input = regularize_microblock_crest_factor(
                        orig_mlp_weight, block_size=16, max_block_cf=max_block16_crest_factor
                    )
                layer.self_attn.o_proj.weight = torch.nn.Parameter(
                    project_out_subspace_norm_preserving(
                        attn_input,
                        resolved_dirs,
                        scale=layer_scale,
                        preserve_row_norm=preserve_row_norm,
                    ),
                    requires_grad=orig_attn_weight.requires_grad,
                )
                layer.mlp.down_proj.weight = torch.nn.Parameter(
                    project_out_subspace_norm_preserving(
                        mlp_input,
                        resolved_dirs,
                        scale=layer_scale,
                        preserve_row_norm=preserve_row_norm,
                    ),
                    requires_grad=orig_mlp_weight.requires_grad,
                )

    return modified_model


@dataclass(frozen=True)
class LancetRecipe:
    """Serializable Lancet surgery recipe: biprojected bases plus per-layer weights."""

    model_id: str
    base_model: str
    extraction_mode: str
    d_model: int
    num_layers: int
    scale: float
    top_k_layers: int
    selected_layers: list[int]
    layer_weights: dict[int, float]
    basis_per_layer: dict[int, torch.Tensor]
    probes_sha256: str
    schema_version: str = "1.0"


def build_lancet_recipe(
    model: torch.nn.Module,
    target_dirs: dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    anchor_dirs: (
        dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor | None
    ),
    anchor_input_ids: torch.Tensor,
    *,
    model_id: str,
    base_model: str,
    extraction_mode: str = "activation_contrastive",
    top_k_layers: int = 3,
    scale: float = 1.0,
    probes_sha256: str = "",
    preserve_row_norm: bool = True,
) -> LancetRecipe:
    """
    Build a serializable Lancet recipe from probe directions and anchors.

    Each layer's target directions are biprojected against its anchor
    directions, candidate layers are scored with ``compute_layer_scr_profile``,
    the ``top_k_layers`` best layers are selected (descending by SCR) with soft
    SCR-derived weights, and an orthonormal CPU float32 basis is stored for
    every selected layer.

    Args:
        model: Transformer model instance (HF or sml-nano).
        target_dirs: Target directions per layer.
        anchor_dirs: Optional anchor directions per layer to protect.
        anchor_input_ids: Token ids used to measure collateral damage in the
            SCR profile.
        model_id: Identifier of the model the recipe targets.
        base_model: Base (pre-surgery) model identifier.
        extraction_mode: How the target directions were extracted.
        top_k_layers: Number of highest-SCR layers to keep.
        scale: Base projection scale for the surgery.
        probes_sha256: Hash of the probe set used for extraction.
        preserve_row_norm: Passed through to the SCR profile projection.

    Returns:
        A frozen ``LancetRecipe`` ready for ``save_lancet_recipe`` /
        ``apply_lancet_recipe``.
    """
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    target_map = _resolve_per_layer_directions(target_dirs, num_layers)
    anchor_map = (
        _resolve_per_layer_directions(anchor_dirs, num_layers) if anchor_dirs is not None else {}
    )

    # Biproject each layer's target directions against its anchor directions so
    # the stored bases do not overlap the protected subspaces.
    effective_dirs: dict[int, list[torch.Tensor]] = {}
    for idx, layer_dirs in target_map.items():
        anchor_layer_dirs = anchor_map.get(idx, [])
        if anchor_layer_dirs:
            effective_dirs[idx] = [biproject_direction(d, anchor_layer_dirs) for d in layer_dirs]
        else:
            effective_dirs[idx] = list(layer_dirs)

    candidate_layers = sorted(idx for idx, dirs in effective_dirs.items() if dirs)
    if not candidate_layers:
        raise ValueError("No target directions resolved for any layer")

    profile = compute_layer_scr_profile(
        model,
        {idx: effective_dirs[idx] for idx in candidate_layers},
        anchor_input_ids,
        candidate_layers=candidate_layers,
        scale=scale,
        preserve_row_norm=preserve_row_norm,
    )

    top_rows = profile[: max(1, int(top_k_layers))]
    selected_layers = [int(row["layer_idx"]) for row in top_rows]
    max_scr = max(float(row["scr"]) for row in profile)
    layer_weights = {
        int(row["layer_idx"]): round(min(1.0, max(0.1, float(row["scr"]) / max(max_scr, 1e-12))), 6)
        for row in top_rows
    }

    basis_per_layer: dict[int, torch.Tensor] = {}
    d_model: int | None = None
    for idx in selected_layers:
        basis_vectors = _orthonormalize_basis(effective_dirs[idx], device=torch.device("cpu"))
        if not basis_vectors:
            raise ValueError(f"Layer {idx} has no usable directions after biprojection")
        basis = torch.stack(basis_vectors, dim=0).contiguous()
        d_model = int(basis.shape[1])
        basis_per_layer[idx] = basis

    if d_model is None:
        raise ValueError("Failed to build a basis for any selected layer")

    return LancetRecipe(
        model_id=model_id,
        base_model=base_model,
        extraction_mode=extraction_mode,
        d_model=d_model,
        num_layers=num_layers,
        scale=float(scale),
        top_k_layers=int(top_k_layers),
        selected_layers=selected_layers,
        layer_weights=layer_weights,
        basis_per_layer=basis_per_layer,
        probes_sha256=probes_sha256,
    )


def save_lancet_recipe(recipe: LancetRecipe, output_dir: Path | str) -> Path:
    """
    Persist a recipe as ``lancet_recipe.json`` + ``lancet_basis.safetensors``.

    Args:
        recipe: Recipe produced by ``build_lancet_recipe``.
        output_dir: Directory to create; existing files are overwritten.

    Returns:
        The resolved output directory as a ``Path``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": recipe.schema_version,
        "model_id": recipe.model_id,
        "base_model": recipe.base_model,
        "extraction_mode": recipe.extraction_mode,
        "d_model": int(recipe.d_model),
        "num_layers": int(recipe.num_layers),
        "scale": float(recipe.scale),
        "top_k_layers": int(recipe.top_k_layers),
        "selected_layers": [int(idx) for idx in recipe.selected_layers],
        "layer_weights": {str(idx): float(w) for idx, w in recipe.layer_weights.items()},
        "probes_sha256": recipe.probes_sha256,
    }
    metadata_path = output_path / "lancet_recipe.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    basis_tensors = {
        f"layer_{idx}": basis.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for idx, basis in recipe.basis_per_layer.items()
    }
    basis_path = output_path / "lancet_basis.safetensors"
    save_file(basis_tensors, str(basis_path))

    return output_path


def load_lancet_recipe(recipe_dir: Path | str) -> LancetRecipe:
    """
    Load and validate a recipe saved by ``save_lancet_recipe``.

    Args:
        recipe_dir: Directory holding ``lancet_recipe.json`` and
            ``lancet_basis.safetensors``.

    Returns:
        The reconstructed ``LancetRecipe`` with CPU float32 bases.

    Raises:
        ValueError: On missing files, unsupported schema, invalid layer
            indices or weights, malformed bases, non-finite entries, or bases
            whose rows are not unit L2 norm.
    """
    dir_path = Path(recipe_dir)
    metadata_path = dir_path / "lancet_recipe.json"
    basis_path = dir_path / "lancet_basis.safetensors"
    if not metadata_path.is_file():
        raise ValueError(f"Missing recipe metadata file: {metadata_path}")
    if not basis_path.is_file():
        raise ValueError(f"Missing recipe basis file: {basis_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported recipe schema_version: {metadata.get('schema_version')!r}")

    try:
        model_id = str(metadata["model_id"])
        base_model = str(metadata["base_model"])
        extraction_mode = str(metadata["extraction_mode"])
        d_model = int(metadata["d_model"])
        num_layers = int(metadata["num_layers"])
        scale = float(metadata["scale"])
        top_k_layers = int(metadata["top_k_layers"])
        probes_sha256 = str(metadata.get("probes_sha256", ""))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid lancet_recipe.json metadata: {exc}") from exc

    if d_model <= 0 or num_layers <= 0:
        raise ValueError(f"Invalid d_model/num_layers: {d_model}/{num_layers}")

    raw_layers = metadata.get("selected_layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("selected_layers must be a non-empty list of layer indices")
    selected_layers = [int(idx) for idx in raw_layers]
    if any(idx < 0 or idx >= num_layers for idx in selected_layers):
        raise ValueError(f"selected_layers {selected_layers} out of range [0, {num_layers})")

    raw_weights = metadata.get("layer_weights")
    if not isinstance(raw_weights, dict):
        raise ValueError("layer_weights must be an object with stringified layer keys")
    layer_weights: dict[int, float] = {}
    for key, value in raw_weights.items():
        weight = float(value)
        if not 0.0 < weight <= 1.0:
            raise ValueError(f"layer_weights[{key!r}] must be in (0, 1]; got {weight}")
        layer_weights[int(key)] = weight
    missing = [idx for idx in selected_layers if idx not in layer_weights]
    if missing:
        raise ValueError(f"layer_weights missing selected layers: {missing}")

    basis_tensors = load_file(str(basis_path))
    basis_per_layer: dict[int, torch.Tensor] = {}
    for idx in selected_layers:
        key = f"layer_{idx}"
        if key not in basis_tensors:
            raise ValueError(f"Missing basis tensor {key!r} in lancet_basis.safetensors")
        basis = basis_tensors[key].to(device="cpu", dtype=torch.float32)
        if basis.ndim != 2 or basis.shape[0] < 1 or basis.shape[1] != d_model:
            raise ValueError(
                f"Basis {key!r} must have shape (k, {d_model}) with k >= 1; "
                f"got {tuple(basis.shape)}"
            )
        if not bool(torch.isfinite(basis).all()):
            raise ValueError(f"Basis {key!r} contains non-finite values")
        row_norm_diff = (basis.norm(p=2, dim=-1) - 1.0).abs()
        if float(row_norm_diff.max()) >= 1e-5:
            raise ValueError(f"Basis {key!r} rows are not unit L2 norm")
        basis_per_layer[idx] = basis.contiguous()

    return LancetRecipe(
        model_id=model_id,
        base_model=base_model,
        extraction_mode=extraction_mode,
        d_model=d_model,
        num_layers=num_layers,
        scale=scale,
        top_k_layers=top_k_layers,
        selected_layers=selected_layers,
        layer_weights=layer_weights,
        basis_per_layer=basis_per_layer,
        probes_sha256=probes_sha256,
    )


def apply_lancet_recipe(
    model: torch.nn.Module, recipe: LancetRecipe, *, in_place: bool = False
) -> torch.nn.Module:
    """
    Apply a saved Lancet recipe to a model.

    Each selected layer's ``self_attn.o_proj`` and ``mlp.down_proj`` weights
    get the stored orthonormal basis projected out with row-norm preservation,
    scaled by ``recipe.scale * recipe.layer_weights.get(idx, 1.0)``.

    Args:
        model: Transformer model instance (HF or sml-nano).
        recipe: Recipe produced by ``build_lancet_recipe`` / ``load_lancet_recipe``.
        in_place: If True, modify ``model`` directly instead of deep-copying.

    Returns:
        The modified model (a deep copy unless ``in_place=True``).
    """
    target_model = model if in_place else copy.deepcopy(model)
    layers = get_transformer_layers(target_model)

    with torch.no_grad():
        for idx in recipe.selected_layers:
            layer = layers[idx]
            basis = recipe.basis_per_layer[idx]
            layer_scale = recipe.scale * float(recipe.layer_weights.get(idx, 1.0))

            orig_attn_weight = layer.self_attn.o_proj.weight
            orig_mlp_weight = layer.mlp.down_proj.weight
            layer.self_attn.o_proj.weight = torch.nn.Parameter(
                project_out_subspace_norm_preserving(
                    orig_attn_weight, basis, scale=layer_scale, preserve_row_norm=True
                ),
                requires_grad=orig_attn_weight.requires_grad,
            )
            layer.mlp.down_proj.weight = torch.nn.Parameter(
                project_out_subspace_norm_preserving(
                    orig_mlp_weight, basis, scale=layer_scale, preserve_row_norm=True
                ),
                requires_grad=orig_mlp_weight.requires_grad,
            )

    return target_model


_TARGET_WEIGHT_KEY_RE = re.compile(
    r"^(?:model\.|language_model\.model\.|language_model\.|model\.language_model\.)?"
    r"layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj)\.weight$"
)


def _resolve_safetensors_shards(checkpoint_dir: Path) -> tuple[list[str], dict[str, Any] | None]:
    """
    Resolve the ordered list of safetensors shard filenames in a checkpoint.

    Resolution order:

    1. ``model.safetensors.index.json``: shard filenames are taken from
       ``weight_map.values()`` preserving first-seen order, and the parsed
       index document is returned as the second element.
    2. ``model.safetensors``: a single-file checkpoint; the index is ``None``.
    3. ``model-*-of-*.safetensors``: sorted glob fallback; the index is ``None``.

    Args:
        checkpoint_dir: Directory that may hold a sharded or single-file
            safetensors checkpoint.

    Returns:
        Tuple ``(shards, index_data)`` with ordered unique shard filenames and
        the parsed index JSON (``None`` when no index file exists).

    Raises:
        FileNotFoundError: If no safetensors shard can be resolved.
    """
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index_data.get("weight_map", {}) if isinstance(index_data, dict) else {}
        shards: list[str] = []
        seen: set[str] = set()
        for filename in weight_map.values():
            name = str(filename)
            if name not in seen:
                seen.add(name)
                shards.append(name)
        return shards, index_data

    if (checkpoint_dir / "model.safetensors").is_file():
        return ["model.safetensors"], None

    shard_names = sorted(p.name for p in checkpoint_dir.glob("model-*-of-*.safetensors"))
    if not shard_names:
        raise FileNotFoundError(f"No safetensors shards found in {checkpoint_dir}")
    return shard_names, None


def _copy_writable_file(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` as a writable 0o644 file.

    Read-only 0o444 HF cache blobs must never cause ``PermissionError`` on
    re-runs, so the destination is unlinked first and forced to 0o644.
    """
    dst.unlink(missing_ok=True)
    shutil.copyfile(src, dst)
    os.chmod(dst, 0o644)


def _copy_aux_files(src_dir: Path, dst_dir: Path, *, copy_index: bool) -> None:
    """Copy the safetensors index (optionally) and all non-shard files."""
    if copy_index:
        index_src = src_dir / "model.safetensors.index.json"
        if index_src.is_file():
            _copy_writable_file(index_src, dst_dir / "model.safetensors.index.json")
    for aux_path in sorted(src_dir.glob("*")):
        name = aux_path.name
        if not aux_path.is_file() or name.endswith(".safetensors"):
            # Embedded drafter/MTP weights must survive the streaming copy so
            # the H22 drafter sync can find them in the materialized output.
            if name == "mtp.safetensors" and aux_path.is_file():
                _copy_writable_file(aux_path, dst_dir / name)
            continue
        if name == "model.safetensors.index.json":
            continue
        _copy_writable_file(aux_path, dst_dir / name)


def _parse_target_layer_weight_key(key: str) -> tuple[int, str] | None:
    """
    Parse a transformer checkpoint weight key into ``(layer_idx, proj_name)``.

    Matches ``layers.<idx>.self_attn.o_proj.weight`` and
    ``layers.<idx>.mlp.down_proj.weight`` with an optional ``model.`` prefix
    (the HF key layout). Any other key returns ``None``.
    """
    match = _TARGET_WEIGHT_KEY_RE.match(key)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def _compute_row_crest_factor(weight: torch.Tensor) -> float:
    """
    Compute the peak-to-mean row-norm ratio (crest factor) of a 2D weight.

    Returns ``max_row_norm / mean_row_norm``; 1.0 when all rows share the same
    norm. Non-2D or empty tensors report 1.0 (no crest to flag).
    """
    if weight.ndim != 2 or weight.numel() == 0:
        return 1.0
    norms = torch.linalg.vector_norm(weight, ord=2, dim=-1, dtype=torch.float32)
    mean_norm = norms.mean().clamp_min(1e-8)
    return float((norms.max() / mean_norm).item())


def apply_lancet_recipe_streaming(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    recipe: LancetRecipe,
    *,
    preserve_row_norm: bool = True,
    cleanup_input_shards: bool = False,
    quant_snap: bool = False,
    quant_snap_iters: int = 4,
    max_block16_crest_factor: float | None = None,
) -> dict[str, Any]:
    """
    Apply a Lancet recipe to a sharded safetensors checkpoint on disk.

    Unlike ``apply_lancet_recipe`` (in-memory, requires the full model), this
    streams one shard at a time on CPU: each shard is loaded, target weights
    (``self_attn.o_proj`` / ``mlp.down_proj`` of layers selected by the recipe)
    are modified in-place via ``project_out_subspace_norm_preserving``, and the
    shard is written back to ``output_dir`` before the next one is loaded. Peak
    memory and per-shard disk footprint are tracked so callers can verify the
    streaming behavior.

    Args:
        checkpoint_dir: Directory holding the source safetensors checkpoint
            (single file, sharded with index JSON, or sharded without index).
        output_dir: Directory to create for the modified checkpoint.
        recipe: Recipe produced by ``build_lancet_recipe`` /
            ``load_lancet_recipe``.
        preserve_row_norm: Passed through to the projection.
        cleanup_input_shards: If True, delete each processed input shard
            (only when ``checkpoint_dir`` and ``output_dir`` differ).
        quant_snap: If True, target weights are modified with
            ``project_and_snap_subspace_norm_preserving`` (Lancet projection
            plus quantization-aware subspace snapping) instead of the plain
            projection.
        quant_snap_iters: Number of snap iterations when ``quant_snap`` is on.
        max_block16_crest_factor: Optional micro-block crest-factor cap. With
            ``quant_snap`` it is passed as ``max_block_cf`` (falling back to
            3.2 when None); without ``quant_snap`` each target weight is first
            regularized with it (``block_size=16``) before the projection.

    Returns:
        Dict with ``shards_processed``, ``tensors_modified``,
        ``max_row_norm_drift``, ``max_row_crest_factor``, ``provider_ready``
        (``max_row_crest_factor < 25.0``), ``peak_memory_bytes``,
        ``peak_memory_gb``, ``peak_disk_shard_bytes``, plus the
        ``quant_snap``, ``quant_snap_iters`` and ``max_block16_crest_factor``
        settings echoed back.
    """
    src_dir = Path(checkpoint_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    shards, index_data = _resolve_safetensors_shards(src_dir)
    same_dir = src_dir.resolve() == dst_dir.resolve()

    tensors_modified = 0
    max_row_norm_drift = 0.0
    max_row_crest_factor = 1.0
    peak_disk_shard_bytes = 0

    tracemalloc.start()
    try:
        for shard_name in shards:
            shard_path = src_dir / shard_name
            shard_size = shard_path.stat().st_size
            peak_disk_shard_bytes = max(peak_disk_shard_bytes, shard_size)

            shard_tensors = load_file(str(shard_path))

            for key, tensor in shard_tensors.items():
                parsed = _parse_target_layer_weight_key(key)
                if parsed is None:
                    continue
                layer_idx, _proj_name = parsed
                if layer_idx not in recipe.selected_layers:
                    continue
                if layer_idx not in recipe.basis_per_layer:
                    continue

                layer_scale = recipe.scale * float(recipe.layer_weights.get(layer_idx, 1.0))
                orig_row_norms = tensor.detach().to(dtype=torch.float32).norm(p=2, dim=-1)
                if quant_snap:
                    # H18: Lancet projection plus quantization-aware snapping.
                    modified = project_and_snap_subspace_norm_preserving(
                        tensor,
                        recipe.basis_per_layer[layer_idx],
                        scale=layer_scale,
                        num_iters=quant_snap_iters,
                        max_block_cf=(
                            max_block16_crest_factor
                            if max_block16_crest_factor is not None
                            else 3.2
                        ),
                        preserve_row_norm=preserve_row_norm,
                    )
                else:
                    proj_input = tensor
                    if max_block16_crest_factor is not None:
                        proj_input = regularize_microblock_crest_factor(
                            tensor, block_size=16, max_block_cf=max_block16_crest_factor
                        )
                    modified = project_out_subspace_norm_preserving(
                        proj_input,
                        recipe.basis_per_layer[layer_idx],
                        scale=layer_scale,
                        preserve_row_norm=preserve_row_norm,
                    )
                if orig_row_norms.numel() > 0:
                    new_row_norms = modified.detach().to(dtype=torch.float32).norm(p=2, dim=-1)
                    drift = float((new_row_norms - orig_row_norms).abs().max().item())
                    max_row_norm_drift = max(max_row_norm_drift, drift)
                shard_tensors[key] = modified
                tensors_modified += 1

            for tensor in shard_tensors.values():
                if tensor.ndim == 2:
                    crest = _compute_row_crest_factor(tensor)
                    max_row_crest_factor = max(max_row_crest_factor, crest)

            out_path = dst_dir / shard_name
            save_file(shard_tensors, str(out_path))
            peak_disk_shard_bytes = max(peak_disk_shard_bytes, out_path.stat().st_size)

            del shard_tensors
            gc.collect()

            if cleanup_input_shards and not same_dir:
                shard_path.unlink()
    finally:
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    if not same_dir:
        _copy_aux_files(src_dir, dst_dir, copy_index=index_data is not None)

    return {
        "shards_processed": len(shards),
        "tensors_modified": tensors_modified,
        "max_row_norm_drift": max_row_norm_drift,
        "max_row_crest_factor": max_row_crest_factor,
        "provider_ready": max_row_crest_factor < 25.0,
        "peak_memory_bytes": int(peak_memory_bytes),
        "peak_memory_gb": float(peak_memory_bytes) / (1024**3),
        "peak_disk_shard_bytes": int(peak_disk_shard_bytes),
        "quant_snap": bool(quant_snap),
        "quant_snap_iters": int(quant_snap_iters),
        "max_block16_crest_factor": max_block16_crest_factor,
    }


# ---------------------------------------------------------------------------
# MoE expert routing profiles (Tri-Anchor Lancet for MoE layers)
# ---------------------------------------------------------------------------


def _parse_router_output(output: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Split a router module's forward output into ``(logits_or_weights, indices)``.

    Routers either return a plain ``(.., E)`` logits tensor or a tuple such as
    ``(weights, indices)`` / ``(logits, aux_loss)``. The first floating tensor
    with at least 1 dimension is treated as logits/weights; the first integer
    tensor is treated as top-k expert indices; non-tensor and 0-dim entries
    (scalar aux losses) are ignored.
    """
    if torch.is_tensor(output):
        return output, None
    if not isinstance(output, (tuple, list)):
        return None, None

    logits: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    for element in output:
        if not torch.is_tensor(element):
            continue
        if element.ndim < 1:
            continue
        if element.dtype.is_floating_point:
            if logits is None:
                logits = element
        elif indices is None:
            indices = element
    return logits, indices


def _router_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert a router's ``(.., E)`` logits/weights tensor to float32 probabilities.

    Already-normalized outputs (non-negative rows) are returned as-is; raw
    logits get a softmax over the last dimension.
    """
    flat = logits.detach().to(dtype=torch.float32)
    if bool((flat < -1e-6).any()) or bool((flat > 1.0 + 1e-3).any()):
        flat = torch.softmax(flat, dim=-1)
    return flat


def _run_router_forward_passes(
    model: torch.nn.Module,
    routers: dict[int, torch.nn.Module],
    inputs: list[torch.Tensor],
    forced_top_k: int | None,
) -> tuple[dict[int, torch.Tensor | None], dict[int, torch.Tensor | None], dict[int, int]]:
    """
    Run forward passes with router hooks registered and accumulate statistics.

    Returns per-layer ``(counts, prob_sum, num_rows)``. When a router emits
    integer indices (or a top-k is forced) ``counts`` holds top-k selection
    tallies; ``prob_sum`` still accumulates the full softmax mass per expert.
    Tensors stay ``None`` for layers whose router never produced output.
    """
    device = next(model.parameters()).device
    counts: dict[int, torch.Tensor | None] = {idx: None for idx in routers}
    prob_sum: dict[int, torch.Tensor | None] = {idx: None for idx in routers}
    num_rows: dict[int, int] = {idx: 0 for idx in routers}

    def _make_hook(layer_idx: int):
        def _hook(_module, _inputs, output):
            logits, indices = _parse_router_output(output)
            if logits is None or logits.ndim < 1:
                return
            num_experts = int(logits.shape[-1])
            probs = _router_probs(logits).reshape(-1, num_experts)
            if prob_sum[layer_idx] is None:
                prob_sum[layer_idx] = torch.zeros(num_experts, device=probs.device)
            prob_sum[layer_idx] = prob_sum[layer_idx] + probs.sum(dim=0)
            num_rows[layer_idx] += int(probs.shape[0])

            if indices is None and forced_top_k is None:
                return
            if counts[layer_idx] is None:
                counts[layer_idx] = torch.zeros(num_experts, device=probs.device)
            if indices is not None:
                idx_flat = indices.detach().reshape(-1).to(dtype=torch.long, device=probs.device)
                idx_flat = idx_flat[(idx_flat >= 0) & (idx_flat < num_experts)]
                counts[layer_idx] += torch.bincount(idx_flat, minlength=num_experts).to(
                    counts[layer_idx].dtype
                )
            else:
                k = min(int(forced_top_k), num_experts)
                sel = torch.topk(probs, k, dim=-1).indices.reshape(-1)
                counts[layer_idx] += torch.bincount(sel, minlength=num_experts).to(
                    counts[layer_idx].dtype
                )

        return _hook

    hooks: list[torch.utils.hooks.RemovableHandle] = []
    for idx, router in routers.items():
        hooks.append(router.register_forward_hook(_make_hook(idx)))

    try:
        with torch.no_grad():
            for batch in inputs:
                model(batch.to(device))
    finally:
        for handle in hooks:
            handle.remove()

    return counts, prob_sum, num_rows


def _infer_num_experts(model: torch.nn.Module) -> int:
    """Best-effort expert count from ``model.config`` (fallback 1)."""
    config = getattr(model, "config", None)
    for attr in ("num_local_experts", "num_experts", "num_routed_experts"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return 1


def collect_moe_expert_profiles(
    model: torch.nn.Module,
    target_inputs: torch.Tensor | list[torch.Tensor],
    anchor_inputs: torch.Tensor | list[torch.Tensor],
    *,
    beta: float = 1.0,
    eps: float = 1e-8,
    top_k: int | None = None,
    layers: list[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Profile per-expert routing affinity of every MoE layer under two input sets.

    Hooks each MoE layer's router module during forward passes over
    ``target_inputs`` and ``anchor_inputs`` and accumulates, per layer:

    - top-k selection counts (normalized to sum to 1.0) when the router emits
      integer expert indices or a top-k is forced (explicit ``top_k`` or
      ``model.config.num_experts_per_tok``); otherwise
    - token-averaged softmax routing probabilities (summing to 1.0).

    From these, per-expert selective weights are computed:
    ``s_e = clip(p_target[e] / (p_target[e] + beta * p_anchor[e] + eps), 0, 1)``
    so target-dominant experts approach 1.0 (full surgery) and
    anchor-only/shared experts approach 0.0 (left untouched).

    Args:
        model: Transformer model instance (HF or sml-nano).
        target_inputs: Batch(es) of token ids driving the surgery.
        anchor_inputs: Batch(es) of token ids whose routing must be protected.
        beta: Anchor weight in the selective-weight ratio.
        eps: Numerical stability constant.
        top_k: Force top-k count accumulation instead of probability averaging;
            ``None`` falls back to ``model.config.num_experts_per_tok``.
        layers: Layer indices to profile (None means all layers).

    Returns:
        Dict mapping layer index -> dict with ``num_experts``, ``p_target``,
        ``p_anchor``, ``selective_weights`` (float32 tensors of shape (E,)),
        plus the ``beta`` and ``eps`` settings echoed back.
    """
    if not isinstance(target_inputs, (list, tuple)):
        target_inputs = [target_inputs]
    if not isinstance(anchor_inputs, (list, tuple)):
        anchor_inputs = [anchor_inputs]

    layers_module = get_transformer_layers(model)
    if layers is None:
        layer_indices = list(range(len(layers_module)))
    else:
        layer_indices = [int(idx) for idx in layers]

    routers: dict[int, torch.nn.Module] = {}
    for idx in layer_indices:
        components = get_moe_layer_components(layers_module[idx])
        if components["is_moe"] and components["router"] is not None:
            routers[idx] = components["router"]
    if not routers:
        return {}

    config = getattr(model, "config", None)
    default_top_k = getattr(config, "num_experts_per_tok", None) if config is not None else None
    effective_top_k = int(top_k) if top_k is not None else default_top_k

    was_training = model.training
    model.eval()
    try:
        target_counts_by_layer, target_prob_sum, target_rows = _run_router_forward_passes(
            model, routers, list(target_inputs), effective_top_k
        )
        anchor_counts_by_layer, anchor_prob_sum, anchor_rows = _run_router_forward_passes(
            model, routers, list(anchor_inputs), effective_top_k
        )
    finally:
        if was_training:
            model.train()

    profiles: dict[int, dict[str, Any]] = {}
    for idx in routers:
        target_counts = target_counts_by_layer.get(idx)
        anchor_counts = anchor_counts_by_layer.get(idx)
        target_prob = target_prob_sum.get(idx)
        anchor_prob = anchor_prob_sum.get(idx)

        if target_prob is not None:
            num_experts = int(target_prob.shape[0])
        else:
            num_experts = _infer_num_experts(model)
        uniform = torch.full((num_experts,), 1.0 / num_experts)

        if target_counts is not None and float(target_counts.sum()) > 0.0:
            p_target = target_counts / target_counts.sum()
        elif target_prob is not None and target_rows[idx] > 0:
            p_target = target_prob / float(target_rows[idx])
        else:
            p_target = uniform

        if anchor_counts is not None and float(anchor_counts.sum()) > 0.0:
            p_anchor = anchor_counts / anchor_counts.sum()
        elif anchor_prob is not None and anchor_rows[idx] > 0:
            p_anchor = anchor_prob / float(anchor_rows[idx])
        else:
            p_anchor = uniform

        selective = (p_target / (p_target + beta * p_anchor + eps)).clamp(0.0, 1.0)

        profiles[int(idx)] = {
            "num_experts": num_experts,
            "p_target": p_target.detach().to(dtype=torch.float32),
            "p_anchor": p_anchor.detach().to(dtype=torch.float32),
            "selective_weights": selective.detach().to(dtype=torch.float32),
            "beta": float(beta),
            "eps": float(eps),
        }

    return profiles


def _apply_lancet_to_weight(
    weight: torch.Tensor,
    dirs: list[torch.Tensor],
    *,
    scale: float,
    quant_snap: bool,
    quant_snap_iters: int,
    quant_snap_eta: float,
    max_block16_crest_factor: float | None,
) -> torch.Tensor:
    """Run the Tri-Anchor Lancet edit on a single 2D weight (shared by MoE paths)."""
    if quant_snap:
        # H18: Lancet projection plus quantization-aware snapping.
        return project_and_snap_subspace_norm_preserving(
            weight,
            dirs,
            scale=scale,
            num_iters=quant_snap_iters,
            step_size=quant_snap_eta,
            max_block_cf=(
                max_block16_crest_factor if max_block16_crest_factor is not None else 3.2
            ),
            preserve_row_norm=True,
        )
    proj_input = weight
    if max_block16_crest_factor is not None:
        proj_input = regularize_microblock_crest_factor(
            weight, block_size=16, max_block_cf=max_block16_crest_factor
        )
    return project_out_subspace_norm_preserving(
        proj_input, dirs, scale=scale, preserve_row_norm=True
    )


def _apply_tiered_lancet_to_weight(
    weight: torch.Tensor,
    dirs: list[torch.Tensor],
    *,
    scale: float,
    quant_snap: bool,
    quant_snap_iters: int,
    quant_snap_eta: float,
    max_block16_crest_factor: float | None,
    hot: bool,
) -> torch.Tensor:
    """Apply the Strata-tier-aware Lancet edit to a single 2D weight.

    Hot (VRAM-resident) experts and shared experts get FP8/Q8-style row-crest
    hygiene plus a row-norm-preserving projection without 4-bit block-16
    snapping; cold experts keep the quantization-aware subspace snap when
    ``quant_snap`` is on.
    """
    if hot:
        cf = 3.20 if max_block16_crest_factor is None else min(max_block16_crest_factor, 3.20)
        proj_input = weight
        if cf > 0:
            proj_input = regularize_microblock_crest_factor(weight, block_size=16, max_block_cf=cf)
        return project_out_subspace_norm_preserving(
            proj_input, dirs, scale=scale, preserve_row_norm=True
        )
    return _apply_lancet_to_weight(
        weight,
        dirs,
        scale=scale,
        quant_snap=quant_snap,
        quant_snap_iters=quant_snap_iters,
        quant_snap_eta=quant_snap_eta,
        max_block16_crest_factor=max_block16_crest_factor,
    )


def _set_module_weight(module: torch.nn.Module, new_weight: torch.Tensor) -> None:
    """Replace a Linear's weight in-place, keeping ``requires_grad`` semantics."""
    orig = module.weight
    module.weight = torch.nn.Parameter(new_weight, requires_grad=orig.requires_grad)


def _find_expert_down_proj(expert_module: torch.nn.Module) -> torch.nn.Module | None:
    """Return an unpacked expert's down projection (``down_proj`` or ``w2``)."""
    down = getattr(expert_module, "down_proj", None)
    if isinstance(down, torch.nn.Module):
        return down
    w2 = getattr(expert_module, "w2", None)
    if isinstance(w2, torch.nn.Module):
        return w2
    return None


def _snapshot_router_weights(
    routers: dict[int, torch.nn.Module],
) -> dict[int, dict[str, torch.Tensor]]:
    """Clone every router parameter/buffer so post-edit drift can be verified."""
    return {
        idx: {
            name: tensor.detach().clone()
            for name, tensor in router.state_dict().items()
            if torch.is_tensor(tensor)
        }
        for idx, router in routers.items()
    }


def _compute_router_weight_drift(
    routers: dict[int, torch.nn.Module],
    snapshots: dict[int, dict[str, torch.Tensor]],
) -> float:
    """Max absolute change across all router parameters/buffers (0.0 == frozen)."""
    drift = 0.0
    for idx, router in routers.items():
        snapshot = snapshots.get(idx, {})
        with torch.no_grad():
            for name, tensor in router.state_dict().items():
                before = snapshot.get(name)
                if before is None or not torch.is_tensor(tensor):
                    continue
                if tensor.is_floating_point() or tensor.is_complex():
                    diff = (tensor.detach().to(dtype=before.dtype) - before).abs().max()
                    drift = max(drift, float(diff.item()) if diff.numel() else 0.0)
                elif not torch.equal(tensor.detach(), before):
                    drift = float("inf")
    return drift


def apply_moe_lancet_surgery(
    model: torch.nn.Module,
    layer_bases: dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    expert_profiles: dict[int, dict[str, Any]] | None = None,
    *,
    scale: float = 1.0,
    min_selective_weight: float = 0.0,
    min_selectivity_threshold: float = 0.25,
    shared_expert_damping: float = 0.40,
    hot_vram_expert_ids: set[int] | None = None,
    quant_snap: bool | None = None,
    quant_snap_iters: int = 4,
    quant_snap_eta: float = 0.75,
    max_block16_crest_factor: float | None = None,
) -> dict[str, Any]:
    """
    Apply Tri-Anchor Lancet surgery to MoE layers with router freeze guaranteed.

    For every layer with a basis in ``layer_bases`` the full Lancet edit
    (``project_and_snap_subspace_norm_preserving`` when ``quant_snap`` else
    ``project_out_subspace_norm_preserving``, plus the optional micro-block
    crest-factor hygiene pass) is applied with ``scale`` to
    ``self_attn.o_proj.weight`` and the shared expert's ``down_proj.weight``.
    Routed experts additionally receive the same edit scaled by their
    selective weight ``s_e * scale``; experts with ``s_e <=
    min_selective_weight`` (or effectively zero ``s_e``) are skipped so
    anchor-only experts see zero drift.

    Router/gate weights are snapshotted before the edits and verified
    untouched afterwards (``router_weight_drift == 0.0``).

    Args:
        model: MoE transformer model, modified in-place.
        layer_bases: Per-layer orthonormal Tri-Anchor bases (dict layer idx ->
            tensor/list of directions, or a single tensor/list for all layers).
        expert_profiles: Output of :func:`collect_moe_expert_profiles`; when
            ``None`` all routed experts are treated as fully selective (s_e=1).
        scale: Base projection scale for the Lancet edit.
        min_selective_weight: Skip routed experts whose selective weight does
            not exceed this threshold.
        quant_snap: Use quantization-aware subspace snapping (None/False =
            plain projection).
        quant_snap_iters: Snap iterations when ``quant_snap`` is on.
        quant_snap_eta: Leak-reduction step size when ``quant_snap`` is on.
        max_block16_crest_factor: Optional micro-block crest-factor cap.

    Returns:
        Summary dict with ``edited_layers`` (layer indices touched),
        ``edited_shared_experts``, ``edited_routed_experts`` (count),
        ``skipped_routed_experts`` (count), ``router_weight_drift``,
        ``quant_snap`` and ``max_block16_crest_factor``.
    """
    use_quant_snap = bool(quant_snap)
    layers = get_transformer_layers(model)
    dirs_map = _resolve_per_layer_directions(layer_bases, len(layers))

    hot_expert_ids = hot_vram_expert_ids if hot_vram_expert_ids is not None else set()
    total_routed_experts = 0
    edited_layers: list[int] = []
    edited_shared_experts = 0
    edited_routed_experts = 0
    skipped_routed_experts = 0

    components_by_idx: dict[int, dict[str, Any]] = {}
    routers: dict[int, torch.nn.Module] = {}
    for idx in dirs_map:
        components = get_moe_layer_components(layers[idx])
        components_by_idx[idx] = components
        if components["router"] is not None:
            routers[idx] = components["router"]

    router_snapshots = _snapshot_router_weights(routers)

    with torch.no_grad():
        for idx, dirs in dirs_map.items():
            if not dirs:
                continue
            components = components_by_idx[idx]

            # Anchor: attention output projection.
            if components["o_proj"] is not None:
                o_proj = components["o_proj"]
                _set_module_weight(
                    o_proj,
                    _apply_lancet_to_weight(
                        o_proj.weight,
                        dirs,
                        scale=scale,
                        quant_snap=use_quant_snap,
                        quant_snap_iters=quant_snap_iters,
                        quant_snap_eta=quant_snap_eta,
                        max_block16_crest_factor=max_block16_crest_factor,
                    ),
                )

            # Shared expert down projection (None for pure dense layers).
            # Shared experts are always treated as the hot tier and damped.
            shared_down = components["shared_down_proj"]
            if shared_down is not None:
                _set_module_weight(
                    shared_down,
                    _apply_tiered_lancet_to_weight(
                        shared_down.weight,
                        dirs,
                        scale=scale * shared_expert_damping,
                        quant_snap=use_quant_snap,
                        quant_snap_iters=quant_snap_iters,
                        quant_snap_eta=quant_snap_eta,
                        max_block16_crest_factor=max_block16_crest_factor,
                        hot=(hot_vram_expert_ids is not None or shared_expert_damping != 1.0),
                    ),
                )
                edited_shared_experts += 1

            edited_layers.append(int(idx))

            # Routed experts, scaled by their selective weight s_e.
            profile = None if expert_profiles is None else expert_profiles.get(int(idx))
            selective = (
                None if profile is None else profile["selective_weights"].detach().to(torch.float32)
            )

            for expert_idx, expert_module in components["routed_experts_unpacked"]:
                down = _find_expert_down_proj(expert_module)
                if down is None:
                    continue
                s_e = 1.0 if selective is None else float(selective[expert_idx])
                total_routed_experts += 1
                if s_e < max(min_selective_weight, min_selectivity_threshold):
                    skipped_routed_experts += 1
                    continue
                _set_module_weight(
                    down,
                    _apply_tiered_lancet_to_weight(
                        down.weight,
                        dirs,
                        scale=float(s_e) * scale,
                        quant_snap=use_quant_snap,
                        quant_snap_iters=quant_snap_iters,
                        quant_snap_eta=quant_snap_eta,
                        max_block16_crest_factor=max_block16_crest_factor,
                        hot=expert_idx in hot_expert_ids,
                    ),
                )
                edited_routed_experts += 1

            packed = components["routed_experts_packed"]
            if packed is not None:
                num_experts = int(packed.shape[0])
                for expert_idx in range(num_experts):
                    s_e = 1.0 if selective is None else float(selective[expert_idx])
                    total_routed_experts += 1
                    if s_e < max(min_selective_weight, min_selectivity_threshold):
                        skipped_routed_experts += 1
                        continue
                    packed.data[expert_idx] = _apply_tiered_lancet_to_weight(
                        packed.data[expert_idx],
                        dirs,
                        scale=float(s_e) * scale,
                        quant_snap=use_quant_snap,
                        quant_snap_iters=quant_snap_iters,
                        quant_snap_eta=quant_snap_eta,
                        max_block16_crest_factor=max_block16_crest_factor,
                        hot=expert_idx in hot_expert_ids,
                    )
                    edited_routed_experts += 1

    router_weight_drift = _compute_router_weight_drift(routers, router_snapshots)

    return {
        "edited_layers": sorted(set(edited_layers)),
        "edited_shared_experts": edited_shared_experts,
        "edited_routed_experts": edited_routed_experts,
        "skipped_routed_experts": skipped_routed_experts,
        "untouched_expert_ratio": (
            skipped_routed_experts / total_routed_experts if total_routed_experts else 0.0
        ),
        "router_weight_drift": float(router_weight_drift),
        "quant_snap": use_quant_snap,
        "max_block16_crest_factor": max_block16_crest_factor,
    }


# ---------------------------------------------------------------------------
# Streaming Tri-Anchor Lancet for MoE checkpoints (shard-by-shard, no model)
# ---------------------------------------------------------------------------

_MOE_WEIGHT_PREFIX_RE = re.compile(
    r"^(?:model\.|language_model\.model\.|language_model\.|model\.language_model\.|model\.llm\.)?"
    r"layers\.(\d+)\."
)
_MOE_ROUTED_EXPERT_KEY_RE = re.compile(
    r"(?:mlp\.experts|experts|block_sparse_moe\.experts)\.(\d+)\.(?:down_proj|w2)\.weight"
)
_MOE_ROUTED_EXPERT_PACKED_KEY_RE = re.compile(
    r"(?:mlp\.experts|experts)\.(?:down_proj(?:\.weight)?|w2_weight)"
)
_MOE_SHARED_DOWN_PROJ_SUFFIXES = frozenset(
    {
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_experts.down_proj.weight",
        # Dense layers (and Gemma4-style shared MLPs) get the full scale.
        "mlp.down_proj.weight",
    }
)


def _parse_moe_weight_key(key: str, tensor_ndim: int = 2) -> tuple[str, int, int | None] | None:
    """
    Parse a MoE transformer checkpoint weight key into a surgery target.

    Returns ``(kind, layer_idx, expert_idx)`` where ``kind`` is one of:

    - ``"router"``: the layer's router/gate weight (``expert_idx`` None).
    - ``"o_proj"``: ``self_attn.o_proj.weight`` (or ``attn.wo_ud.weight``).
    - ``"shared_down_proj"``: the shared expert's ``down_proj.weight``, or a
      dense layer's ``mlp.down_proj.weight``.
    - ``"routed_expert_unpacked"``: ``experts.<e>.down_proj.weight`` /
      ``experts.<e>.w2.weight`` with ``expert_idx`` set.
    - ``"routed_expert_packed"``: a 3D ``experts.down_proj.weight`` stacking
      all experts, or a 3D ``mlp.experts.w2_weight`` (matched only when
      ``tensor_ndim == 3``).

    Any other key returns ``None``. Recognized prefixes: ``model.``,
    ``language_model.model.``, ``language_model.``, ``model.language_model.``
    (all optional).
    """
    match = _MOE_WEIGHT_PREFIX_RE.match(key)
    if match is None:
        return None
    layer_idx = int(match.group(1))
    if is_router_weight_key(key):
        return ("router", layer_idx, None)
    suffix = key[match.end() :]
    if suffix in ("self_attn.o_proj.weight", "attn.wo_ud.weight"):
        return ("o_proj", layer_idx, None)
    if suffix in _MOE_SHARED_DOWN_PROJ_SUFFIXES:
        return ("shared_down_proj", layer_idx, None)
    expert_match = _MOE_ROUTED_EXPERT_KEY_RE.fullmatch(suffix)
    if expert_match is not None:
        return ("routed_expert_unpacked", layer_idx, int(expert_match.group(1)))
    if tensor_ndim == 3 and _MOE_ROUTED_EXPERT_PACKED_KEY_RE.fullmatch(suffix) is not None:
        return ("routed_expert_packed", layer_idx, None)
    return None


# NVFP4 (E2M1) nibble value lookup: sign(1) | exp(2) | mant(1).
_E2M1_DEQUANT_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _edit_quantized_routed_expert_packed(
    key: str,
    tensor: torch.Tensor,
    shard_tensors: dict[str, torch.Tensor],
    edit_expert_fn: Any,
) -> torch.Tensor | None:
    """
    Best-effort dequant -> per-expert Lancet edit -> requant for a quantized
    packed ``mlp.experts.w2_weight`` tensor.

    Supported layouts:

    - NVFP4: uint8 packed E2M1 nibbles with companion ``w2_weight_scale``
      (``w2_weight_amax`` / ``w2_weight_outer_scale_denominator`` are optional).
    - MXFP8: ``torch.float8_e4m3fn`` with a ``[32, 32]`` block
      ``w2_weight_scale`` (ue8m0).

    Returns the edited tensor, or ``None`` when the companion scale tensors are
    missing, the layout is unrecognized, or anything goes wrong mid-way — the
    caller then leaves the quantized ``w2_weight`` untouched (safe skip).
    """
    base = key[: -len("w2_weight")] if key.endswith("w2_weight") else None
    if base is None:
        return None
    scale = shard_tensors.get(base + "w2_weight_scale")
    if scale is None:
        return None
    try:
        if tensor.dtype == torch.uint8:
            if not hasattr(torch, "float8_e4m3fn"):
                return None
            # NVFP4: low nibble then high nibble per byte, per-group scale
            # broadcast along the last (input) dimension in groups of 32.
            low = (tensor & 0x0F).to(torch.int64)
            high = ((tensor >> 4) & 0x0F).to(torch.int64)
            nibbles = torch.stack((low, high), dim=-1).reshape(*tensor.shape[:-1], -1)
            dequant = _E2M1_DEQUANT_LUT.to(tensor.device)[nibbles].to(torch.float32)
            group = 32
            if dequant.shape[-1] % group != 0 or scale.shape != (
                *tensor.shape[:-1],
                tensor.shape[-1] * 2 // group,
            ):
                return None
            scale_f = scale.to(torch.float32).repeat_interleave(group, dim=-1)
            dequant = dequant * scale_f
        elif tensor.dtype == getattr(torch, "float8_e4m3fn", None):
            # MXFP8: [32, 32] block scale (ue8m0) broadcast over the weight.
            dequant = tensor.to(torch.float32)
            if scale.shape != dequant.shape:
                return None
            dequant = dequant * scale.to(torch.float32)
        else:
            return None

        edited = torch.empty_like(dequant)
        any_edited = False
        for e_idx in range(dequant.shape[0]):
            result = edit_expert_fn(dequant[e_idx], e_idx)
            if result is None:
                edited[e_idx] = dequant[e_idx]
            else:
                edited[e_idx] = result.to(torch.float32)
                any_edited = True
        if not any_edited:
            return None

        # Requantize back into the original packed layout.
        if tensor.dtype == torch.uint8:
            quantized = dequant / scale_f
            quantized = quantized.clamp(-6.0, 6.0)
            codes = (
                torch.where(
                    quantized >= 0,
                    torch.round(torch.log2(quantized.clamp_min(0.5) * 2)) + 1,
                    torch.round(torch.log2((-quantized).clamp_min(0.5) * 2)) + 9,
                )
                .clamp(0, 15)
                .to(torch.uint8)
            )
            out = (codes[..., 0::2] & 0x0F) | (codes[..., 1::2] << 4)
            return out.contiguous()
        return (edited / scale.to(torch.float32)).to(tensor.dtype)
    except Exception:
        return None


def apply_moe_lancet_recipe_streaming(
    model_dir: str | Path,
    output_dir: str | Path,
    layer_bases: dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    expert_profiles: dict[int, dict[str, Any]] | None = None,
    *,
    scale: float = 1.0,
    min_selective_weight: float = 0.0,
    min_selectivity_threshold: float = 0.25,
    shared_expert_damping: float = 0.40,
    hot_vram_expert_ids: set[int] | None = None,
    quant_snap: bool | None = None,
    quant_snap_iters: int = 4,
    quant_snap_eta: float = 0.75,
    max_block16_crest_factor: float | None = None,
) -> dict[str, Any]:
    """
    Apply the Tri-Anchor Lancet MoE edit to a sharded safetensors checkpoint.

    Streaming counterpart of :func:`apply_moe_lancet_surgery`: like
    :func:`apply_lancet_recipe_streaming` it loads one shard at a time on CPU,
    edits the target weights, and writes the shard to ``output_dir`` before the
    next shard is loaded, so the full model is never materialized.

    Per weight key (parsed via :func:`_parse_moe_weight_key`):

    - ``"router"``: copied through bit-for-bit (drift verified as 0.0).
    - ``"o_proj"`` / ``"shared_down_proj"``: full-scale Lancet edit for layers
      with a basis in ``layer_bases``.
    - ``"routed_expert_unpacked"`` / ``"routed_expert_packed"``: per-expert
      Lancet edit at ``scale * s_e``; experts with ``s_e <=
      min_selective_weight`` (or effectively zero) are skipped untouched.

    Args:
        model_dir: Directory holding the source safetensors checkpoint.
        output_dir: Directory to create for the modified checkpoint.
        layer_bases: Per-layer Tri-Anchor bases (dict layer idx -> vectors,
            tensor/list for all layers), resolved as in
            :func:`_resolve_per_layer_directions`.
        expert_profiles: Output of :func:`collect_moe_expert_profiles`; when
            ``None`` every routed expert is treated as fully selective (s_e=1).
        scale: Base projection scale for the Lancet edit.
        min_selective_weight: Skip routed experts whose selective weight does
            not exceed this threshold.
        quant_snap: Use quantization-aware subspace snapping (None/False =
            plain projection).
        quant_snap_iters: Snap iterations when ``quant_snap`` is on.
        quant_snap_eta: Leak-reduction step size when ``quant_snap`` is on.
        max_block16_crest_factor: Optional micro-block crest-factor cap.

    Returns:
        Dict with ``edited_layers``, ``edited_o_proj``,
        ``edited_shared_experts``, ``edited_routed_experts``,
        ``skipped_routed_experts``, ``router_weight_drift`` (0.0),
        ``max_row_crest_factor``, ``max_block16_crest_factor``,
        ``peak_memory_mb`` and ``output_dir``.
    """
    use_quant_snap = bool(quant_snap)
    src_dir = Path(model_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    shards, index_data = _resolve_safetensors_shards(src_dir)
    same_dir = src_dir.resolve() == dst_dir.resolve()

    # Resolve layer_bases without a model instance (no known layer count).
    dirs_map: dict[int, list[torch.Tensor]] = {}
    uniform_dirs: list[torch.Tensor] | None = None
    if isinstance(layer_bases, dict):
        dirs_map = {int(idx): _as_direction_list(v) for idx, v in layer_bases.items()}
    elif isinstance(layer_bases, torch.Tensor):
        if layer_bases.ndim == 1:
            uniform_dirs = [layer_bases]
        else:
            dirs_map = {i: [layer_bases[i]] for i in range(layer_bases.shape[0])}
    else:
        dirs_map = {i: _as_direction_list(entry) for i, entry in enumerate(layer_bases)}

    edited_layers: list[int] = []
    edited_o_proj = 0
    edited_shared_experts = 0
    edited_routed_experts = 0
    skipped_routed_experts = 0
    total_routed_experts = 0
    hot_expert_ids = hot_vram_expert_ids if hot_vram_expert_ids is not None else set()
    router_weight_drift = 0.0
    max_row_crest_factor = 1.0

    tracemalloc.start()
    try:
        for shard_name in shards:
            shard_path = src_dir / shard_name
            shard_tensors = load_file(str(shard_path))
            router_originals: dict[str, torch.Tensor] = {}

            for key, tensor in shard_tensors.items():
                parsed = _parse_moe_weight_key(key, tensor.ndim)
                if parsed is None:
                    continue
                kind, layer_idx, expert_idx = parsed

                if kind == "router":
                    # Routers are frozen: snapshot and copy through untouched.
                    router_originals[key] = tensor.detach().clone()
                    continue

                dirs = uniform_dirs if uniform_dirs is not None else dirs_map.get(layer_idx)
                if not dirs:
                    continue

                if kind in ("o_proj", "shared_down_proj"):
                    if kind == "shared_down_proj":
                        # Shared experts: hot tier, damped scale, no 4-bit snap.
                        shard_tensors[key] = _apply_tiered_lancet_to_weight(
                            tensor,
                            dirs,
                            scale=scale * shared_expert_damping,
                            quant_snap=use_quant_snap,
                            quant_snap_iters=quant_snap_iters,
                            quant_snap_eta=quant_snap_eta,
                            max_block16_crest_factor=max_block16_crest_factor,
                            hot=(hot_vram_expert_ids is not None or shared_expert_damping != 1.0),
                        )
                    else:
                        shard_tensors[key] = _apply_lancet_to_weight(
                            tensor,
                            dirs,
                            scale=scale,
                            quant_snap=use_quant_snap,
                            quant_snap_iters=quant_snap_iters,
                            quant_snap_eta=quant_snap_eta,
                            max_block16_crest_factor=max_block16_crest_factor,
                        )
                    if kind == "o_proj":
                        edited_o_proj += 1
                    else:
                        edited_shared_experts += 1
                    edited_layers.append(layer_idx)
                    continue

                profile = None if expert_profiles is None else expert_profiles.get(layer_idx)
                selective = (
                    None
                    if profile is None
                    else profile["selective_weights"].detach().to(dtype=torch.float32)
                )

                if kind == "routed_expert_unpacked":
                    s_e = 1.0 if selective is None else float(selective[expert_idx])
                    total_routed_experts += 1
                    if s_e < max(min_selective_weight, min_selectivity_threshold):
                        skipped_routed_experts += 1
                        continue
                    shard_tensors[key] = _apply_tiered_lancet_to_weight(
                        tensor,
                        dirs,
                        scale=scale * s_e,
                        quant_snap=use_quant_snap,
                        quant_snap_iters=quant_snap_iters,
                        quant_snap_eta=quant_snap_eta,
                        max_block16_crest_factor=max_block16_crest_factor,
                        hot=expert_idx in hot_expert_ids,
                    )
                    edited_routed_experts += 1
                    edited_layers.append(layer_idx)
                    continue

                # kind == "routed_expert_packed": 3D (E, out, in) stacked experts.
                if tensor.dtype not in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                ):
                    # Quantized packed experts (NVFP4 uint8 nibbles / MXFP8
                    # float8_e4m3fn): attempt safe dequant -> edit -> requant
                    # and skip untouched when the layout is unsupported, so the
                    # pass never crashes and pure bfloat16 tensors are still
                    # edited.
                    def _edit_packed_expert(
                        expert_tensor: torch.Tensor,
                        e_idx: int,
                        _selective: Any = selective,
                        _dirs: Any = dirs,
                    ) -> torch.Tensor | None:
                        s_e = 1.0 if _selective is None else float(_selective[e_idx])
                        if s_e < max(min_selective_weight, min_selectivity_threshold):
                            return None
                        return _apply_tiered_lancet_to_weight(
                            expert_tensor,
                            _dirs,
                            scale=scale * s_e,
                            quant_snap=use_quant_snap,
                            quant_snap_iters=quant_snap_iters,
                            quant_snap_eta=quant_snap_eta,
                            max_block16_crest_factor=max_block16_crest_factor,
                            hot=e_idx in hot_expert_ids,
                        )

                    replacement = _edit_quantized_routed_expert_packed(
                        key, tensor, shard_tensors, _edit_packed_expert
                    )
                    if replacement is not None:
                        shard_tensors[key] = replacement
                        edited_layers.append(layer_idx)
                    continue

                any_expert_edited = False
                orig_expert_device = tensor.device
                work_tensor = (
                    tensor.to(device="cuda")
                    if orig_expert_device.type == "cpu" and torch.cuda.is_available()
                    else tensor
                )
                for e_idx in range(int(work_tensor.shape[0])):
                    s_e = 1.0 if selective is None else float(selective[e_idx])
                    total_routed_experts += 1
                    if s_e < max(min_selective_weight, min_selectivity_threshold):
                        skipped_routed_experts += 1
                        continue
                    work_tensor[e_idx] = _apply_tiered_lancet_to_weight(
                        work_tensor[e_idx],
                        dirs,
                        scale=scale * s_e,
                        quant_snap=use_quant_snap,
                        quant_snap_iters=quant_snap_iters,
                        quant_snap_eta=quant_snap_eta,
                        max_block16_crest_factor=max_block16_crest_factor,
                        hot=e_idx in hot_expert_ids,
                    )
                    edited_routed_experts += 1
                    any_expert_edited = True
                if any_expert_edited:
                    if work_tensor.device != orig_expert_device:
                        shard_tensors[key] = work_tensor.to(device=orig_expert_device)
                        del work_tensor
                    edited_layers.append(layer_idx)

            # Verify routers survived the shard pass bit-for-bit.
            for router_key, before in router_originals.items():
                current = shard_tensors[router_key]
                if current.is_floating_point() or current.is_complex():
                    diff = (current.detach().to(dtype=before.dtype) - before).abs().max()
                    router_weight_drift = max(
                        router_weight_drift, float(diff.item()) if diff.numel() else 0.0
                    )
                elif not torch.equal(current.detach(), before):
                    router_weight_drift = float("inf")

            for shard_tensor in shard_tensors.values():
                if shard_tensor.ndim == 2 and shard_tensor.dtype in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                ):
                    crest = _compute_row_crest_factor(shard_tensor)
                    max_row_crest_factor = max(max_row_crest_factor, crest)

            save_file(shard_tensors, str(dst_dir / shard_name))
            del shard_tensors, router_originals
            gc.collect()
    finally:
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    if not same_dir:
        _copy_aux_files(src_dir, dst_dir, copy_index=index_data is not None)

    return {
        "edited_layers": sorted(set(edited_layers)),
        "edited_o_proj": edited_o_proj,
        "edited_shared_experts": edited_shared_experts,
        "edited_routed_experts": edited_routed_experts,
        "skipped_routed_experts": skipped_routed_experts,
        "untouched_expert_ratio": (
            skipped_routed_experts / total_routed_experts if total_routed_experts else 0.0
        ),
        "router_weight_drift": float(router_weight_drift),
        "max_row_crest_factor": float(max_row_crest_factor),
        "max_block16_crest_factor": max_block16_crest_factor,
        "peak_memory_mb": float(peak_memory_bytes) / (1024**2),
        "quant_snap": use_quant_snap,
        "output_dir": str(dst_dir),
    }
