"""Linear CKA computation and layer activation collection utilities."""

from __future__ import annotations

import re
from typing import Any

import torch


def get_transformer_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """
    Return the transformer layer stack for a model.

    Supports Hugging Face causal LM layouts (`model.model.layers`), conditional
    LM wrappers (`model.language_model.model.layers`,
    `model.language_model.layers`, `model.model.language_model.layers`, used by
    Gemma 4 / Mistral 3.1 ``*ForConditionalGeneration`` classes) and
    sml-nano layouts (`model.layers`).

    Args:
        model: Transformer model instance (HF or sml-nano)

    Returns:
        ModuleList of transformer layers
    """
    if hasattr(model, "language_model"):
        if hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
            return model.language_model.model.layers
        if hasattr(model.language_model, "layers"):
            return model.language_model.layers
    if hasattr(model, "model"):
        if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
            return model.model.language_model.layers
        if hasattr(model.model, "layers"):
            return model.model.layers
    if hasattr(model, "llm") and hasattr(model.llm, "layers"):
        return model.llm.layers
    return model.layers


_ROUTER_WEIGHT_KEY_RE = re.compile(
    r"\.(?:"
    r"(?:mlp\.)?(?:gate|router|routing_weights)(?:\.proj)?"
    r"|block_sparse_moe\.gate"
    r")\.(?:weight|bias|e_score_correction_bias)$"
)

# Structural controller/memory tables that must be hard-frozen (exactly 0.0
# drift) during surgery: Engram N-gram memory tables, 4-branch Gated
# Residual / iHC hyper-connections, and Looped MoE loop controllers. Any
# key containing one of these tokens in a name segment is frozen whole.
_FROZEN_STRUCTURE_TOKENS = (
    "engram",
    "ngram_table",
    "memory_table",
    "gated_residual",
    "residual_gate",
    "ihc",
    "hyper_conn",
    "loop_controller",
    "loop_gate",
    "tick_embed",
    "l2_clip",
    "attn_res",
)


def _is_frozen_structure_key(key: str) -> bool:
    """Return True when ``key`` names a frozen structural controller/memory table."""
    return any(
        token in segment for segment in key.lower().split(".") for token in _FROZEN_STRUCTURE_TOKENS
    )


def is_router_weight_key(key: str) -> bool:
    """
    Return True when ``key`` names a router/gate parameter of a MoE layer
    or a frozen structural controller/memory table.

    Matches routing weights and biases such as ``...mlp.gate.weight``,
    ``...mlp.router.weight`` / ``...mlp.router.bias``,
    ``...mlp.routing_weights.weight``, ``...mlp.router.proj.weight``
    (wrapped-router layouts), ``...router.weight`` /
    ``...router.proj.weight`` and ``...block_sparse_moe.gate.weight``
    (Mixtral-style), plus DeepSeek-style ``...gate.e_score_correction_bias``
    correction biases.

    Also hard-freezes Engram N-gram memory tables (``engram``,
    ``ngram_table``, ``memory_table``), 4-branch Gated Residual / iHC
    hyper-connections (``gated_residual``, ``residual_gate``, ``ihc``,
    ``hyper_conn``) and Looped MoE loop controllers (``loop_controller``,
    ``loop_gate``, ``tick_embed``) wherever they appear in the key.

    Per-expert ``gate_proj`` / ``gate_up_proj`` weights are *not* router
    parameters and are never matched.
    """
    if key.endswith(("mlp.gate.weight", "mlp.gate.bias")):
        return True
    return _ROUTER_WEIGHT_KEY_RE.search(key) is not None or _is_frozen_structure_key(key)


def _resolve_module_path(root: Any, dotted_path: str) -> Any:
    """Walk a dotted attribute path with ``getattr``; None when any part is missing."""
    obj = root
    for part in dotted_path.split("."):
        if obj is None or not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _find_experts_container(layer: torch.nn.Module) -> Any:
    """Locate the routed-experts container of a layer, if any."""
    for path in ("mlp.experts", "experts", "block_sparse_moe.experts"):
        container = _resolve_module_path(layer, path)
        if container is not None:
            return container
    return None


def get_moe_layer_components(layer: torch.nn.Module) -> dict[str, Any]:
    """
    Introspect one transformer layer for its MoE surgery-relevant components.

    Handles the common MoE layouts: Llama/Qwen-style ``mlp.gate`` /
    ``mlp.router`` with ``shared_expert``, DeepSeek-style ``mlp.gate``,
    Mixtral-style ``block_sparse_moe.gate`` with ``w1/w2/w3`` experts, and
    packed (3D ``experts.down_proj``) expert tensors.

    Args:
        layer: A single transformer block module.

    Returns:
        Dict with keys:

        - ``o_proj``: ``layer.self_attn.o_proj`` when present, else the
          first hybrid attention output projection (Gated DeltaNet / QSA /
          IndexPool / SWA) when the layer has no dense ``o_proj``, else
          ``None``.
        - ``hybrid_attn_out_projs``: list of hybrid linear/sparse attention
          output projection modules (``linear_attn.out_proj``,
          ``gdn.out_proj``, ``deltanet.o_proj``, ``qsa.o_proj``,
          ``index_pool.o_proj``, ``swa.o_proj``); empty for pure dense
          attention layers.
        - ``latent_experts``: the ``LatentMoE`` module
          (``latent_experts``) when present, else a dict of the
          ``latent_down_proj`` / ``latent_up_proj`` projections when only
          those exist, else ``None``.
        - ``drafter_heads``: dict of embedded speculative drafter / MTP
          head submodules (``dspark``, ``dfly2``, ``drafter``,
          ``mtp_heads``, ``eh_proj``) keyed by dotted module name; empty
          when the layer carries none.
        - ``router``: the router/gate module (the wrapper itself when the
          router keeps its Linear in a ``.proj`` submodule, so callers can
          hook the wrapper's forward).
        - ``shared_down_proj``: the shared expert's ``down_proj`` (or
          ``mlp.down_proj`` when routed experts coexist with a fused shared
          MLP); ``None`` for pure dense layers without routed experts.
        - ``routed_experts_unpacked``: list of ``(idx, expert_module)``
          tuples for ``ModuleList`` expert containers; each expert module
          exposes ``down_proj`` or ``w2``.
        - ``routed_experts_packed``: 3D ``(E, out_features, in_features)``
          ``experts.down_proj`` parameter/tensor when experts are packed,
          else ``None``.
        - ``is_moe``: True when the layer has a router, routed experts or
          latent experts.
    """
    o_proj = _resolve_module_path(layer, "self_attn.o_proj")

    # Hybrid linear/sparse attention output projections (Gated DeltaNet,
    # QSA, IndexPool/SWA layouts). Dense o_proj takes precedence; when it
    # is absent, fall back to the first hybrid projection found.
    hybrid_attn_out_projs = [
        module
        for module in (
            _resolve_module_path(layer, path)
            for path in (
                "self_attn.linear_attn.out_proj",
                "linear_attn.out_proj",
                "gdn.out_proj",
                "deltanet.o_proj",
                "qsa.o_proj",
                "index_pool.o_proj",
                "swa.o_proj",
                "attn.wo_ud",
            )
        )
        if module is not None
    ]
    if o_proj is None and hybrid_attn_out_projs:
        o_proj = hybrid_attn_out_projs[0]

    router = None
    for path in ("mlp.gate", "mlp.router", "router", "block_sparse_moe.gate"):
        candidate = _resolve_module_path(layer, path)
        if candidate is not None:
            # Wrapped routers keep the Linear in `.proj`; return the wrapper so
            # hooks see the full router forward. `is_router_weight_key` covers
            # both `.router.weight` and `.router.proj.weight` key layouts.
            router = candidate
            break

    experts_container = _find_experts_container(layer)

    routed_experts_unpacked: list[tuple[int, torch.nn.Module]] = []
    if experts_container is not None and isinstance(experts_container, torch.nn.ModuleList):
        routed_experts_unpacked = [(i, expert) for i, expert in enumerate(experts_container)]

    routed_experts_packed = None
    if experts_container is not None:
        packed = getattr(experts_container, "down_proj", None)
        if packed is None:
            packed = getattr(experts_container, "w2_weight", None)
        if isinstance(packed, (torch.nn.Parameter, torch.Tensor)) and packed.ndim == 3:
            routed_experts_packed = packed

    has_routed_experts = bool(routed_experts_unpacked) or routed_experts_packed is not None

    shared_down_proj = None
    for path in ("mlp.shared_expert.down_proj", "mlp.shared_experts.down_proj"):
        candidate = _resolve_module_path(layer, path)
        if candidate is not None:
            shared_down_proj = candidate
            break
    if shared_down_proj is None and has_routed_experts:
        mlp_down = _resolve_module_path(layer, "mlp.down_proj")
        if mlp_down is not None:
            shared_down_proj = mlp_down

    # LatentMoE: dedicated latent expert pool with its own down/up projections.
    latent_experts = None
    for path in ("mlp.latent_experts", "latent_experts"):
        candidate = _resolve_module_path(layer, path)
        if candidate is not None:
            latent_experts = candidate
            break
    if latent_experts is None:
        latent_projs: dict[str, Any] = {}
        for path in ("mlp.latent_down_proj", "latent_down_proj"):
            candidate = _resolve_module_path(layer, path)
            if candidate is not None:
                latent_projs["latent_down_proj"] = candidate
                break
        for path in ("mlp.latent_up_proj", "latent_up_proj"):
            candidate = _resolve_module_path(layer, path)
            if candidate is not None:
                latent_projs["latent_up_proj"] = candidate
                break
        if latent_projs:
            latent_experts = latent_projs

    # Embedded speculative drafters / MTP heads (dspark, dfly2, drafter,
    # mtp_heads, eh_proj) wherever they appear under this layer.
    drafter_tokens = ("dspark", "dfly2", "drafter", "mtp_heads", "eh_proj")
    drafter_heads: dict[str, torch.nn.Module] = {}
    for name, module in layer.named_modules():
        if not name:
            continue
        if any(token in segment for segment in name.lower().split(".") for token in drafter_tokens):
            drafter_heads[name] = module

    is_moe = router is not None or has_routed_experts or latent_experts is not None

    return {
        "o_proj": o_proj,
        "hybrid_attn_out_projs": hybrid_attn_out_projs,
        "router": router,
        "shared_down_proj": shared_down_proj,
        "routed_experts_unpacked": routed_experts_unpacked,
        "routed_experts_packed": routed_experts_packed,
        "latent_experts": latent_experts,
        "drafter_heads": drafter_heads,
        "is_moe": is_moe,
    }


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Compute Linear CKA similarity between two activation matrices.

    Args:
        x: Tensor of shape (B*T, D) or (B, T, D)
        y: Tensor of shape (B*T, D) or (B, T, D)
        eps: Small constant for numerical stability

    Returns:
        float: Linear CKA similarity value in [0.0, 1.0]
    """
    # Flatten leading dimensions if 3D
    if x.ndim == 3:
        x = x.view(-1, x.shape[-1]).to(torch.float32)
    if y.ndim == 3:
        y = y.view(-1, y.shape[-1]).to(torch.float32)

    # Center features column-wise
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    # Compute linear CKA via Frobenius norm identity
    xx = torch.matmul(x.t(), x)
    yy = torch.matmul(y.t(), y)
    xy = torch.matmul(x.t(), y)

    numerator = torch.trace(torch.matmul(xy.t(), xy))
    denominator = torch.trace(torch.matmul(xx.t(), xx)) * torch.trace(torch.matmul(yy.t(), yy))

    cka = numerator / torch.sqrt(denominator + eps)
    return float(torch.clamp(cka, 0.0, 1.0))


def collect_layer_activations(
    model: torch.nn.Module, input_ids: torch.Tensor
) -> list[torch.Tensor]:
    """
    Collect activations from each layer of the transformer model.

    Args:
        model: Transformer model instance (HF or sml-nano)
        input_ids: Input tensor of shape (B, T)

    Returns:
        List of activation tensors, one per layer, each of shape (B*T, D)
    """
    activations = []

    def hook_fn(module, input, output):
        # TransformerBlock forward returns (x, present_kv), so we take output[0]
        if isinstance(output, tuple):
            activations.append(output[0].detach().view(-1, output[0].shape[-1]).to(torch.float32))
        else:
            activations.append(output.detach().view(-1, output.shape[-1]).to(torch.float32))

    # Register forward hooks on all layers
    hooks = []
    for layer in get_transformer_layers(model):
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Run model forward pass
    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations


def compute_layerwise_cka_drift(
    base_model: torch.nn.Module,
    adapted_model: torch.nn.Module,
    input_ids: torch.Tensor | None = None,
    *,
    prompt_inputs: list[torch.Tensor] | None = None,
) -> list[dict[str, float | int]]:
    """
    Compute per-block CKA similarity and drift between `base_model` and `adapted_model`.

    Supports two mutually exclusive input modes:
    - `input_ids`: single batched tensor of shape (B, T) (backwards-compatible path).
    - `prompt_inputs`: list of per-prompt tensors, each of shape (1, seq_len).
      Forward passes run per prompt without padding; per-layer residual deltas
      (out - in) are flattened to [tokens, hidden] and concatenated across
      prompts before computing CKA.
    """
    if prompt_inputs is not None:
        base_deltas = _collect_prompt_deltas(base_model, prompt_inputs)
        adapted_deltas = _collect_prompt_deltas(adapted_model, prompt_inputs)
    else:
        if input_ids is None:
            raise ValueError("Either `input_ids` or `prompt_inputs` must be provided.")
        base_deltas = _collect_deltas(base_model, input_ids)
        adapted_deltas = _collect_deltas(adapted_model, input_ids)

    results: list[dict[str, float | int]] = []
    for i, (base_d, adapted_d) in enumerate(zip(base_deltas, adapted_deltas, strict=False)):
        sim = linear_cka(base_d, adapted_d)
        results.append(
            {"layer_idx": i, "cka_similarity": round(sim, 6), "cka_drift": round(1.0 - sim, 6)}
        )

    return results


def _make_delta_hook(store: list[torch.Tensor]):
    """Build a forward hook that records per-layer residual deltas (out - in)."""

    def _hook(module, inputs, output):
        out_t = output[0] if isinstance(output, tuple) else output
        in_t = inputs[0]
        delta = (out_t - in_t).detach().view(-1, out_t.shape[-1]).to(torch.float32)
        store.append(delta)

    return _hook


def _collect_deltas(model: torch.nn.Module, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Run a single forward pass and return per-layer residual deltas, one per layer."""
    device = next(model.parameters()).device
    deltas: list[torch.Tensor] = []

    hooks = [
        layer.register_forward_hook(_make_delta_hook(deltas))
        for layer in get_transformer_layers(model)
    ]
    try:
        with torch.no_grad():
            model(input_ids.to(device))
    finally:
        for h in hooks:
            h.remove()

    return deltas


def _collect_prompt_deltas(
    model: torch.nn.Module, prompt_inputs: list[torch.Tensor]
) -> list[torch.Tensor]:
    """
    Run forward passes over each prompt (no padding) and return per-layer residual
    deltas concatenated across prompts, each of shape [tokens, hidden].
    """
    device = next(model.parameters()).device
    layers = get_transformer_layers(model)
    per_layer: list[list[torch.Tensor]] = [[] for _ in layers]

    def make_prompt_delta_hook(layer_idx: int):
        def _hook(module, inputs, output):
            out_t = output[0] if isinstance(output, tuple) else output
            in_t = inputs[0]
            delta = (out_t - in_t).detach().view(-1, out_t.shape[-1]).to(torch.float32)
            per_layer[layer_idx].append(delta)

        return _hook

    # Register hooks once; each prompt's forward pass appends one chunk per layer.
    hooks = [
        layer.register_forward_hook(make_prompt_delta_hook(i)) for i, layer in enumerate(layers)
    ]
    try:
        with torch.no_grad():
            for prompt in prompt_inputs:
                model(prompt.to(device))
    finally:
        for h in hooks:
            h.remove()

    return [torch.cat(chunks, dim=0) for chunks in per_layer]


def compute_blockwise_quantization_drift(
    model: torch.nn.Module,
    prompt_inputs: list[torch.Tensor],
    quant_bits: int = 4,
    group_size: int = 32,
) -> list[dict[str, float | int]]:
    """
    Compute per-layer leave-one-layer-quantized forward-KL drift (the H7 metric).

    For each transformer layer, all `torch.nn.Linear` weights are quantized in-place
    with group-wise fake quantization while every other layer keeps its original
    weights. The token-averaged forward KL divergence between the clean model's
    output distribution and the leave-one-out quantized model's distribution is
    averaged over a small set of calibration prompts. Original weights are always
    restored in a `finally` block.
    """
    # Imported lazily: `sml.surgery.quantization` imports `get_transformer_layers`
    # from this module, so a module-level import here would be circular.
    from sml.surgery.quantization import quantize_tensor_groupwise

    device = next(model.parameters()).device
    layers = get_transformer_layers(model)
    cal_inputs = [p.to(device) for p in prompt_inputs[:8]]

    def _logits(m: torch.nn.Module, p: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return float32 logits."""
        try:
            return m(p, use_cache=False).logits.float()
        except TypeError:
            return m(p).logits.float()

    with torch.no_grad():
        ref_log_probs = [
            torch.nn.functional.log_softmax(_logits(model, p), dim=-1) for p in cal_inputs
        ]
        ref_probs = [lp.exp() for lp in ref_log_probs]

    results: list[dict[str, float | int]] = []
    for i, layer in enumerate(layers):
        linears = [m for m in layer.modules() if isinstance(m, torch.nn.Linear)]
        originals = {id(m): m.weight.data.clone() for m in linears}
        try:
            for m in linears:
                m.weight.data = quantize_tensor_groupwise(
                    originals[id(m)], bits=quant_bits, group_size=group_size
                )
            kl_per_prompt: list[torch.Tensor] = []
            with torch.no_grad():
                for j, p in enumerate(cal_inputs):
                    quant_log_probs = torch.nn.functional.log_softmax(_logits(model, p), dim=-1)
                    kl = (
                        (ref_probs[j] * (ref_log_probs[j] - quant_log_probs))
                        .sum(dim=-1)
                        .mean()
                        .clamp_min(0.0)
                    )
                    kl_per_prompt.append(kl)
            mean_kl = float(torch.stack(kl_per_prompt).mean()) if kl_per_prompt else 0.0
        finally:
            for m in linears:
                m.weight.data = originals[id(m)]

        results.append(
            {
                "layer_idx": i,
                "leave_one_out_kl": round(mean_kl, 6),
                "cka_drift": round(mean_kl, 6),
            }
        )

    return results
