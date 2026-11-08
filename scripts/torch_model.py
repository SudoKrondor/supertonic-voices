"""Supertonic's four ONNX graphs converted to PyTorch modules via onnx2torch.

The graphs are exported at opset 19, but onnx2torch only registers converters up to
~opset 17/18. For the ops these graphs use, the opset-19 revisions only add things we
don't use (float8 casts, new pad modes), so we fall back to the newest converter at or
below the requested version. Step 2 checks the result against ONNX Runtime.
"""

import onnx2torch.converter as _conv
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, get_converter as _get_converter

import torch

from common import ONNX

GRAPHS = ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]


def _get_converter_with_fallback(operation_type, version, domain=""):
    try:
        return _get_converter(operation_type, version, domain)
    except NotImplementedError:
        known = [d.version for d in _CONVERTER_REGISTRY
                 if d.operation_type == operation_type and d.domain == domain and d.version <= version]
        if not known:
            raise
        return _get_converter(operation_type, max(known), domain)


_conv.get_converter = _get_converter_with_fallback


def _clip_converter(node, graph):
    # onnx2torch treats an omitted optional input ("") as a missing constant and fails.
    # ONNX uses "" for "no bound", so map it to None like an absent input.
    from onnx2torch.node_converters.clip import _create_torch_module
    from onnx2torch.utils.common import OnnxMapping, OperationConverterResult, get_const_value

    def bound(i):
        name = node.input_values[i] if len(node.input_values) > i else ""
        return float(get_const_value(name, graph)) if name else None

    return OperationConverterResult(
        torch_module=_create_torch_module(min_val=bound(1), max_val=bound(2)),
        onnx_mapping=OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values),
    )


for _d in [d for d in _CONVERTER_REGISTRY if d.operation_type == "Clip" and d.version >= 11]:
    _CONVERTER_REGISTRY[_d] = _clip_converter


def unit_rows(style: torch.Tensor) -> torch.Tensor:
    """Every row of every preset style (50x256 ttl, 8x16 dp) has length exactly 1.
    The model never saw anything else, so fitted styles are kept on that shape too."""
    return torch.nn.functional.normalize(style, dim=-1)


def synthesize_latent(m, ids, tmask, ttl, noise, lmask, total_steps: int = 5) -> torch.Tensor:
    """Text + timbre style + starting noise -> final latent, differentiable in ttl.
    The noise length (from a known duration) fixes how long the utterance is."""
    emb = m["text_encoder"](ids, ttl, tmask)
    x = noise
    total = torch.tensor([total_steps], dtype=torch.float32, device=noise.device)
    for step in range(total_steps):
        cur = torch.tensor([step], dtype=torch.float32, device=noise.device)
        x = m["vector_estimator"](x, emb, ttl, lmask, tmask, cur, total)
    return x


def load(device: str = "cuda") -> dict[str, torch.nn.Module]:
    """Return the four graphs as frozen PyTorch modules in eval mode."""
    mods = {}
    for name in GRAPHS:
        m = _conv.convert(str(ONNX / f"{name}.onnx")).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        mods[name] = m
    return mods
