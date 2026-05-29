from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bz2
import numpy as np

from .container import encode_metadata
from .deploy_registry import DEPLOYABLE_PIPELINES
from .feature_extraction import build_feature_dict
from .reversible import CompressionArtifact, compress_with_pipeline, typed_array
from .runtime import ComponentUnavailable, PipelineContext, ROOT_DIR, canonical_name, canonical_pipeline_name, normalize_context

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


SELECTOR_MODE_CODES = {
    "exhaustive": 1,
    "model": 2,
    "hybrid": 3,
}


def deployable_pipeline_names() -> list[str]:
    return [row.name for row in DEPLOYABLE_PIPELINES]


def artifact_total_size(artifact: CompressionArtifact) -> int:
    return len(artifact.payload) + len(artifact.tail) + len(encode_metadata(artifact.metadata))


@dataclass(frozen=True)
class SelectionResult:
    selector_mode: str
    artifact: CompressionArtifact
    ranking: list[dict[str, Any]]
    fallback_used: bool = False


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts.astype(np.float64) / total
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def _byte_hist(raw: bytes) -> np.ndarray:
    if not raw:
        return np.zeros(256, dtype=np.float32)
    return np.bincount(np.frombuffer(raw, dtype=np.uint8), minlength=256).astype(np.float32)


def _window_entropies(raw: bytes, window: int = 4096) -> np.ndarray:
    if not raw:
        return np.zeros(1, dtype=np.float32)
    values = []
    for start in range(0, len(raw), window):
        chunk = raw[start:start + window]
        values.append(_entropy_from_counts(_byte_hist(chunk)))
    return np.array(values, dtype=np.float32)


def _byte_delta(raw: bytes) -> bytes:
    if len(raw) <= 1:
        return raw
    arr = np.frombuffer(raw, dtype=np.uint8)
    out = arr.copy()
    out[1:] = arr[1:] - arr[:-1]
    return out.tobytes()


def _byte_xor(raw: bytes) -> bytes:
    if len(raw) <= 1:
        return raw
    arr = np.frombuffer(raw, dtype=np.uint8)
    out = arr.copy()
    out[1:] = arr[1:] ^ arr[:-1]
    return out.tobytes()


def _gzip_rate(raw: bytes) -> float:
    if not raw:
        return 1.0
    return float(len(gzip.compress(raw))) / float(len(raw))


def _bz2_rate(raw: bytes) -> float:
    if not raw:
        return 1.0
    return float(len(bz2.compress(raw))) / float(len(raw))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]
        if a.size < 2:
            return 0.0
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std < 1e-12 or b_std < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _float_bit_features(arr: np.ndarray) -> tuple[float, float, float]:
    if arr.dtype == np.float32:
        bits = arr.view(np.uint32)
        exponent = ((bits >> 23) & 0xFF).astype(np.uint16)
        mantissa = bits & 0x7FFFFF
    elif arr.dtype == np.float64:
        bits = arr.view(np.uint64)
        exponent = ((bits >> 52) & 0x7FF).astype(np.uint16)
        mantissa = bits & 0xFFFFFFFFFFFFF
    else:
        return 0.0, 0.0, 0.0
    counts = np.bincount(exponent, minlength=int(exponent.max()) + 1 if exponent.size else 1).astype(np.float32)
    entropy = _entropy_from_counts(counts)
    top1 = float(counts.max() / counts.sum()) if counts.sum() > 0 else 0.0
    mantissa_zero = float(np.mean(mantissa == 0)) if mantissa.size else 0.0
    return entropy, top1, mantissa_zero


def _typed_best_offset(raw: bytes, dtype_name: str | None, typesize: int) -> int:
    if not dtype_name or typesize <= 1:
        return 0
    best_offset = 0
    best_ratio = -1.0
    for offset in range(typesize):
        usable = raw[offset:]
        usable = usable[:len(usable) - (len(usable) % typesize)]
        if not usable:
            continue
        arr = np.frombuffer(usable, dtype=np.dtype(dtype_name))
        if arr.dtype.kind == "f":
            score = float(np.isfinite(arr).mean())
        else:
            score = 1.0
        if score > best_ratio:
            best_ratio = score
            best_offset = offset
    return best_offset


def build_feature_vector(
    raw: bytes,
    context: PipelineContext,
    context_metadata: dict[str, Any],
) -> dict[str, float]:
    return build_feature_dict(raw, context, context_metadata)

    raw = raw or b""
    ctx = normalize_context(context, raw)
    sample = raw[: min(len(raw), 64 * 1024)]
    raw_arr = np.frombuffer(raw, dtype=np.uint8) if raw else np.zeros(0, dtype=np.uint8)
    hist = _byte_hist(raw)
    delta_bytes = _byte_delta(sample)
    xor_bytes = _byte_xor(sample)
    window_ent = _window_entropies(sample)

    feat: dict[str, float] = {}
    feat["log_file_size"] = float(math.log1p(len(raw)))
    feat["sample_ratio"] = float(len(sample) / max(1, len(raw)))
    feat["mean_byte"] = float(raw_arr.mean()) if raw_arr.size else 0.0
    feat["std_byte"] = float(raw_arr.std()) if raw_arr.size else 0.0
    feat["entropy"] = _entropy_from_counts(hist)
    feat["zero_frac"] = float(hist[0] / max(1, len(raw)))
    feat["ff_frac"] = float(hist[255] / max(1, len(raw)))
    top_sorted = np.sort(hist)[::-1]
    feat["top1_frac"] = float(top_sorted[0] / max(1, len(raw))) if top_sorted.size else 0.0
    feat["top4_frac"] = float(top_sorted[:4].sum() / max(1, len(raw))) if top_sorted.size else 0.0
    feat["unique_byte_ratio"] = float(np.count_nonzero(hist) / 256.0)
    feat["same_adjacent_ratio"] = float(np.mean(raw_arr[1:] == raw_arr[:-1])) if raw_arr.size > 1 else 0.0
    if raw_arr.size > 1:
        delta1 = np.abs(raw_arr[1:].astype(np.int16) - raw_arr[:-1].astype(np.int16))
        feat["mean_abs_delta"] = float(delta1.mean())
        feat["std_abs_delta"] = float(delta1.std())
    else:
        feat["mean_abs_delta"] = 0.0
        feat["std_abs_delta"] = 0.0
    feat["delta_entropy"] = _entropy_from_counts(_byte_hist(delta_bytes))
    feat["xor_entropy"] = _entropy_from_counts(_byte_hist(xor_bytes))
    feat["window_entropy_mean"] = float(window_ent.mean()) if window_ent.size else 0.0
    feat["window_entropy_std"] = float(window_ent.std()) if window_ent.size else 0.0
    feat["window_entropy_min"] = float(window_ent.min()) if window_ent.size else 0.0
    feat["window_entropy_max"] = float(window_ent.max()) if window_ent.size else 0.0

    dtype_name = ctx.dtype_name
    feat["is_f32"] = 1.0 if dtype_name == "float32" else 0.0
    feat["is_f64"] = 1.0 if dtype_name == "float64" else 0.0
    feat["is_unknown_dtype"] = 0.0 if dtype_name in {"float32", "float64"} else 1.0

    typed_defaults = {
        "typed_finite_ratio": 0.0,
        "typed_best_offset": 0.0,
        "typed_log_num_values": 0.0,
        "typed_zero_frac": 0.0,
        "typed_near_zero_frac": 0.0,
        "typed_sign_frac": 0.0,
        "typed_log_range": 0.0,
        "typed_std_over_meanabs": 0.0,
        "typed_mean_abs_diff1_norm": 0.0,
        "typed_mean_abs_diff2_norm": 0.0,
        "typed_diff_std_ratio": 0.0,
        "typed_repeat_frac": 0.0,
        "typed_sign_change_frac": 0.0,
        "typed_lag1_corr": 0.0,
        "typed_lag2_corr": 0.0,
        "typed_exp_entropy": 0.0,
        "typed_exp_top1": 0.0,
        "typed_mantissa_zero_score": 0.0,
    }
    feat.update(typed_defaults)

    aligned_size = len(raw)
    item_size = int(ctx.typesize or 1)
    if item_size > 1:
        aligned_size = len(raw) - (len(raw) % item_size)
    aligned = raw[:aligned_size]
    if aligned and dtype_name in {
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
    }:
        try:
            arr = typed_array(aligned, ctx).reshape(-1)
            feat["typed_best_offset"] = float(_typed_best_offset(raw, dtype_name, item_size))
            feat["typed_log_num_values"] = float(math.log1p(arr.size))
            if arr.dtype.kind == "f":
                finite_mask = np.isfinite(arr)
                feat["typed_finite_ratio"] = float(finite_mask.mean()) if arr.size else 0.0
                work = arr[finite_mask]
                near_zero_atol = 1e-6 if arr.dtype == np.float32 else 1e-12
            else:
                feat["typed_finite_ratio"] = 1.0
                work = arr
                near_zero_atol = 0.0
            if work.size:
                mean_abs = float(np.mean(np.abs(work))) if work.size else 0.0
                denom = max(mean_abs, 1e-12)
                feat["typed_zero_frac"] = float(np.mean(work == 0))
                feat["typed_near_zero_frac"] = float(np.mean(np.isclose(work, 0, atol=near_zero_atol, rtol=0.0)))
                feat["typed_sign_frac"] = float(np.mean(work < 0))
                feat["typed_log_range"] = float(math.log1p(float(np.max(work) - np.min(work)))) if work.size else 0.0
                feat["typed_std_over_meanabs"] = float(np.std(work) / denom)
                if work.size > 1:
                    diff1 = np.diff(work)
                    feat["typed_mean_abs_diff1_norm"] = float(np.mean(np.abs(diff1)) / denom)
                    feat["typed_diff_std_ratio"] = float(np.std(diff1) / max(float(np.std(work)), 1e-12))
                    feat["typed_repeat_frac"] = float(np.mean(work[1:] == work[:-1]))
                    sign_left = np.signbit(work[1:])
                    sign_right = np.signbit(work[:-1])
                    feat["typed_sign_change_frac"] = float(np.mean(sign_left != sign_right))
                    feat["typed_lag1_corr"] = _safe_corr(work[:-1], work[1:])
                if work.size > 2:
                    diff2 = np.diff(work, n=2)
                    feat["typed_mean_abs_diff2_norm"] = float(np.mean(np.abs(diff2)) / denom)
                    feat["typed_lag2_corr"] = _safe_corr(work[:-2], work[2:])
                if arr.dtype.kind == "f":
                    exp_entropy, exp_top1, mant_zero = _float_bit_features(arr[np.isfinite(arr)])
                    feat["typed_exp_entropy"] = exp_entropy
                    feat["typed_exp_top1"] = exp_top1
                    feat["typed_mantissa_zero_score"] = mant_zero
        except Exception:
            pass

    for idx in range(256):
        feat[f"byte_hist_{idx}"] = float(hist[idx] / max(1, len(raw)))
    for bit in range(8):
        if raw_arr.size:
            feat[f"bit_density_{bit}"] = float(np.mean(((raw_arr >> bit) & 1) != 0))
        else:
            feat[f"bit_density_{bit}"] = 0.0
    feat["pilot_gzip_raw"] = _gzip_rate(sample)
    feat["pilot_bz2_raw"] = _bz2_rate(sample)
    feat["pilot_gzip_delta"] = _gzip_rate(delta_bytes)
    feat["pilot_gzip_xor"] = _gzip_rate(xor_bytes)

    shape = tuple(ctx.shape) if ctx.shape is not None else ()
    feat["meta_ndim"] = float(len(shape))
    element_count = shape and int(np.prod(shape)) or (aligned_size // max(1, item_size))
    feat["meta_log_element_count"] = float(math.log1p(max(0, element_count)))
    for dim_idx in range(4):
        dim_value = shape[dim_idx] if dim_idx < len(shape) else 0
        feat[f"meta_log_shape_dim{dim_idx}"] = float(math.log1p(dim_value))
    shape_source = context_metadata.get("shape_source", "unknown")
    for name in (
        "dataset_table",
        "filename_numbers",
        "1d_fallback",
        "byte_fallback",
        "unknown",
        "user_override",
    ):
        feat[f"meta_shape_source_{name}"] = 1.0 if shape_source == name else 0.0
    feat["meta_dtype_float32"] = 1.0 if dtype_name == "float32" else 0.0
    feat["meta_dtype_float64"] = 1.0 if dtype_name == "float64" else 0.0
    feat["meta_dtype_unknown_or_other"] = 0.0 if dtype_name in {"float32", "float64"} else 1.0
    return feat


if nn is not None:
    class ThresholdResidualBlock(nn.Module):
        def __init__(self, dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.net(x)


    class QuantileBinnedDualHeadSelector(nn.Module):
        def __init__(self, in_dim: int, num_classes: int, d_model: int, num_blocks: int, dropout: float):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.blocks = nn.ModuleList(
                [ThresholdResidualBlock(d_model, dropout) for _ in range(num_blocks)]
            )
            self.cls_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes),
            )
            self.util_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.stem(x)
            for block in self.blocks:
                h = block(h)
            return self.cls_head(h), self.util_head(h)


if torch is not None and nn is not None:
    class NNMaxEnsembleSelector:
        def __init__(self, model_dir: Path | None = None):
            self.model_dir = model_dir or (ROOT_DIR / "Model")
            ckpt_path = self.model_dir / "selector_nnmax_ensemble.pt"
            feat_path = self.model_dir / "feature_names_nnmax.json"
            if not ckpt_path.exists():
                raise FileNotFoundError(ckpt_path)
            self.feature_info = json.loads(feat_path.read_text(encoding="utf-8"))
            self.base_feature_names = list(self.feature_info["base_feature_names"])
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.labels = [canonical_pipeline_name(label) for label in checkpoint["labels"]]
            self.alpha = float(checkpoint["alpha"])
            self.mean = np.asarray(checkpoint["mean"], dtype=np.float32)
            self.std = np.asarray(checkpoint["std"], dtype=np.float32)
            self.thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float32)
            self.models: list[QuantileBinnedDualHeadSelector] = []
            for row in checkpoint["models"]:
                cfg = row["config"]
                model = QuantileBinnedDualHeadSelector(
                    in_dim=int(checkpoint["input_dim"]),
                    num_classes=len(self.labels),
                    d_model=int(cfg["d_model"]),
                    num_blocks=int(cfg["num_blocks"]),
                    dropout=float(cfg["dropout"]),
                )
                model.load_state_dict(row["state"])
                model.eval()
                self.models.append(model)

        def _tensor_features(self, raw: bytes, context: PipelineContext, context_metadata: dict[str, Any]) -> torch.Tensor:
            feat_dict = build_feature_vector(raw, context, context_metadata)
            values = np.array([feat_dict.get(name, 0.0) for name in self.base_feature_names], dtype=np.float32)
            if values.shape[0] != self.mean.shape[0]:
                raise RuntimeError(
                    f"raw feature dim mismatch: got {values.shape[0]}, expected {self.mean.shape[0]}"
                )
            std = self.std.copy()
            std[std < 1e-6] = 1.0
            x_raw = ((values - self.mean) / std).astype(np.float32)
            bins = (x_raw[:, None] > self.thresholds).astype(np.float32)
            x = np.concatenate([x_raw, bins.reshape(-1)], axis=0).astype(np.float32)
            return torch.from_numpy(x[None, :])

        @torch.no_grad()
        def rank(
            self,
            raw: bytes,
            context: PipelineContext,
            context_metadata: dict[str, Any],
            available: set[str],
        ) -> list[dict[str, Any]]:
            x = self._tensor_features(raw, context, context_metadata)
            cls_rows = []
            util_rows = []
            for model in self.models:
                cls, util = model(x)
                cls_rows.append(cls.numpy())
                util_rows.append(util.numpy())
            cls_mean = np.mean(cls_rows, axis=0)[0]
            util_mean = np.mean(util_rows, axis=0)[0]
            scores = cls_mean + self.alpha * util_mean
            rows = []
            for idx, label in enumerate(self.labels):
                available_flag = label in available
                rows.append(
                    {
                        "pipeline": label,
                        "score": float(scores[idx]),
                        "available": available_flag,
                    }
                )
            rows.sort(key=lambda item: item["score"], reverse=True)
            return rows
else:
    class NNMaxEnsembleSelector:
        def __init__(self, model_dir: Path | None = None):
            raise RuntimeError("torch is not installed")


def exhaustive_select(
    raw: bytes,
    context: PipelineContext,
    context_metadata: dict[str, Any],
    candidates: list[str],
) -> SelectionResult:
    ranking: list[dict[str, Any]] = []
    best_artifact: CompressionArtifact | None = None
    best_size: int | None = None
    for pipeline_name in candidates:
        selector_meta = {"mode": "exhaustive", "candidate_count": len(candidates)}
        try:
            artifact = compress_with_pipeline(
                raw,
                pipeline_name,
                context,
                context_metadata=context_metadata,
                selector_metadata=selector_meta,
            )
            total_size = artifact_total_size(artifact)
            ranking.append({"pipeline": pipeline_name, "stored_size": total_size, "available": True})
            if best_size is None or total_size < best_size:
                best_size = total_size
                best_artifact = artifact
        except Exception as exc:
            ranking.append({"pipeline": pipeline_name, "available": False, "reason": str(exc)})
    if best_artifact is None:
        raise RuntimeError("no deployable pipeline could compress this file")
    ranking.sort(key=lambda item: (not item["available"], item.get("stored_size", float("inf"))))
    return SelectionResult(selector_mode="exhaustive", artifact=best_artifact, ranking=ranking)


def model_select(
    raw: bytes,
    context: PipelineContext,
    context_metadata: dict[str, Any],
    candidates: list[str],
    *,
    model_dir: Path | None = None,
) -> SelectionResult:
    selector = NNMaxEnsembleSelector(model_dir=model_dir)
    ranking = selector.rank(raw, context, context_metadata, set(candidates))
    picked = next((row for row in ranking if row["available"]), None)
    if picked is None:
        raise RuntimeError("model predicted no available pipeline")
    selector_meta = {"mode": "model", "top_prediction": picked["pipeline"]}
    artifact = compress_with_pipeline(
        raw,
        picked["pipeline"],
        context,
        context_metadata=context_metadata,
        selector_metadata=selector_meta,
    )
    return SelectionResult(selector_mode="model", artifact=artifact, ranking=ranking)


def hybrid_select(
    raw: bytes,
    context: PipelineContext,
    context_metadata: dict[str, Any],
    candidates: list[str],
    *,
    model_dir: Path | None = None,
    top_k: int = 3,
) -> SelectionResult:
    selector = NNMaxEnsembleSelector(model_dir=model_dir)
    model_ranking = selector.rank(raw, context, context_metadata, set(candidates))
    shortlist = [row["pipeline"] for row in model_ranking if row["available"]][: max(1, top_k)]
    if not shortlist:
        raise RuntimeError("hybrid mode found no available shortlist")

    ranking: list[dict[str, Any]] = []
    best_artifact: CompressionArtifact | None = None
    best_size: int | None = None
    for pipeline_name in shortlist:
        selector_meta = {
            "mode": "hybrid",
            "shortlist": shortlist,
            "top_k": int(top_k),
        }
        try:
            artifact = compress_with_pipeline(
                raw,
                pipeline_name,
                context,
                context_metadata=context_metadata,
                selector_metadata=selector_meta,
            )
            total_size = artifact_total_size(artifact)
            ranking.append({"pipeline": pipeline_name, "stored_size": total_size, "available": True})
            if best_size is None or total_size < best_size:
                best_size = total_size
                best_artifact = artifact
        except Exception as exc:
            ranking.append({"pipeline": pipeline_name, "available": False, "reason": str(exc)})
    if best_artifact is None:
        raise RuntimeError("hybrid mode could not compress with shortlist pipelines")
    ranking.sort(key=lambda item: (not item["available"], item.get("stored_size", float("inf"))))
    return SelectionResult(selector_mode="hybrid", artifact=best_artifact, ranking=ranking)


def select_pipeline(
    raw: bytes,
    context: PipelineContext,
    context_metadata: dict[str, Any],
    *,
    mode: str = "auto",
    model_dir: Path | None = None,
    top_k: int = 3,
    candidates: list[str] | None = None,
) -> SelectionResult:
    mode = canonical_name(mode)
    candidates = [canonical_pipeline_name(item) for item in (candidates or deployable_pipeline_names())]
    try:
        if mode == "exhaustive":
            return exhaustive_select(raw, context, context_metadata, candidates)
        if mode == "model":
            return model_select(raw, context, context_metadata, candidates, model_dir=model_dir)
        if mode == "hybrid":
            return hybrid_select(raw, context, context_metadata, candidates, model_dir=model_dir, top_k=top_k)
        if mode != "auto":
            raise ValueError(f"unknown selector mode: {mode}")
        try:
            return hybrid_select(raw, context, context_metadata, candidates, model_dir=model_dir, top_k=top_k)
        except Exception:
            result = exhaustive_select(raw, context, context_metadata, candidates)
            return SelectionResult(
                selector_mode="exhaustive",
                artifact=result.artifact,
                ranking=result.ranking,
                fallback_used=True,
            )
    except Exception as exc:
        if mode in {"model", "hybrid"}:
            raise RuntimeError(
                f"{mode} selector failed. Consider installing torch and optional codecs, "
                f"or switch to --selector exhaustive. Root cause: {exc}"
            ) from exc
        raise
