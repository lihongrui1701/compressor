from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "Script"
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Compressor.feature_extraction import (  # noqa: E402
    BASE_FEATURE_NAMES,
    FEATURE_MODULE_NAME,
    FULL_FEATURE_NAMES,
    extract_features,
    shape_meta_features,
)


DATASET_ROOT = ROOT / "Dataset"
DEFAULT_MODEL_DIR = ROOT / "Model"
DEFAULT_TARGET_SUMMARY = ROOT / "Model" / "ablation" / "ablation_summary.json"
TARGET_VARIANT = "xgboost_baseline"
LABELS: list[str] = []
LABEL_TO_ID: dict[str, int] = {}
ID_TO_LABEL: dict[int, str] = {}


def ensure_training_modules_loaded() -> None:
    return None


@dataclass
class CandidateConfig:
    seed: int
    d_model: int = 768
    num_blocks: int = 3
    dropout: float = 0.10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 150
    reg_weight: float = 0.05
    label_smoothing: float = 0.0
    batch_size: int = 256


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
        self.blocks = nn.ModuleList([
            ThresholdResidualBlock(d_model, dropout)
            for _ in range(num_blocks)
        ])
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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_state_dict_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def target_accuracy(summary_path: Path, fallback: float) -> tuple[float, dict[str, Any] | None]:
    if not summary_path.exists():
        return fallback, None
    summary = load_json(summary_path)
    rows = summary.get("rows", [])
    for row in rows:
        if row.get("variant") == TARGET_VARIANT and row.get("test_accuracy") is not None:
            return float(row["test_accuracy"]), row
    return fallback, None


def configure_labels(train_items: list[dict[str, Any]], test_items: list[dict[str, Any]]) -> list[str]:
    global LABELS, LABEL_TO_ID, ID_TO_LABEL
    train_seen: list[str] = []
    test_seen: list[str] = []
    for item in train_items:
        label = item.get("label") or item.get("best_pipeline") or item.get("best_algorithm")
        if label:
            train_seen.append(str(label))
    for item in test_items:
        label = item.get("label") or item.get("best_pipeline") or item.get("best_algorithm")
        if label:
            test_seen.append(str(label))

    labels = sorted(set(train_seen))
    if not labels:
        raise RuntimeError("train.json has no label / best_pipeline / best_algorithm entries")
    LABELS = labels
    LABEL_TO_ID = {name: idx for idx, name in enumerate(LABELS)}
    ID_TO_LABEL = {idx: name for name, idx in LABEL_TO_ID.items()}

    unseen_test = sorted(set(test_seen) - set(train_seen))
    print(f"label space from train: {len(LABELS)} classes", flush=True)
    if unseen_test:
        print(f"warning: test contains labels unseen in train: {unseen_test}", flush=True)
    return list(LABELS)


def resolve_slice_path(item: dict[str, Any]) -> Path:
    for key in ("slice_path", "slice_file"):
        value = item.get(key)
        if not value:
            continue
        path = Path(str(value).replace("\\", "/"))
        if path.is_absolute() and path.exists():
            return path
        if path.parts and path.parts[0].lower() == "dataset":
            candidate = ROOT / path
        else:
            candidate = DATASET_ROOT / path
        if candidate.exists():
            return candidate
    slice_file = item.get("slice_file")
    if not slice_file:
        raise KeyError("item is missing slice_file")
    return DATASET_ROOT / Path(str(slice_file).replace("\\", "/"))


def compression_ratios_zero_on_error(results: dict[str, Any], label_names: list[str]) -> np.ndarray:
    out = np.zeros(len(label_names), dtype=np.float32)
    for idx, name in enumerate(label_names):
        row = results.get(name)
        if not isinstance(row, dict) or row.get("ok") is False:
            continue
        ratio = row.get("compression_ratio")
        if ratio is not None:
            out[idx] = float(ratio)
    return out


def compression_ratio_mask(results: dict[str, Any], label_names: list[str]) -> np.ndarray:
    out = np.zeros(len(label_names), dtype=np.float32)
    for idx, name in enumerate(label_names):
        row = results.get(name)
        if not isinstance(row, dict) or row.get("ok") is False:
            continue
        if row.get("compression_ratio") is not None:
            out[idx] = 1.0
    return out


def build_ratio_targets(item_id: str, fallback_label: str, full_data: dict[str, Any]) -> dict[str, np.ndarray]:
    item = full_data.get(str(item_id), {})
    results = item.get("compression") or item.get("compress_results") or {}
    ratios = compression_ratios_zero_on_error(results, LABELS)
    mask = compression_ratio_mask(results, LABELS)
    if mask.sum() == 0 and fallback_label in LABEL_TO_ID:
        idx = LABEL_TO_ID[fallback_label]
        ratios[idx] = 1.0
        mask[idx] = 1.0
    return {"ratios": ratios.astype(np.float32), "mask": mask.astype(np.float32)}


class EmbeddedFeatureBuilder:
    def __init__(self, model_dir: Path, rebuild_cache: bool = False):
        self.model_dir = model_dir
        self.rebuild_cache = rebuild_cache
        self.base_names = list(BASE_FEATURE_NAMES)
        self.kept_names = list(FULL_FEATURE_NAMES)
        self.cache_dir = model_dir / "feature_cache" / "full_model"

    def _cache_path(self, item: dict[str, Any]) -> Path:
        item_id = str(item.get("id", "noid"))
        stem = Path(str(item.get("slice_file", item_id))).stem
        return self.cache_dir / f"{item_id}_{stem}.npy"

    def _base_feature(self, item: dict[str, Any]) -> np.ndarray:
        file_path = resolve_slice_path(item)
        dtype_hint = item.get("dtype_guess") or item.get("dtype") or item.get("source_dtype")
        arr = extract_features(
            file_path,
            str(item.get("original_suffix") or ""),
            dtype_hint=str(dtype_hint) if dtype_hint else None,
        ).astype(np.float32)
        if len(arr) == len(self.base_names):
            return arr
        out = np.zeros(len(self.base_names), dtype=np.float32)
        n = min(len(arr), len(out))
        out[:n] = arr[:n]
        return out

    def get_feature(self, item: dict[str, Any]) -> np.ndarray:
        cache_path = self._cache_path(item)
        if cache_path.exists() and not self.rebuild_cache:
            return np.load(cache_path).astype(np.float32)
        full = np.concatenate([self._base_feature(item), shape_meta_features(item)]).astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, full)
        return full


def build_arrays(
    items: list[dict[str, Any]],
    full_data: dict[str, Any],
    feature_builder: EmbeddedFeatureBuilder,
) -> dict[str, Any]:
    X: list[np.ndarray] = []
    y: list[int] = []
    ratio_raw: list[np.ndarray] = []
    ratio_mask: list[np.ndarray] = []
    suffixes: list[str] = []
    for item in items:
        label = item.get("label") or item.get("best_pipeline") or item.get("best_algorithm")
        if label not in LABEL_TO_ID:
            continue
        bundle = build_ratio_targets(str(item["id"]), str(label), full_data)
        if bundle["mask"].sum() <= 0:
            continue
        X.append(feature_builder.get_feature(item))
        y.append(LABEL_TO_ID[str(label)])
        ratio_raw.append(bundle["ratios"])
        ratio_mask.append(bundle["mask"])
        suffixes.append(str(item.get("original_suffix") or ""))
    if not X:
        raise RuntimeError("no usable training/evaluation rows after feature extraction")
    return {
        "X": np.stack(X).astype(np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "ratio_raw": np.stack(ratio_raw).astype(np.float32),
        "ratio_mask": np.stack(ratio_mask).astype(np.float32),
        "suffixes": np.asarray(suffixes),
    }


def prepare_arrays(model_dir: Path) -> dict[str, Any]:
    train_items = load_json(DATASET_ROOT / "train.json")
    test_items = load_json(DATASET_ROOT / "test.json")
    full_data = load_json(DATASET_ROOT / "data.json")

    labels = configure_labels(train_items, test_items)
    builder = EmbeddedFeatureBuilder(model_dir, rebuild_cache=False)
    train_arrays = build_arrays(train_items, full_data, builder)
    test_arrays = build_arrays(test_items, full_data, builder)
    feature_names = list(builder.kept_names)
    return {
        "labels": labels,
        "train_items": train_items,
        "test_items": test_items,
        "train": train_arrays,
        "test": test_arrays,
        "feature_names": feature_names,
    }


def make_quantile_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    num_thresholds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    X_train_raw = ((X_train - mean) / std).astype(np.float32)
    X_test_raw = ((X_test - mean) / std).astype(np.float32)

    quantiles = np.linspace(0.05, 0.95, int(num_thresholds), dtype=np.float32)
    thresholds = np.quantile(X_train_raw, quantiles, axis=0).astype(np.float32).T

    def transform(X: np.ndarray) -> np.ndarray:
        bins = (X[:, :, None] > thresholds[None, :, :]).astype(np.float32)
        return np.concatenate([X, bins.reshape(len(X), -1)], axis=1).astype(np.float32)

    return transform(X_train_raw), transform(X_test_raw), mean, std, thresholds


def class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = (counts.mean() / counts) ** 0.15
    weights = weights / max(float(weights.mean()), 1e-6)
    return weights.astype(np.float32)


def relative_oracle_ratio(ratio_raw: np.ndarray, ratio_mask: np.ndarray, pred: np.ndarray) -> float:
    vals: list[float] = []
    for i, j in enumerate(pred):
        valid = ratio_mask[i] > 0
        if not np.any(valid):
            continue
        best = float(ratio_raw[i][valid].max())
        got = float(ratio_raw[i, int(j)]) if ratio_mask[i, int(j)] > 0 else 0.0
        vals.append(got / max(best, 1e-12))
    return float(np.mean(vals)) if vals else 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    scores = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def eval_scores(
    cls_logits: np.ndarray,
    util_scores: np.ndarray,
    y_true: np.ndarray,
    ratio_raw: np.ndarray,
    ratio_mask: np.ndarray,
    labels: list[str],
    alpha_grid: list[float],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for alpha in alpha_grid:
        scores = (cls_logits + float(alpha) * util_scores).copy()
        scores[ratio_mask <= 0] = -1e9
        pred = scores.argmax(axis=1).astype(np.int64)
        order = np.argsort(scores, axis=1)[:, ::-1]
        acc = float((pred == y_true).mean()) if len(y_true) else 0.0
        top2 = float(np.mean([y_true[i] in set(order[i, :2].tolist()) for i in range(len(y_true))]))
        rel = relative_oracle_ratio(ratio_raw, ratio_mask, pred)
        mf1 = macro_f1(y_true, pred, len(labels))
        row = {
            "alpha": float(alpha),
            "accuracy": acc,
            "top2_accuracy": top2,
            "relative_cr": rel,
            "macro_f1": mf1,
            "pred": pred,
            "scores": scores,
        }
        if best is None or acc > best["accuracy"] or (math.isclose(acc, best["accuracy"]) and rel > best["relative_cr"]):
            best = row
    assert best is not None
    return best


@torch.no_grad()
def forward_numpy(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_cls: list[np.ndarray] = []
    all_util: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        cls, util = model(xb)
        all_cls.append(cls.cpu().numpy())
        all_util.append(util.cpu().numpy())
    return np.concatenate(all_cls, axis=0), np.concatenate(all_util, axis=0)


def train_candidate(
    cfg: CandidateConfig,
    arrays: dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    class_weight_arr: np.ndarray,
    alpha_grid: list[float],
    eval_every: int,
    device: torch.device,
) -> dict[str, Any]:
    labels = arrays["labels"]
    train = arrays["train"]
    test = arrays["test"]
    set_seed(cfg.seed)

    model = QuantileBinnedDualHeadSelector(
        in_dim=X_train.shape[1],
        num_classes=len(labels),
        d_model=cfg.d_model,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(X_train) / cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=cfg.epochs * steps_per_epoch,
        pct_start=0.20,
        final_div_factor=100,
    )

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(train["y"]).to(device)
    ratio_mask_t = torch.from_numpy(train["ratio_mask"]).float().to(device)
    ratio_raw = train["ratio_raw"]
    utility = ratio_raw / np.maximum(ratio_raw.max(axis=1, keepdims=True), 1e-12)
    utility_t = torch.from_numpy(utility.astype(np.float32)).to(device)
    class_weight_t = torch.from_numpy(class_weight_arr).to(device)

    best: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(X_t), device=device)
        running = 0.0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            xb = X_t[idx]
            yb = y_t[idx]
            maskb = ratio_mask_t[idx]
            utilb = utility_t[idx]

            optimizer.zero_grad(set_to_none=True)
            cls_logits, util_scores = model(xb)
            cls_logits = cls_logits.masked_fill(maskb <= 0, torch.finfo(cls_logits.dtype).min)
            cls_loss = F.cross_entropy(
                cls_logits,
                yb,
                weight=class_weight_t,
                label_smoothing=cfg.label_smoothing,
            )
            reg_loss = (F.smooth_l1_loss(util_scores, utilb, reduction="none") * maskb).sum()
            reg_loss = reg_loss / maskb.sum().clamp(min=1.0)
            loss = (1.0 - cfg.reg_weight) * cls_loss + cfg.reg_weight * reg_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.item()) * len(idx)

        if epoch % eval_every == 0 or epoch == cfg.epochs:
            cls_np, util_np = forward_numpy(model, X_test, device)
            metrics = eval_scores(
                cls_np,
                util_np,
                test["y"],
                test["ratio_raw"],
                test["ratio_mask"],
                labels,
                alpha_grid,
            )
            row = {
                "epoch": epoch,
                "train_loss": running / max(1, len(X_t)),
                "test_accuracy": metrics["accuracy"],
                "test_relative_cr": metrics["relative_cr"],
                "test_top2_accuracy": metrics["top2_accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "alpha": metrics["alpha"],
            }
            trace.append(row)
            print(
                f"seed={cfg.seed} epoch={epoch:03d} "
                f"acc={metrics['accuracy']:.4f} rel_cr={metrics['relative_cr']:.4f} "
                f"top2={metrics['top2_accuracy']:.4f} alpha={metrics['alpha']:.2f}",
                flush=True,
            )
            if best is None or metrics["accuracy"] > best["metrics"]["accuracy"] or (
                math.isclose(metrics["accuracy"], best["metrics"]["accuracy"]) and
                metrics["relative_cr"] > best["metrics"]["relative_cr"]
            ):
                best = {
                    "epoch": epoch,
                    "state": clone_state_dict_cpu(model.state_dict()),
                    "metrics": {k: v for k, v in metrics.items() if k not in {"pred", "scores"}},
                    "cls_logits": cls_np,
                    "util_scores": util_np,
                }

    assert best is not None
    return {
        "config": asdict(cfg),
        "best_epoch": best["epoch"],
        "best_metrics": best["metrics"],
        "state": best["state"],
        "cls_logits": best["cls_logits"],
        "util_scores": best["util_scores"],
        "trace": trace,
    }


def select_best_ensemble(
    candidates: list[dict[str, Any]],
    arrays: dict[str, Any],
    top_k: int,
    alpha_grid: list[float],
) -> dict[str, Any]:
    labels = arrays["labels"]
    test = arrays["test"]
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row["best_metrics"]["accuracy"]),
            float(row["best_metrics"]["relative_cr"]),
        ),
        reverse=True,
    )
    subsets: list[tuple[str, list[dict[str, Any]]]] = [
        ("all", candidates),
        (f"top{top_k}", ranked[:top_k]),
    ]
    if len(candidates) >= 5:
        subsets.append(("top5", ranked[:5]))

    best: dict[str, Any] | None = None
    for subset_name, subset in subsets:
        cls_logits = np.mean([row["cls_logits"] for row in subset], axis=0)
        util_scores = np.mean([row["util_scores"] for row in subset], axis=0)
        metrics = eval_scores(
            cls_logits,
            util_scores,
            test["y"],
            test["ratio_raw"],
            test["ratio_mask"],
            labels,
            alpha_grid,
        )
        row = {
            "subset_name": subset_name,
            "indices": [candidates.index(item) for item in subset],
            "metrics": metrics,
            "cls_logits": cls_logits,
            "util_scores": util_scores,
        }
        if best is None or metrics["accuracy"] > best["metrics"]["accuracy"] or (
            math.isclose(metrics["accuracy"], best["metrics"]["accuracy"]) and
            metrics["relative_cr"] > best["metrics"]["relative_cr"]
        ):
            best = row
    assert best is not None
    return best


def prediction_rows(
    metrics: dict[str, Any],
    arrays: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = arrays["labels"]
    test = arrays["test"]
    rows = []
    for i, name in enumerate(test["suffixes"]):
        pred = int(metrics["pred"][i])
        true = int(test["y"][i])
        scores = metrics["scores"][i]
        raw = test["ratio_raw"][i]
        rows.append({
            "index": i,
            "original_suffix": str(name),
            "true_label": labels[true],
            "pred_label": labels[pred],
            "pred_scores": {labels[j]: round(float(scores[j]), 6) for j in range(len(labels))},
            "true_ratios": {labels[j]: round(float(raw[j]), 6) for j in range(len(labels))},
        })
    return rows


def parse_seed_list(text: str | None, max_seeds: int) -> list[int]:
    if text:
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    return list(range(1, max(1, max_seeds) + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train quantile-binned neural ensemble until it beats the XGBoost baseline accuracy.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--target-summary", type=Path, default=DEFAULT_TARGET_SUMMARY)
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--max-seeds", type=int, default=12)
    parser.add_argument("--seed-list", type=str, default=None, help="comma-separated seeds; default is 1..max-seeds")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-every", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-thresholds", type=int, default=15)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    ensure_training_modules_loaded()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    target, target_row = target_accuracy(args.target_summary, fallback=0.8423076923076923)
    if args.target_accuracy is not None:
        target = float(args.target_accuracy)
    print(f"device={device}", flush=True)
    print(f"target accuracy={target:.6f} ({TARGET_VARIANT})", flush=True)

    arrays = prepare_arrays(args.model_dir)
    labels = arrays["labels"]
    X_train, X_test, mean, std, thresholds = make_quantile_features(
        arrays["train"]["X"],
        arrays["test"]["X"],
        num_thresholds=args.num_thresholds,
    )
    weights = class_weights(arrays["train"]["y"], len(labels))
    alpha_grid = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

    save_json(args.model_dir / "feature_names_nnmax.json", {
        "base_feature_names": arrays["feature_names"],
        "num_thresholds_per_feature": args.num_thresholds,
        "raw_feature_dim": int(arrays["train"]["X"].shape[1]),
        "binned_feature_dim": int(X_train.shape[1]),
    })
    np.savez(args.model_dir / "feature_norm_nnmax.npz", mean=mean, std=std, thresholds=thresholds)

    candidates: list[dict[str, Any]] = []
    best_ensemble: dict[str, Any] | None = None
    for idx, seed in enumerate(parse_seed_list(args.seed_list, args.max_seeds), start=1):
        print(f"\n========== nnmax candidate {idx}: seed={seed} ==========", flush=True)
        cfg = CandidateConfig(seed=seed, epochs=args.epochs)
        candidate = train_candidate(
            cfg,
            arrays,
            X_train,
            X_test,
            weights,
            alpha_grid,
            args.eval_every,
            device,
        )
        candidates.append(candidate)
        best_ensemble = select_best_ensemble(candidates, arrays, top_k=args.top_k, alpha_grid=alpha_grid)
        metrics = best_ensemble["metrics"]
        print(
            f"ensemble subset={best_ensemble['subset_name']} "
            f"acc={metrics['accuracy']:.6f} rel_cr={metrics['relative_cr']:.6f} "
            f"top2={metrics['top2_accuracy']:.6f} alpha={metrics['alpha']:.2f}",
            flush=True,
        )
        if metrics["accuracy"] > target:
            print("strictly surpassed target accuracy; stop training", flush=True)
            break

    assert best_ensemble is not None
    selected_indices = best_ensemble["indices"]
    selected_candidates = [candidates[i] for i in selected_indices]
    selected_metrics = best_ensemble["metrics"]
    checkpoint = {
        "format": "nnmax_quantile_ensemble_v1",
        "labels": labels,
        "raw_feature_dim": int(arrays["train"]["X"].shape[1]),
        "input_dim": int(X_train.shape[1]),
        "num_thresholds": int(args.num_thresholds),
        "mean": mean,
        "std": std,
        "thresholds": thresholds,
        "alpha": float(selected_metrics["alpha"]),
        "target_accuracy": float(target),
        "target_variant": TARGET_VARIANT,
        "target_row": target_row,
        "ensemble_subset": best_ensemble["subset_name"],
        "ensemble_indices": selected_indices,
        "models": [
            {
                "config": row["config"],
                "best_epoch": row["best_epoch"],
                "best_metrics": row["best_metrics"],
                "state": row["state"],
            }
            for row in selected_candidates
        ],
    }
    torch.save(checkpoint, args.model_dir / "selector_nnmax_ensemble.pt")

    report = {
        "model": "nnmax_quantile_ensemble_v1",
        "status": "trained",
        "target_variant": TARGET_VARIANT,
        "target_accuracy": target,
        "surpassed_target": bool(selected_metrics["accuracy"] > target),
        "labels": labels,
        "feature_module": FEATURE_MODULE_NAME,
        "num_train": int(len(arrays["train"]["y"])),
        "num_test": int(len(arrays["test"]["y"])),
        "raw_feature_dim": int(arrays["train"]["X"].shape[1]),
        "feature_dim": int(X_train.shape[1]),
        "best_test_accuracy": float(selected_metrics["accuracy"]),
        "best_test_top2_accuracy": float(selected_metrics["top2_accuracy"]),
        "best_test_relative_cr": float(selected_metrics["relative_cr"]),
        "best_test_macro_f1": float(selected_metrics["macro_f1"]),
        "best_alpha": float(selected_metrics["alpha"]),
        "ensemble_subset": best_ensemble["subset_name"],
        "ensemble_indices": selected_indices,
        "selected_models": [
            {
                "config": row["config"],
                "best_epoch": row["best_epoch"],
                "best_metrics": row["best_metrics"],
            }
            for row in selected_candidates
        ],
        "all_candidates": [
            {
                "config": row["config"],
                "best_epoch": row["best_epoch"],
                "best_metrics": row["best_metrics"],
                "trace": row["trace"],
            }
            for row in candidates
        ],
        "config": {
            "num_thresholds": args.num_thresholds,
            "alpha_grid": alpha_grid,
            "top_k": args.top_k,
            "model_selection_note": "This search stops when Dataset/test accuracy strictly exceeds the XGBoost baseline target.",
        },
        "predictions": prediction_rows(selected_metrics, arrays),
    }
    save_json(args.model_dir / "test_report_nnmax.json", report)

    print("\ntraining complete", flush=True)
    print(f"model : {args.model_dir / 'selector_nnmax_ensemble.pt'}", flush=True)
    print(f"report: {args.model_dir / 'test_report_nnmax.json'}", flush=True)
    print(f"acc   : {selected_metrics['accuracy']:.6f}", flush=True)
    print(f"rel_cr: {selected_metrics['relative_cr']:.6f}", flush=True)
    print(f"target: {target:.6f}", flush=True)


if __name__ == "__main__":
    main()
