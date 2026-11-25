"""Public API: run all layers and issue a pass/fail certificate."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .layers import layer1, layer2, layer3


@dataclass
class Thresholds:
    min_template_uniqueness: float = 0.95
    max_near_dup_pairs: int = 0
    max_label_conflicting_groups: int = 0
    max_normalized_recovery: float = 0.50
    min_lookup_margin: float = 0.10


@dataclass
class AuditConfig:
    seed: int = 42
    sample_cap: int = 20000          # documents used for layers 2 and 3
    near_dup_cap: int = 50000        # documents used for MinHash-LSH
    jaccard_threshold: float = 0.8
    folds: int = 5
    thresholds: Thresholds = field(default_factory=Thresholds)


def audit(df: pd.DataFrame, text: str, label: str, metadata: Optional[Sequence[str]] = None,
          split: Optional[str] = None, config: Optional[AuditConfig] = None,
          name: Optional[str] = None) -> Dict:
    """Audit a labelled text corpus.

    df        -- one document per row
    text      -- text column
    label     -- label column
    metadata  -- columns to test as stand-alone predictors (layer 2)
    split     -- optional column with values 'train'/'test' used by layer 3
    """
    # numpy on some BLAS builds (e.g. macOS Accelerate) emits spurious matmul
    # RuntimeWarnings inside scikit-learn; they do not affect results.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return _audit(df, text, label, metadata, split, config, name)


def _audit(df, text, label, metadata, split, config, name) -> Dict:
    cfg = config or AuditConfig()
    th = cfg.thresholds
    metadata = list(metadata or [])
    data = df[df[text].notna() & df[label].notna()].copy()
    data[text] = data[text].astype(str)
    data = data[data[text].str.strip() != ""].reset_index(drop=True)

    classes = sorted(data[label].astype(str).unique())
    y_all = data[label].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    texts_all = data[text].tolist()
    timings = {}

    t0 = time.time()
    L1 = layer1(texts_all, y_all, cfg.near_dup_cap, cfg.jaccard_threshold, cfg.seed)
    timings["layer1"] = round(time.time() - t0, 1)

    rng = np.random.RandomState(cfg.seed)
    idx = np.arange(len(data))
    if len(idx) > cfg.sample_cap:
        idx = np.sort(rng.permutation(len(idx))[:cfg.sample_cap])
    sub = data.iloc[idx].reset_index(drop=True)
    texts, y = sub[text].tolist(), y_all[idx]

    t0 = time.time()
    L2 = layer2(texts, y, {m: sub[m].tolist() for m in metadata}, folds=cfg.folds, seed=cfg.seed)
    timings["layer2"] = round(time.time() - t0, 1)

    tr = te = None
    if split:
        s = sub[split].astype(str).str.lower()
        tr, te = np.where(s == "train")[0], np.where(s == "test")[0]
        if len(tr) == 0 or len(te) == 0:
            tr = te = None
    t0 = time.time()
    L3 = layer3(texts, y, tr, te, seed=cfg.seed)
    timings["layer3"] = round(time.time() - t0, 1)

    checks = {
        "L1_template_uniqueness": L1["template_uniqueness"] >= th.min_template_uniqueness,
        "L1_near_duplicates": L1["near_dup_pairs"] <= th.max_near_dup_pairs,
        "L1_label_conflicts": L1["label_conflicting_duplicate_groups"] <= th.max_label_conflicting_groups,
        "L2_surface_views": not (L2["max_normalized_recovery"] > th.max_normalized_recovery),
        "L3_lookup_margin": L3["margin"] >= th.min_lookup_margin,
    }
    reasons: List[str] = []
    if not checks["L1_template_uniqueness"]:
        reasons.append(f"template uniqueness {L1['template_uniqueness']} < {th.min_template_uniqueness}")
    if not checks["L1_near_duplicates"]:
        reasons.append(f"{L1['near_dup_pairs']} near-duplicate pairs (Jaccard >= {cfg.jaccard_threshold})")
    if not checks["L1_label_conflicts"]:
        reasons.append(f"{L1['label_conflicting_duplicate_groups']} duplicate groups carry conflicting labels")
    if not checks["L2_surface_views"]:
        reasons.append(f"view '{L2['max_view']}' recovers {L2['max_normalized_recovery']:.0%} "
                       f"of above-chance full-text performance")
    if not checks["L3_lookup_margin"]:
        reasons.append(f"zero-learning sentence lookup reaches {L3['lookup_rule_macro_f1']} macro-F1; "
                       f"trained model margin only {L3['margin']}")

    return {
        "tool": "leakaudit", "version": __version__,
        "corpus": name, "text_column": text, "label_column": label,
        "metadata_columns": metadata, "split_column": split,
        "classes": {c: int((y_all == i).sum()) for i, c in enumerate(classes)},
        "n_documents": int(len(data)), "n_audited_layers_2_3": int(len(sub)),
        "config": asdict(cfg),
        "layer1": L1, "layer2": L2, "layer3": L3,
        "certificate": {"passed": all(checks.values()), "checks": checks, "reasons": reasons},
        "seconds": timings,
    }
