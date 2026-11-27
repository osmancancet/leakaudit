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
    max_lookup_recovery: float = 0.50   # normalized recovery of the sentence lookup


@dataclass
class ContentAwareThresholds:
    """Second, rate-based profile. Duplication and label conflicts are judged per
    1,000 documents, opening-token views by their excess over a random window of
    the same length (content that every window carries is not a positional
    shortcut), and the lookup only where it is informative."""
    min_template_uniqueness: float = 0.95
    max_near_dup_pairs_per_1000: float = 1.0
    max_label_conflicts_per_1000: float = 1.0
    max_opening_excess: float = 0.25
    max_normalized_recovery: float = 0.50
    max_lookup_recovery: float = 0.50
    max_lookup_fallback: float = 0.90


def content_aware_certificate(report: Dict, th: Optional[ContentAwareThresholds] = None) -> Dict:
    """Verdict under the rate-based, content-aware profile, computed from a report."""
    th = th or ContentAwareThresholds()
    L1, L2, L3 = report["layer1"], report["layer2"], report["layer3"]
    n = max(1, report["n_documents"])
    views = L2["views"]
    rw = views.get("random_window_10_tokens", {}).get("normalized_recovery", 0.0)
    worst, worst_view = float("-inf"), None
    for k, v in views.items():
        if k == "random_window_10_tokens":
            continue
        score = v["normalized_recovery"] - rw if k.startswith("first_") else v["normalized_recovery"]
        limit = th.max_opening_excess if k.startswith("first_") else th.max_normalized_recovery
        if score - limit > worst:
            worst, worst_view = score - limit, k
    informative = L3["lookup_fallback_rate"] <= th.max_lookup_fallback
    checks = {
        "L1_template_uniqueness": L1["template_uniqueness"] >= th.min_template_uniqueness,
        "L1_near_duplicates": L1["near_dup_pairs_per_1000_docs"] <= th.max_near_dup_pairs_per_1000,
        "L1_label_conflicts": 1000 * L1["label_conflicting_duplicate_groups"] / n <= th.max_label_conflicts_per_1000,
        "L2_surface_views": worst <= 0,
        "L3_lookup": (not informative) or L3["normalized_lookup_recovery"] <= th.max_lookup_recovery,
    }
    return {"passed": all(checks.values()), "checks": checks, "worst_view": worst_view,
            "lookup_informative": informative, "thresholds": asdict(th)}


@dataclass
class AuditConfig:
    seed: int = 42
    sample_cap: int = 20000          # documents used for layers 2 and 3
    near_dup_cap: int = 50000        # documents used for MinHash-LSH
    jaccard_threshold: float = 0.8
    folds: int = 5
    grouped_splits: bool = True     # keep duplicate groups on one side of every split
    language: str = "tr"            # lower-casing rule: "tr" or "generic"
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
    from .text import set_language
    set_language(cfg.language)
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
    groups_all = L1.pop("_groups")
    groups = groups_all[idx] if cfg.grouped_splits else None

    t0 = time.time()
    L2 = layer2(texts, y, {m: sub[m].tolist() for m in metadata}, folds=cfg.folds, seed=cfg.seed,
                groups=groups)
    timings["layer2"] = round(time.time() - t0, 1)

    tr = te = None
    if split:
        s = sub[split].astype(str).str.lower()
        tr, te = np.where(s == "train")[0], np.where(s == "test")[0]
        if len(tr) == 0 or len(te) == 0:
            tr = te = None
    t0 = time.time()
    L3 = layer3(texts, y, tr, te, seed=cfg.seed, groups=groups)
    timings["layer3"] = round(time.time() - t0, 1)

    checks = {
        "L1_template_uniqueness": L1["template_uniqueness"] >= th.min_template_uniqueness,
        "L1_near_duplicates": L1["near_dup_pairs"] <= th.max_near_dup_pairs,
        "L1_label_conflicts": L1["label_conflicting_duplicate_groups"] <= th.max_label_conflicting_groups,
        "L2_surface_views": not (L2["max_normalized_recovery"] > th.max_normalized_recovery),
        "L3_lookup": not (L3["normalized_lookup_recovery"] > th.max_lookup_recovery),
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
    if not checks["L3_lookup"]:
        reasons.append(f"learning-free sentence lookup reaches {L3['lookup_rule_macro_f1']} macro-F1 and recovers "
                       f"{L3['normalized_lookup_recovery']:.0%} of the trained model's above-chance performance")

    warnings: List[str] = []
    if L2.get("folds", "grouped") != "grouped" and cfg.grouped_splits:
        warnings.append("no grouped partition places every class in every fold; layers 2 and 3 used random "
                        "folds, so their scores include memorization of duplicates")
    if L3["lookup_fallback_rate"] > 0.9:
        warnings.append(f"the sentence lookup fell back to the majority class for "
                        f"{L3['lookup_fallback_rate']:.0%} of test documents; check L3 is not informative here")
    if "user_split_test_overlap" in L3 and L3["user_split_test_overlap"] > 0:
        warnings.append(f"{L3['user_split_test_overlap']:.1%} of documents in the given test split have a "
                        f"duplicate in the given training split")

    out = {
        "tool": "leakaudit", "version": __version__,
        "corpus": name, "text_column": text, "label_column": label,
        "metadata_columns": metadata, "split_column": split,
        "classes": {c: int((y_all == i).sum()) for i, c in enumerate(classes)},
        "n_documents": int(len(data)), "n_audited_layers_2_3": int(len(sub)),
        "config": asdict(cfg),
        "layer1": L1, "layer2": L2, "layer3": L3,
        "certificate": {"passed": all(checks.values()), "checks": checks, "reasons": reasons,
                        "warnings": warnings},
        "certificate_content_aware": None,
        "seconds": timings,
    }
    out["certificate_content_aware"] = content_aware_certificate(out)
    return out
