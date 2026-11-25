"""The three audit layers.

Layer 1 - documents: template uniqueness after masking, exact duplicate
          groups (and label conflicts inside them), MinHash-LSH near duplicates.
Layer 2 - surface views: how much of full-text performance an impoverished
          view (first k tokens, punctuation profile, length, a metadata column)
          recovers on its own.
Layer 3 - sentences: a zero-learning lookup rule over class-exclusive
          sentences, compared with a trained full-text reference.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from . import text as T
from .minhash import near_duplicate_pairs


# --------------------------------------------------------------------------- models
def _word_model(ngram=(1, 2), min_df=2) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=T.mask, token_pattern=r"\S+",
                                  ngram_range=ngram, min_df=min_df, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def _full_text_model(min_df=2) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=T.mask, token_pattern=r"\S+",
                                 ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=T.mask, analyzer="char_wb",
                                 ngram_range=(3, 5), min_df=min_df, sublinear_tf=True)),
    ])
    return Pipeline([("features", feats),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def _token_model() -> Pipeline:
    """For already-discretised views (profiles, buckets, metadata values)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=False, token_pattern=r"\S+", min_df=1)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def _metadata_model() -> Pipeline:
    """Whole value as a token plus character 2-4-grams, so that a column whose
    values are all distinct but follow a class-specific pattern is still caught."""
    feats = FeatureUnion([
        ("value", TfidfVectorizer(lowercase=False, token_pattern=r"\S+", min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("features", feats), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def _cv_macro_f1(model: Pipeline, X: Sequence[str], y: np.ndarray, folds: int, seed: int) -> float:
    n_min = int(np.bincount(y).min())
    k = max(2, min(folds, n_min))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    try:
        pred = cross_val_predict(model, list(X), y, cv=cv)
    except ValueError:  # e.g. empty vocabulary for a degenerate view
        return float("nan")
    return float(f1_score(y, pred, average="macro"))


def _bucket(v: float, n: int = 20) -> str:
    return "q%d" % min(n - 1, int(v * n))


# --------------------------------------------------------------------------- layer 1
def layer1(texts: Sequence[str], labels: np.ndarray, near_dup_cap: int = 50000,
           jaccard_threshold: float = 0.8, seed: int = 42) -> Dict:
    n = len(texts)
    masked = [T.mask(t) for t in texts]
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(masked):
        groups[m].append(i)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    conflicting = [g for g in dup_groups if len({labels[i] for i in g}) > 1]

    per_class = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        per_class[str(c)] = round(len({masked[i] for i in idx}) / max(1, len(idx)), 4)

    rng = np.random.RandomState(seed)
    sample = rng.permutation(n)[:min(n, near_dup_cap)]
    sh = [T.shingles(texts[i]) for i in sample]
    pairs, best = near_duplicate_pairs(sh, threshold=jaccard_threshold)
    pairs = [(int(sample[a]), int(sample[b]), j) for a, b, j in pairs]
    non_identical = [p for p in pairs if masked[p[0]] != masked[p[1]]]
    cross_label = [p for p in pairs if labels[p[0]] != labels[p[1]]]

    return {
        "n_documents": n,
        "template_uniqueness": round(len(groups) / max(1, n), 4),
        "template_uniqueness_per_class": per_class,
        "distinct_masked_templates": len(groups),
        "duplicate_groups": len(dup_groups),
        "documents_in_duplicate_groups": int(sum(len(g) for g in dup_groups)),
        "label_conflicting_duplicate_groups": len(conflicting),
        "near_dup_sample_size": int(len(sample)),
        "near_dup_jaccard_threshold": jaccard_threshold,
        "near_dup_pairs": len(pairs),
        "near_dup_pairs_non_identical": len(non_identical),
        "near_dup_pairs_cross_label": len(cross_label),
        "near_dup_max_jaccard": round(best, 4),
    }


# --------------------------------------------------------------------------- layer 2
def layer2(texts: Sequence[str], y: np.ndarray, metadata: Optional[Mapping[str, Sequence]] = None,
           first_k: Sequence[int] = (3, 5, 10), folds: int = 5, seed: int = 42) -> Dict:
    nc = len(np.unique(y))
    full = _cv_macro_f1(_full_text_model(), texts, y, folds, seed)
    chance = 1.0 / nc  # expected macro-F1 of a prior-matching random guess

    views: Dict[str, float] = {}
    for k in first_k:
        views[f"first_{k}_tokens"] = _cv_macro_f1(_word_model((1, 2), 1),
                                                  [T.first_tokens(t, k) for t in texts], y, folds, seed)
    profiles = []
    for t in texts:
        s = str(t)
        L = max(1, len(s))
        feats = [len(T.PUNCT_RE.findall(s)) / L, len(T.EMOJI_RE.findall(s)) / L * 20,
                 sum(ch.isupper() for ch in s) / L, sum(ch.isdigit() for ch in s) / L]
        profiles.append(" ".join(f"f{i}{_bucket(v)}" for i, v in enumerate(feats)))
    views["punctuation_emoji_case_profile"] = _cv_macro_f1(_token_model(), profiles, y, folds, seed)

    lengths = np.array(T.n_tokens(texts))
    views["length_only"] = _cv_macro_f1(
        _token_model(), [_bucket(v / max(1, lengths.max()), 30) for v in lengths], y, folds, seed)
    for name, values in (metadata or {}).items():
        views[f"metadata:{name}"] = _cv_macro_f1(
            _metadata_model(), ["v_" + str(v).replace(" ", "_") for v in values], y, folds, seed)

    def norm_recovery(v: float) -> float:
        if np.isnan(v) or full - chance <= 1e-9:
            return float("nan")
        return max(0.0, (v - chance) / (full - chance))

    out = {
        "full_text_macro_f1": round(full, 4),
        "chance_macro_f1": round(chance, 4),
        "views": {k: {"macro_f1": round(v, 4),
                      "recovery_ratio": round(v / full, 4) if full else float("nan"),
                      "normalized_recovery": round(norm_recovery(v), 4)}
                  for k, v in views.items()},
    }
    if nc == 2:
        a, b = lengths[y == 0], lengths[y == 1]
        ks = ks_2samp(a, b)
        out["length_ks_D"] = round(float(ks.statistic), 4)
        out["length_ks_p"] = round(float(ks.pvalue), 4)
    nr = [v["normalized_recovery"] for v in out["views"].values() if not np.isnan(v["normalized_recovery"])]
    out["max_normalized_recovery"] = round(max(nr), 4) if nr else float("nan")
    out["max_view"] = max(out["views"], key=lambda k: -1 if np.isnan(out["views"][k]["normalized_recovery"])
                          else out["views"][k]["normalized_recovery"]) if out["views"] else None
    return out


# --------------------------------------------------------------------------- layer 3
def layer3(texts: Sequence[str], y: np.ndarray, train_idx: Optional[np.ndarray] = None,
           test_idx: Optional[np.ndarray] = None, seed: int = 42, top_k: int = 40) -> Dict:
    n = len(texts)
    nc = len(np.unique(y))
    if train_idx is None or test_idx is None:
        train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=seed, stratify=y)

    sent_counts: List[Counter] = [Counter() for _ in range(nc)]
    for i in train_idx:
        for s in set(T.sentences(texts[i])):
            sent_counts[y[i]][s] += 1
    exclusive = []
    for c in range(nc):
        others = set().union(*[set(sent_counts[k]) for k in range(nc) if k != c]) if nc > 1 else set()
        exclusive.append({s for s in sent_counts[c] if s not in others})

    majority = int(np.bincount(y[train_idx], minlength=nc).argmax())
    preds = []
    for i in test_idx:
        st = set(T.sentences(texts[i]))
        hits = [len(st & exclusive[c]) for c in range(nc)]
        preds.append(int(np.argmax(hits)) if max(hits) > 0 else majority)
    lookup_f1 = float(f1_score(y[test_idx], preds, average="macro"))

    model = _full_text_model().fit([texts[i] for i in train_idx], y[train_idx])
    ref_pred = model.predict([texts[i] for i in test_idx])
    ref_f1 = float(f1_score(y[test_idx], ref_pred, average="macro"))

    inventory = {}
    for c in range(nc):
        n_c = int((y == c).sum())
        ranked = sorted(((s, v) for s, v in sent_counts[c].items() if s in exclusive[c]),
                        key=lambda kv: -kv[1])
        inventory[str(c)] = {
            "sentence_types": len(sent_counts[c]),
            "exclusive_types": len(exclusive[c]),
            f"top{top_k}_exclusive_coverage": round(min(1.0, sum(v for _, v in ranked[:top_k]) / max(1, n_c)), 4),
        }
    return {
        "test_size": int(len(test_idx)),
        "sentence_inventory": inventory,
        "lookup_rule_macro_f1": round(lookup_f1, 4),
        "trained_reference_macro_f1": round(ref_f1, 4),
        "trained_reference_accuracy": round(float(accuracy_score(y[test_idx], ref_pred)), 4),
        "margin": round(ref_f1 - lookup_f1, 4),
    }
