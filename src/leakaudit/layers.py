"""The three audit layers.

Layer 1 - documents: template uniqueness after masking, exact duplicate
          groups (and label conflicts inside them), MinHash-LSH near duplicates.
          It also returns duplicate groups (exact plus verified near duplicates,
          joined by union-find) that layers 2 and 3 use to keep related
          documents on the same side of every split.
Layer 2 - surface views: how much of full-text performance an impoverished
          view (first k tokens, a random window, punctuation profile, length,
          a metadata column) recovers on its own.
Layer 3 - sentences: a learning-free lookup rule over class-exclusive
          sentences, compared with a trained full-text reference.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from . import text as T
from .minhash import near_duplicate_pairs

BOOTSTRAP = 500


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


def _folds(y: np.ndarray, groups: Optional[np.ndarray], folds: int, seed: int, info: Optional[dict] = None):
    """Stratified folds; grouped (duplicates never straddle folds) when groups are given.

    A grouped partition is accepted only if every training and test fold contains
    every class; up to 20 shuffles are tried, after which plain stratified folds
    are used and ``info['folds']`` records the fallback.
    """
    n_min = int(np.bincount(y).min())
    k = max(2, min(folds, n_min))
    nc = len(np.unique(y))
    if groups is not None and len(np.unique(groups)) >= k:
        for s in range(seed, seed + 20):
            cv = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=s).split(np.zeros(len(y)), y, groups))
            if all(len(np.unique(y[tr])) == nc and len(np.unique(y[te])) == nc for tr, te in cv):
                if info is not None:
                    info["folds"], info["fold_seed"] = "grouped", s
                return cv
    if info is not None:
        info["folds"] = "random" if groups is None else "random (grouped partition infeasible)"
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def _oof(model: Pipeline, X: Sequence[str], y: np.ndarray, cv) -> Optional[np.ndarray]:
    try:
        return cross_val_predict(model, list(X), y, cv=cv)
    except ValueError:  # e.g. empty vocabulary for a degenerate view
        return None


def _f1(y, p) -> float:
    return float(f1_score(y, p, average="macro")) if p is not None else float("nan")


def _bucket(v: float, n: int = 20) -> str:
    return "q%d" % min(n - 1, int(v * n))


def _quantile_buckets(values: np.ndarray, n: int = 10) -> List[str]:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n + 1)[1:-1]))
    return ["len%d" % int(np.searchsorted(edges, v, side="right")) for v in values]


def recovery(f_view: float, f_full: float, chance: float) -> float:
    """Normalized recovery R = (F_view - F0) / (F_full - F0), unclipped.

    R = 0: the view is at chance; R = 1: it matches the full-text reference;
    R > 1: the view alone beats the reference model; R < 0: below chance.
    """
    if np.isnan(f_view) or f_full - chance <= 1e-9:
        return float("nan")
    return (f_view - chance) / (f_full - chance)


def _boot_recovery(y, p_view, p_full, chance, seed) -> Tuple[float, float]:
    """95% percentile interval of R from a paired bootstrap over documents."""
    if p_view is None or p_full is None:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    n, vals = len(y), []
    for _ in range(BOOTSTRAP):
        b = rng.randint(0, n, n)
        vals.append(recovery(_f1(y[b], p_view[b]), _f1(y[b], p_full[b]), chance))
    vals = np.array([v for v in vals if not np.isnan(v)])
    if not len(vals):
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------- layer 1
def _union_find(n: int, edges) -> np.ndarray:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(n)])


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

    # Near duplicates on a seeded sample. Identical masked texts are counted
    # analytically (sum of C(g, 2) per group); LSH runs on one representative
    # per distinct masked text, so large exact-duplicate clusters cannot hide
    # near-duplicate candidates behind the LSH bucket cap.
    rng = np.random.RandomState(seed)
    sample = np.sort(rng.permutation(n)[:min(n, near_dup_cap)])
    in_sample: Dict[str, List[int]] = defaultdict(list)
    for i in sample:
        in_sample[masked[i]].append(int(i))
    reps = [m[0] for m in in_sample.values()]
    identical_pairs = sum(len(m) * (len(m) - 1) // 2 for m in in_sample.values())
    identical_cross = sum(sum(1 for a in range(len(m)) for b in range(a + 1, len(m)) if labels[m[a]] != labels[m[b]])
                          for m in in_sample.values() if len(m) > 1 and len({labels[i] for i in m}) > 1)
    sh = [T.shingles(texts[i]) for i in reps]
    pairs, best, skipped = near_duplicate_pairs(sh, threshold=jaccard_threshold, seed=seed, return_skipped=True)
    rep_pairs = [(reps[a], reps[b], j) for a, b, j in pairs]
    # expand representative pairs to every member pair across the two groups
    non_identical, cross_label = 0, identical_cross
    for a, b, _ in rep_pairs:
        ga, gb = in_sample[masked[a]], in_sample[masked[b]]
        non_identical += len(ga) * len(gb)
        cross_label += sum(1 for x in ga for z in gb if labels[x] != labels[z])
    if identical_pairs:
        best = 1.0

    edges = [(g[0], j) for g in dup_groups for j in g[1:]] + [(a, b) for a, b, _ in rep_pairs]
    group_ids = _union_find(n, edges)
    total_pairs = identical_pairs + non_identical
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
        "near_dup_pairs": int(total_pairs),
        "near_dup_pairs_identical": int(identical_pairs),
        "near_dup_pairs_non_identical": int(non_identical),
        "near_dup_pairs_cross_label": int(cross_label),
        "near_dup_documents_affected": int(sum(np.bincount(group_ids)[group_ids] > 1)),
        "near_dup_pairs_per_1000_docs": round(1000 * total_pairs / max(1, len(sample)), 3),
        "near_dup_max_jaccard": round(best, 4),
        "lsh_buckets_skipped": int(skipped),
        "split_groups": int(len(np.unique(group_ids))),
        "_groups": group_ids,  # removed before the report is written
    }


# --------------------------------------------------------------------------- layer 2
def layer2(texts: Sequence[str], y: np.ndarray, metadata: Optional[Mapping[str, Sequence]] = None,
           first_k: Sequence[int] = (3, 5, 10), folds: int = 5, seed: int = 42,
           groups: Optional[np.ndarray] = None) -> Dict:
    nc = len(np.unique(y))
    fold_info: Dict = {}
    cv = _folds(y, groups, folds, seed, fold_info)
    p_full = _oof(_full_text_model(), texts, y, cv)
    full = _f1(y, p_full)
    # A label-independent guess that draws class c with its prior p_c has
    # precision = recall = p_c for class c, so its expected macro-F1 is
    # (1/C) * sum_c p_c = 1/C for any class balance.
    chance = 1.0 / nc

    view_inputs: Dict[str, Tuple[Pipeline, List[str]]] = {}
    for k in first_k:
        view_inputs[f"first_{k}_tokens"] = (_word_model((1, 2), 1), [T.first_tokens(t, k) for t in texts])
    # control view: a window of the same length starting at a random position
    rng = np.random.RandomState(seed)
    win = max(first_k)
    rnd = []
    for t in texts:
        toks = T.mask(t).split(" ")
        s = rng.randint(0, max(1, len(toks) - win + 1))
        rnd.append(" ".join(toks[s:s + win]))
    view_inputs[f"random_window_{win}_tokens"] = (_word_model((1, 2), 1), rnd)
    profiles = []
    for t in texts:
        s = str(t)
        L = max(1, len(s))
        feats = [len(T.PUNCT_RE.findall(s)) / L, len(T.EMOJI_RE.findall(s)) / L * 20,
                 sum(ch.isupper() for ch in s) / L, sum(ch.isdigit() for ch in s) / L]
        profiles.append(" ".join(f"f{i}{_bucket(v)}" for i, v in enumerate(feats)))
    view_inputs["punctuation_emoji_case_profile"] = (_token_model(), profiles)
    lengths = np.array(T.n_tokens(texts))
    view_inputs["length_only"] = (_token_model(), _quantile_buckets(lengths, 10))
    for name, values in (metadata or {}).items():
        view_inputs[f"metadata:{name}"] = (_metadata_model(), ["v_" + str(v).replace(" ", "_") for v in values])

    views = {}
    for k, (model, X) in view_inputs.items():
        p = _oof(model, X, y, cv)
        f = _f1(y, p)
        r = recovery(f, full, chance)
        lo, hi = _boot_recovery(y, p, p_full, chance, seed)
        views[k] = {"macro_f1": round(f, 4),
                    "recovery_ratio": round(f / full, 4) if full else float("nan"),
                    "normalized_recovery": round(r, 4),
                    "normalized_recovery_ci95": [round(lo, 4), round(hi, 4)]}

    # the same reference with plain (ungrouped) folds, to show what grouping changes
    p_full_rand = _oof(_full_text_model(), texts, y, _folds(y, None, folds, seed))
    out = {
        **fold_info,
        "full_text_macro_f1": round(full, 4),
        "full_text_macro_f1_random_folds": round(_f1(y, p_full_rand), 4),
        "chance_macro_f1": round(chance, 4),
        "views": views,
    }
    if nc == 2:
        a, b = lengths[y == 0], lengths[y == 1]
        ks = ks_2samp(a, b)
        out["length_ks_D"] = round(float(ks.statistic), 4)
        out["length_ks_p"] = round(float(ks.pvalue), 4)
    scored = {k: v["normalized_recovery"] for k, v in views.items()
              if not np.isnan(v["normalized_recovery"]) and not k.startswith("random_window")}
    out["max_normalized_recovery"] = round(max(scored.values()), 4) if scored else float("nan")
    out["max_view"] = max(scored, key=scored.get) if scored else None
    return out


# --------------------------------------------------------------------------- layer 3
def _grouped_holdout(y: np.ndarray, groups: Optional[np.ndarray], seed: int, info: dict):
    tr, te = _folds(y, groups, 5, seed, info)[0]
    return tr, te


def layer3(texts: Sequence[str], y: np.ndarray, train_idx: Optional[np.ndarray] = None,
           test_idx: Optional[np.ndarray] = None, seed: int = 42, top_k: int = 40,
           groups: Optional[np.ndarray] = None) -> Dict:
    nc = len(np.unique(y))
    split_kind = "user"
    if train_idx is None or test_idx is None:
        info: Dict = {}
        train_idx, test_idx = _grouped_holdout(y, groups, seed, info)
        split_kind = info["folds"]

    sent_counts: List[Counter] = [Counter() for _ in range(nc)]
    for i in train_idx:
        for s in set(T.sentences(texts[i])):
            sent_counts[y[i]][s] += 1
    exclusive = []
    for c in range(nc):
        others = set().union(*[set(sent_counts[k]) for k in range(nc) if k != c]) if nc > 1 else set()
        exclusive.append({s for s in sent_counts[c] if s not in others})

    # Each test document goes to the class with the most distinct exclusive
    # sentences it contains (ties: lowest class index); documents with no hit
    # fall back to the training majority class.
    majority = int(np.bincount(y[train_idx], minlength=nc).argmax())
    preds, fallback = [], 0
    test_sents = {}
    for i in test_idx:
        st = set(T.sentences(texts[i]))
        test_sents[i] = st
        hits = [len(st & exclusive[c]) for c in range(nc)]
        if max(hits) > 0:
            preds.append(int(np.argmax(hits)))
        else:
            preds.append(majority)
            fallback += 1
    preds = np.array(preds)
    yt = y[test_idx]
    lookup_f1 = _f1(yt, preds)

    model = _full_text_model().fit([texts[i] for i in train_idx], y[train_idx])
    ref_pred = model.predict([texts[i] for i in test_idx])
    ref_f1 = _f1(yt, ref_pred)
    chance = 1.0 / nc
    r_lookup = recovery(lookup_f1, ref_f1, chance)

    rng = np.random.RandomState(seed)
    m_b, r_b = [], []
    for _ in range(BOOTSTRAP):
        b = rng.randint(0, len(yt), len(yt))
        fl, fr = _f1(yt[b], preds[b]), _f1(yt[b], ref_pred[b])
        m_b.append(fr - fl)
        r_b.append(recovery(fl, fr, chance))
    r_b = [v for v in r_b if not np.isnan(v)]

    inventory = {}
    for c in range(nc):
        ranked = sorted(((s, v) for s, v in sent_counts[c].items() if s in exclusive[c]), key=lambda kv: -kv[1])
        top = {s for s, _ in ranked[:top_k]}
        te_c = [i for i in test_idx if y[i] == c]
        inventory[str(c)] = {
            "sentence_types": len(sent_counts[c]),
            "exclusive_types": len(exclusive[c]),
            # share of held-out documents of this class that contain at least one
            # of the class's top-k most frequent exclusive training sentences
            f"top{top_k}_exclusive_coverage": round(
                sum(1 for i in te_c if test_sents[i] & top) / max(1, len(te_c)), 4),
        }
    out_split = {}
    if split_kind == "user" and groups is not None:
        # leakage of a published split: test documents whose duplicate group
        # (exact or verified near-duplicate) also has a member in training
        tr_groups = set(np.asarray(groups)[train_idx])
        out_split["user_split_test_overlap"] = round(
            float(np.mean([groups[i] in tr_groups for i in test_idx])), 4)
    return {
        "split": split_kind,
        **out_split,
        "test_size": int(len(test_idx)),
        "sentence_inventory": inventory,
        "lookup_rule_macro_f1": round(lookup_f1, 4),
        "lookup_fallback_rate": round(fallback / max(1, len(test_idx)), 4),
        "majority_constant_macro_f1": round(_f1(yt, np.full(len(yt), majority)), 4),
        "trained_reference_macro_f1": round(ref_f1, 4),
        "trained_reference_accuracy": round(float(accuracy_score(yt, ref_pred)), 4),
        "margin": round(ref_f1 - lookup_f1, 4),
        "margin_ci95": [round(float(np.percentile(m_b, 2.5)), 4), round(float(np.percentile(m_b, 97.5)), 4)],
        "normalized_lookup_recovery": round(r_lookup, 4),
        "normalized_lookup_recovery_ci95": ([round(float(np.percentile(r_b, 2.5)), 4),
                                             round(float(np.percentile(r_b, 97.5)), 4)] if r_b else None),
    }
