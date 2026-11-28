"""Experiments added in the pre-submission revision (leakaudit 0.2.0).

E5  paired split inflation: on the same random test set, a model trained on the
    full training part vs. one trained after removing every training document
    that shares a duplicate group with a test document. Logistic regression and
    1-nearest-neighbour; five seeds; Nadeau-Bengio corrected interval. Also
    reports macro-F1 on test documents with and without a training duplicate.
E6  injection controls on a clean base (TurkPhish v2 body, deduplicated): planted
    duplicates, a planted opening token, a planted metadata column, and the same
    planted opening under several class balances (comparability of R).
E7  provenance column of the aggregated sentiment corpus: can the source be
    predicted from the text, and full-text macro-F1 within each source.

Usage: python exp_r1.py <data_dir> <out.json>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import clean, load  # noqa: E402
from leakaudit import text as T  # noqa: E402
from leakaudit.layers import _full_text_model, layer1, layer2, recovery  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
np.seterr(all="ignore")


def knn_model():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=T.mask, token_pattern=r"\S+", ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=T.mask, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 sublinear_tf=True)),
    ])
    return Pipeline([("features", feats), ("clf", KNeighborsClassifier(n_neighbors=1, metric="cosine"))])


def f1(y, p):
    return float(f1_score(y, p, average="macro"))


def nb_interval(d, n_train, n_test):
    """Nadeau-Bengio corrected resampled t interval (95%) for the mean of d."""
    d = np.asarray(d, float)
    J = len(d)
    var = np.var(d, ddof=1) * (1 / J + n_test / n_train)
    h = stats.t.ppf(0.975, J - 1) * np.sqrt(var)
    return [round(float(d.mean() - h), 4), round(float(d.mean() + h), 4)]


def e5_inflation(texts, y, max_n=50000):
    if len(texts) > max_n:
        idx = np.sort(np.random.RandomState(42).permutation(len(texts))[:max_n])
        texts, y = [texts[i] for i in idx], y[idx]
    g = layer1(texts, y, near_dup_cap=len(texts))["_groups"]
    res = {"n": len(texts), "runs": []}
    for s in SEEDS:
        tr, te = train_test_split(np.arange(len(texts)), test_size=0.2, random_state=s, stratify=y)
        te_groups = set(g[te])
        tr_groups = set(g[tr])
        overlap = np.array([g[i] in tr_groups for i in te])
        tr_clean = np.array([i for i in tr if g[i] not in te_groups])
        row = {"seed": s, "overlap": round(float(overlap.mean()), 4),
               "train_removed": int(len(tr) - len(tr_clean))}
        if len(tr_clean) == 0 or len(np.unique(y[tr_clean])) < len(np.unique(y)):
            row["infeasible"] = "removing duplicates of the test part leaves a class without training documents"
            res["runs"].append(row)
            print("E5", s, row, flush=True)
            continue
        for name, mk in (("lr", _full_text_model), ("knn1", knn_model)):
            pa = mk().fit([texts[i] for i in tr], y[tr]).predict([texts[i] for i in te])
            pb = mk().fit([texts[i] for i in tr_clean], y[tr_clean]).predict([texts[i] for i in te])
            row[name] = {"full_train": round(f1(y[te], pa), 4), "dedup_train": round(f1(y[te], pb), 4),
                         "inflation_pts": round(100 * (f1(y[te], pa) - f1(y[te], pb)), 3)}
            if overlap.any() and (~overlap).any():
                row[name]["acc_overlapping"] = round(float((pa[overlap] == y[te][overlap]).mean()), 4)
                row[name]["acc_non_overlapping"] = round(float((pa[~overlap] == y[te][~overlap]).mean()), 4)
        res["runs"].append(row)
        print("E5", s, row, flush=True)
    n_te = len(texts) // 5
    for name in ("lr", "knn1"):
        d = [r[name]["inflation_pts"] for r in res["runs"] if name in r]
        if len(d) < 2:
            continue
        res[name] = {"inflation_mean_pts": round(float(np.mean(d)), 3), "inflation_sd_pts": round(float(np.std(d, ddof=1)), 3),
                     "nb_ci95_pts": nb_interval(d, len(texts) - n_te, n_te)}
    return res


def _base_v2_body(corpora):
    df, _, lcol = corpora["TurkPhish v2"]
    d = df[df["icerik"].notna()].copy()
    d["text"] = d["icerik"].astype(str)
    d = d.drop_duplicates(subset="text").reset_index(drop=True)
    classes = sorted(d[lcol].astype(str).unique())
    y = d[lcol].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    return d["text"].tolist(), y


def e6_controls(corpora, seed=42):
    texts, y = _base_v2_body(corpora)
    rng = np.random.RandomState(seed)
    out = {"base_n": len(texts)}
    base = layer1(texts, y, near_dup_cap=len(texts))
    out["base_layer1"] = {k: base[k] for k in ("template_uniqueness", "near_dup_pairs", "label_conflicting_duplicate_groups")}

    dup = []
    for rate in (0.01, 0.05, 0.10):
        k = int(rate * len(texts))
        pick = rng.choice(len(texts), k, replace=False)
        t2, y2 = texts + [texts[i] for i in pick], np.concatenate([y, y[pick]])
        r = layer1(t2, y2, near_dup_cap=len(t2))
        dup.append({"rate": rate, "injected": k, "template_uniqueness": r["template_uniqueness"],
                    "near_dup_pairs": r["near_dup_pairs"], "duplicate_groups": r["duplicate_groups"]})
    out["planted_duplicates"] = dup

    g = layer1(texts, y, near_dup_cap=len(texts))["_groups"]
    op = []
    for p in (0.0, 0.1, 0.25, 0.5, 1.0):
        t2 = [("kodx " + t) if (y[i] == 1 and rng.rand() < p) else t for i, t in enumerate(texts)]
        L2 = layer2(t2, y, first_k=(3,), seed=seed, groups=g)
        v = L2["views"]["first_3_tokens"]
        op.append({"p": p, "first3_R": v["normalized_recovery"], "first3_R_ci95": v["normalized_recovery_ci95"],
                   "full": L2["full_text_macro_f1"]})
        print("E6 opening", op[-1], flush=True)
    out["planted_opening"] = op

    md = []
    for q in (0.5, 0.6, 0.75, 0.9, 1.0):
        col = [str(c) if rng.rand() < q else str(1 - c) for c in y]
        L2 = layer2(texts, y, {"src": col}, first_k=(3,), seed=seed, groups=g)
        v = L2["views"]["metadata:src"]
        md.append({"agreement": q, "R": v["normalized_recovery"], "R_ci95": v["normalized_recovery_ci95"],
                   "raw_ratio": v["recovery_ratio"]})
        print("E6 metadata", md[-1], flush=True)
    out["planted_metadata"] = md

    # comparability: same planted opening (p = 0.5 within class 1), different class balance
    bal = []
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    for frac1 in (0.5, 0.2, 0.06):
        n1 = int(len(i0) * frac1 / (1 - frac1))
        keep = np.sort(np.concatenate([i0, rng.choice(i1, min(n1, len(i1)), replace=False)]))
        tk, yk, gk = [texts[i] for i in keep], y[keep], g[keep]
        t2 = [("kodx " + t) if (yk[j] == 1 and rng.rand() < 0.5) else t for j, t in enumerate(tk)]
        L2 = layer2(t2, yk, first_k=(3,), seed=seed, groups=gk)
        v = L2["views"]["first_3_tokens"]
        maj = max(np.mean(yk), 1 - np.mean(yk))
        maj_f1 = maj / (1 + maj) / 1  # macro-F1 of the majority constant for C = 2
        bal.append({"minority_share": round(float(min(np.mean(yk), 1 - np.mean(yk))), 3), "n": len(tk),
                    "R": v["normalized_recovery"], "raw_ratio": v["recovery_ratio"],
                    "majority_based": round(recovery(v["macro_f1"], L2["full_text_macro_f1"], maj_f1), 4)})
        print("E6 balance", bal[-1], flush=True)
    out["balance"] = bal
    return out


def e7_provenance(corpora, seed=42, cap=20000):
    df, tcol, lcol = corpora["Sentiment (test)"]
    d = df[df[tcol].notna() & df[lcol].notna() & df["dataset"].notna()].copy().reset_index(drop=True)
    if len(d) > cap:
        d = d.iloc[np.sort(np.random.RandomState(seed).permutation(len(d))[:cap])].reset_index(drop=True)
    texts = d[tcol].astype(str).tolist()
    src = d["dataset"].astype(str)
    srcs = sorted(src.unique())
    ys = src.map({c: i for i, c in enumerate(srcs)}).to_numpy()
    tr, te = train_test_split(np.arange(len(d)), test_size=0.2, random_state=seed, stratify=ys)
    m = _full_text_model().fit([texts[i] for i in tr], ys[tr])
    out = {"n": len(d), "sources": {}, "source_from_text_macro_f1": round(f1(ys[te], m.predict([texts[i] for i in te])), 4),
           "source_chance": round(1 / len(srcs), 4)}
    labels = d[lcol].astype(str)
    for s in srcs:
        idx = np.where(src == s)[0]
        lab = labels.iloc[idx]
        info = {"n": int(len(idx)), "label_shares": {k: round(v, 3) for k, v in lab.value_counts(normalize=True).items()}}
        if lab.nunique() > 1 and lab.value_counts().min() >= 5:
            yy = lab.map({c: i for i, c in enumerate(sorted(lab.unique()))}).to_numpy()
            a, b = train_test_split(idx, test_size=0.2, random_state=seed, stratify=yy)
            ya = labels.iloc[a].map({c: i for i, c in enumerate(sorted(lab.unique()))}).to_numpy()
            yb = labels.iloc[b].map({c: i for i, c in enumerate(sorted(lab.unique()))}).to_numpy()
            mm = _full_text_model().fit([texts[i] for i in a], ya)
            info["within_source_macro_f1"] = round(f1(yb, mm.predict([texts[i] for i in b])), 4)
        out["sources"][s] = info
    return out


def main():
    corpora = load()
    res = {"leakaudit_version": "0.2.0"}
    t0 = time.time()
    res["E5"] = {}
    for name in ["TTC-4900", "Product reviews", "Sentiment (test)", "TurkPhish v1"]:
        texts, y = clean(*corpora[name])
        res["E5"][name] = e5_inflation(texts, y)
    res["E6"] = e6_controls(corpora)
    res["E7"] = e7_provenance(corpora)
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print("saved", OUT, res["seconds"], "s")


if __name__ == "__main__":
    main()
