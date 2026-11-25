"""Additional experiments for the LeakAudit journal article.

E1  split inflation: random stratified split vs. duplicate-aware grouped split
E2  near-duplicate threshold sensitivity (Jaccard 0.7 / 0.8 / 0.9)
E3  masking ablation: template uniqueness before and after entity masking
E4  layer-1 runtime scaling on the product-review corpus

Usage: python run_experiments.py <data_dir> <out.json>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from leakaudit import text as T
from leakaudit.layers import _full_text_model, layer1
from leakaudit.minhash import near_duplicate_pairs

SEED = 42
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results.json")
np.seterr(all="ignore")


def load():
    v1 = pd.read_csv(DATA / "turkish_phishing_dataset.csv")
    v1["text"] = (v1["Konu"].fillna("") + "\n" + v1["İçerik"].fillna("")).str.strip()
    v2 = pd.read_csv(DATA / "turkphish_v2.csv")
    v2["text"] = (v2["konu"].fillna("") + "\n" + v2["icerik"].fillna("")).str.strip()
    ttc = pd.read_parquet(DATA / "ttc4900.parquet")
    rev = pd.read_parquet(DATA / "urun_yorum.parquet").rename(columns={"sentence": "text"})
    sen = pd.read_parquet(DATA / "duygu_test.parquet")
    return {
        "TurkPhish v1": (v1, "text", "Kategori"),
        "TurkPhish v2": (v2, "text", "kategori"),
        "TTC-4900": (ttc, "text", "category"),
        "Product reviews": (rev, "text", "sentiment"),
        "Sentiment (test)": (sen, "text", "label"),
    }


def clean(df, text, label):
    d = df[df[text].notna() & df[label].notna()].copy()
    d[text] = d[text].astype(str)
    d = d[d[text].str.strip() != ""].reset_index(drop=True)
    classes = sorted(d[label].astype(str).unique())
    y = d[label].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    return d[text].tolist(), y


def groups_for(texts, threshold=0.8):
    """Union-find over exact masked duplicates and verified near-duplicate pairs."""
    n = len(texts)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first = {}
    for i, m in enumerate(T.mask(t) for t in texts):
        if m in first:
            union(first[m], i)
        else:
            first[m] = i
    pairs, _ = near_duplicate_pairs([T.shingles(t) for t in texts], threshold=threshold)
    for a, b, _ in pairs:
        union(a, b)
    return np.array([find(i) for i in range(n)])


def split_inflation(texts, y, max_n=50000):
    rng = np.random.RandomState(SEED)
    if len(texts) > max_n:
        idx = np.sort(rng.permutation(len(texts))[:max_n])
        texts, y = [texts[i] for i in idx], y[idx]
    g = groups_for(texts)
    out = {"n": len(texts), "n_groups": int(len(np.unique(g)))}

    tr, te = train_test_split(np.arange(len(texts)), test_size=0.2, random_state=SEED, stratify=y)
    train_groups = set(g[tr])
    out["random_test_with_group_in_train"] = round(float(np.mean([g[i] in train_groups for i in te])), 4)
    m = _full_text_model().fit([texts[i] for i in tr], y[tr])
    out["random_split_macro_f1"] = round(float(f1_score(y[te], m.predict([texts[i] for i in te]), average="macro")), 4)

    # grouped split: retry seeds until every class appears in both sides
    for s in range(SEED, SEED + 50):
        gtr, gte = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s).split(texts, y, g))
        if len(set(y[gte])) == len(set(y)) and len(set(y[gtr])) == len(set(y)):
            break
    else:
        out["grouped_split_macro_f1"] = None
        return out
    m = _full_text_model().fit([texts[i] for i in gtr], y[gtr])
    out["grouped_split_macro_f1"] = round(float(f1_score(y[gte], m.predict([texts[i] for i in gte]), average="macro")), 4)
    out["grouped_split_seed"] = s
    out["grouped_test_size"] = int(len(gte))
    out["inflation"] = round(out["random_split_macro_f1"] - out["grouped_split_macro_f1"], 4)
    return out


def threshold_sweep(texts, cap=50000):
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(texts))[:min(cap, len(texts))]
    sh = [T.shingles(texts[i]) for i in idx]
    pairs, best = near_duplicate_pairs(sh, threshold=0.7)
    out = {f"pairs_j{t}": int(sum(j >= t for _, _, j in pairs)) for t in (0.7, 0.8, 0.9)}
    out["sample"] = int(len(idx))
    return out


def masking_ablation(texts):
    raw = len({T.WS_RE.sub(" ", T.tr_lower(t)).strip() for t in texts})
    masked = len({T.mask(t) for t in texts})
    return {"lowercase_only": round(raw / len(texts), 4), "masked": round(masked / len(texts), 4)}


def runtime_scaling(texts, y, sizes=(25000, 50000, 100000, 235144)):
    rng = np.random.RandomState(SEED)
    res = []
    for n in sizes:
        n = min(n, len(texts))
        idx = np.sort(rng.permutation(len(texts))[:n])
        t0 = time.time()
        layer1([texts[i] for i in idx], y[idx], near_dup_cap=n)
        res.append({"n": n, "seconds": round(time.time() - t0, 1)})
    return res


def main():
    results = {"seed": SEED}
    corpora = load()
    for name, (df, tcol, lcol) in corpora.items():
        texts, y = clean(df, tcol, lcol)
        print(f"[{name}] n={len(texts)}", flush=True)
        r = {"masking": masking_ablation(texts)}
        if name != "TurkPhish v2":
            r["threshold_sweep"] = threshold_sweep(texts)
            r["split_inflation"] = split_inflation(texts, y)
        results[name] = r
        print(json.dumps(r), flush=True)

    texts, y = clean(*corpora["Product reviews"])
    results["runtime_layer1_product_reviews"] = runtime_scaling(texts, y)
    print(json.dumps(results["runtime_layer1_product_reviews"]), flush=True)
    OUT.write_text(json.dumps(results, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
