"""Split inflation over five seeds: random stratified vs. duplicate-aware grouped split.

Groups (exact masked duplicates + verified near-duplicates) are computed once per corpus.
Usage: python run_inflation_seeds.py <data_dir> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import clean, groups_for, load  # noqa: E402
from leakaudit.layers import _full_text_model  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
np.seterr(all="ignore")


def f1_on(texts, y, tr, te):
    m = _full_text_model().fit([texts[i] for i in tr], y[tr])
    return float(f1_score(y[te], m.predict([texts[i] for i in te]), average="macro"))


res = {}
corpora = load()
for name in ["TTC-4900", "Product reviews", "Sentiment (test)"]:
    texts, y = clean(*corpora[name])
    if len(texts) > 50000:
        idx = np.sort(np.random.RandomState(42).permutation(len(texts))[:50000])
        texts, y = [texts[i] for i in idx], y[idx]
    g = groups_for(texts)
    rows = []
    for s in SEEDS:
        tr, te = train_test_split(np.arange(len(texts)), test_size=0.2, random_state=s, stratify=y)
        overlap = float(np.mean([gg in set(g[tr]) for gg in g[te]]))
        f_r = f1_on(texts, y, tr, te)
        for k in range(s, s + 50):
            gtr, gte = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=k).split(texts, y, g))
            if len(set(y[gte])) == len(set(y)):
                break
        f_g = f1_on(texts, y, gtr, gte)
        rows.append({"seed": s, "overlap": round(overlap, 4), "random": round(f_r, 4), "grouped": round(f_g, 4),
                     "inflation": round(f_r - f_g, 4)})
        print(name, rows[-1], flush=True)
    infl = np.array([r["inflation"] for r in rows])
    res[name] = {"runs": rows, "inflation_mean": round(float(infl.mean()), 4),
                 "inflation_sd": round(float(infl.std(ddof=1)), 4),
                 "random_mean": round(float(np.mean([r["random"] for r in rows])), 4),
                 "grouped_mean": round(float(np.mean([r["grouped"] for r in rows])), 4),
                 "overlap_mean": round(float(np.mean([r["overlap"] for r in rows])), 4)}
OUT.write_text(json.dumps(res, indent=2))
print(json.dumps({k: {kk: v[kk] for kk in v if kk != "runs"} for k, v in res.items()}, indent=1))
