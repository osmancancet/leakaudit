"""R12: the size-matched inflation measurement of R2a repeated on two English corpora
that the content-aware profile flags for near-duplicates (AI-referee round 8).

AG News and IMDB training splits, audited with --lang generic; AG News is capped at a
seeded sample of 20,000 documents to bound the 1-nearest-neighbour cost. Same five
seeds, splits, models and Nadeau-Bengio intervals as R2a.

Usage: python exp_r12_inflation_en.py <data_dir> <out.json> [IMDB|AG News]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
ONLY = sys.argv[3] if len(sys.argv) > 3 else None
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
from exp_r2 import r2a_size_matched  # noqa: E402
from run_experiments import clean  # noqa: E402
from leakaudit import text as T  # noqa: E402

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
CAP = 20000


def main():
    T.set_language("generic")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["cap"] = CAP
    for name, f in (("IMDB", "imdb_train.parquet"), ("AG News", "agnews_train.parquet")):
        if ONLY and name != ONLY:
            continue
        df = pd.read_parquet(DATA / f)
        t, y = clean(df, "text", "label")
        if len(t) > CAP:
            idx = np.sort(np.random.RandomState(42).permutation(len(t))[:CAP])
            t, y = [t[i] for i in idx], y[idx]
        res[name] = r2a_size_matched(t, y)
        print(name, json.dumps({k: res[name][k] for k in ("lr", "knn1")}), flush=True)
        OUT.write_text(json.dumps(res, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
