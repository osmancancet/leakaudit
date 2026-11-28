"""R8: negative control and interval for the opening check (AI-referee round 5).

For every audited corpus the opening excess (largest, over k in {3, 5, 10}, of the
recovery of the first-k view minus that of a random k-token window) is computed
(a) on the corpus as audited and (b) on a position-shuffled copy in which the tokens
of every document are put in random order. Shuffling keeps each document's words,
and with them the class signal, but removes any positional concentration, so a
sound opening check should not flag the shuffled copies: the share flagged is an
empirical false-alarm rate. Intervals are 95% group-level bootstrap intervals
(1,000 resamples of duplicate groups, models fixed, all views resampled together).

Usage: python exp_r8_position_control.py <data_dir> <out.json>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import clean, load  # noqa: E402
from leakaudit import text as T  # noqa: E402
from leakaudit.layers import (layer1, _folds, _oof, _word_model, _full_text_model, _f1,  # noqa: E402
                              recovery, _boot_index)

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
np.seterr(all="ignore")
SEED, CAP, KS, B, LIMIT = 42, 20000, (3, 5, 10), 1000, 0.25


def views(texts, seed):
    v = {f"first_{k}": [T.first_tokens(t, k) for t in texts] for k in KS}
    rng = np.random.RandomState(seed)
    for k in KS:
        out = []
        for t in texts:
            toks = T.mask(t).split(" ")
            s = rng.randint(0, max(1, len(toks) - k + 1))
            out.append(" ".join(toks[s:s + k]))
        v[f"window_{k}"] = out
    return v


def excess(y, P, full_p, chance, idx=None):
    if idx is not None:
        y = y[idx]; P = {k: p[idx] for k, p in P.items()}; full_p = full_p[idx]
    full = _f1(y, full_p)
    return max(recovery(_f1(y, P[f"first_{k}"]), full, chance) - recovery(_f1(y, P[f"window_{k}"]), full, chance)
               for k in KS)


def measure(texts, y, groups):
    cv = _folds(y, groups, 5, SEED, {})
    chance = 1.0 / len(np.unique(y))
    full_p = _oof(_full_text_model(), texts, y, cv)
    P = {k: _oof(_word_model((1, 2), 1), X, y, cv) for k, X in views(texts, SEED).items()}
    point = excess(y, P, full_p, chance)
    rng = np.random.RandomState(SEED)
    boots = [excess(y, P, full_p, chance, _boot_index(rng, len(y), groups)) for _ in range(B)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"excess": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "flag_point": bool(point > LIMIT), "flag_lower_bound": bool(lo > LIMIT)}


def shuffled(texts, seed):
    rng = np.random.RandomState(seed)
    out = []
    for t in texts:
        toks = T.mask(t).split(" ")
        rng.shuffle(toks)
        out.append(" ".join(toks))
    return out


def main():
    corpora = load()
    ag = pd.read_parquet(DATA / "agnews_train.parquet")
    jobs = [(n, *corpora[n], "tr") for n in corpora] + [("AG News", ag, "text", "label", "generic")]
    res = {"seed": SEED, "bootstrap": B, "limit": LIMIT}
    for name, df, tcol, lcol, lang in jobs:
        T.set_language(lang)
        texts_all, y_all = clean(df, tcol, lcol)
        g_all = layer1(texts_all, y_all, 50000, 0.8, SEED)["_groups"]
        idx = np.arange(len(texts_all))
        if len(idx) > CAP:
            idx = np.sort(np.random.RandomState(SEED).permutation(len(idx))[:CAP])
        texts, y, g = [texts_all[i] for i in idx], y_all[idx], g_all[idx]
        r = {"n": int(len(texts)), "original": measure(texts, y, g), "shuffled": measure(shuffled(texts, SEED), y, g)}
        res[name] = r
        print(name, json.dumps(r), flush=True)
    sh = [res[n]["shuffled"] for n, *_ in jobs]
    res["false_alarms_point"] = int(sum(s["flag_point"] for s in sh))
    res["false_alarms_lower_bound"] = int(sum(s["flag_lower_bound"] for s in sh))
    res["n_corpora"] = len(sh)
    OUT.write_text(json.dumps(res, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
