"""R10: specificity of the layer 2 gates, an n-gram lookup and refitted variability
of the opening excess (AI-referee round 6).

(A) Null false alarms. Every report stores, for each view, the macro-F1 of the same
    view model refitted on permuted labels with the same folds. The recovery of that
    null, (N_v - 1/C) / (F_full - 1/C), is what a view without any label signal
    scores. A sound gate should not flag it: we count how many null views exceed the
    content-aware limit for other views (R <= 0.50) and how many null openings exceed
    the opening limit (excess <= 0.25). One permutation per view; no model is refitted.
(B) N-gram lookup. Layer 3 matches whole sentences. As a softer learning-free variant,
    each test document is assigned to the class whose class-exclusive word 4-grams
    (masked tokens, seen in at least two training documents of that class and in no
    other class) it shares most; no hit falls back to the training majority class.
    Same grouped hold-out split and reference model as layer 3.
(C) Refitted variability of the opening excess. The bootstrap intervals hold the
    models fixed. Here the whole opening check (folds, random windows and all models)
    is refitted under 10 different seeds, which gives the spread of the excess that
    comes from fold assignment and window placement.

Usage: python exp_r10_specificity.py <data_dir> <out.json>
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import clean, load  # noqa: E402
from leakaudit import text as T  # noqa: E402
from leakaudit.layers import (layer1, _folds, _oof, _word_model, _full_text_model, _f1,  # noqa: E402
                              recovery, _grouped_holdout)

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ROOT = DATA.resolve().parent
np.seterr(all="ignore")
SEED, CAP, KS, NSEEDS, N = 42, 20000, (3, 5, 10), 10, 4
OTHER_LIMIT, OPEN_LIMIT = 0.50, 0.25
REPS = {"TurkPhish v1": "turkphish_v1", "TurkPhish v2": "turkphish_v2", "TTC-4900": "ttc4900",
        "Product reviews": "urun_yorum", "Sentiment (test)": "duygu", "AG News": "agnews"}


def part_a():
    out, other, opening = {}, [], []
    for name, key in REPS.items():
        rep = json.loads((ROOT / "reports" / f"{key}.json").read_text())
        L2 = rep["layer2"]
        full, ch = L2["full_text_macro_f1"], L2["chance_macro_f1"]
        rn = {v: recovery(d["permuted_label_macro_f1"], full, ch) for v, d in L2["views"].items()}
        oth = {v: r for v, r in rn.items() if not v.startswith(("first_", "random_window_"))}
        exc = max(rn[f"first_{k}_tokens"] - rn[f"random_window_{k}_tokens"] for k in KS
                  if f"first_{k}_tokens" in rn)
        other += list(oth.values()); opening.append(exc)
        out[name] = {"null_other_R_max": round(max(oth.values()), 4), "null_opening_excess": round(exc, 4)}
    out["summary"] = {"n_other": len(other), "other_flagged": int(sum(r > OTHER_LIMIT for r in other)),
                      "other_max": round(max(other), 4), "n_opening": len(opening),
                      "opening_flagged": int(sum(e > OPEN_LIMIT for e in opening)),
                      "opening_max": round(max(opening), 4)}
    return out


def grams(t):
    toks = [w for w in T.mask(t).split(" ") if w]
    return {" ".join(toks[i:i + N]) for i in range(len(toks) - N + 1)}


def ngram_lookup(texts, y, groups):
    info = {}
    tr, te = _grouped_holdout(y, groups, SEED, info)
    nc = len(np.unique(y))
    cnt = [Counter() for _ in range(nc)]
    for i in tr:
        cnt[y[i]].update(grams(texts[i]))
    seen = Counter()
    for c in range(nc):
        seen.update(set(cnt[c]))
    excl = [{g for g, v in cnt[c].items() if v >= 2 and seen[g] == 1} for c in range(nc)]
    maj = int(np.bincount(y[tr], minlength=nc).argmax())
    pred, fb = [], 0
    for i in te:
        g = grams(texts[i])
        h = [len(g & excl[c]) for c in range(nc)]
        if max(h) > 0:
            pred.append(int(np.argmax(h)))
        else:
            pred.append(maj); fb += 1
    yt = y[te]
    lf = _f1(yt, np.array(pred))
    ref = _full_text_model().fit([texts[i] for i in tr], y[tr]).predict([texts[i] for i in te])
    rf = _f1(yt, ref)
    return {"fallback": round(fb / len(te), 4), "lookup_f1": round(lf, 4), "reference_f1": round(rf, 4),
            "RL": round(recovery(lf, rf, 1.0 / nc), 4), "split": info.get("folds")}


def window_views(texts, seed):
    rng = np.random.RandomState(seed)
    v = {f"first_{k}": [T.first_tokens(t, k) for t in texts] for k in KS}
    for k in KS:
        o = []
        for t in texts:
            toks = T.mask(t).split(" ")
            s = rng.randint(0, max(1, len(toks) - k + 1))
            o.append(" ".join(toks[s:s + k]))
        v[f"window_{k}"] = o
    return v


def opening_excess(texts, y, groups, seed):
    cv = _folds(y, groups, 5, seed, {})
    ch = 1.0 / len(np.unique(y))
    full = _f1(y, _oof(_full_text_model(), texts, y, cv))
    P = {k: _f1(y, _oof(_word_model((1, 2), 1), X, y, cv)) for k, X in window_views(texts, seed).items()}
    return max(recovery(P[f"first_{k}"], full, ch) - recovery(P[f"window_{k}"], full, ch) for k in KS)


def main():
    res = {"seed": SEED, "ngram": N, "refit_seeds": NSEEDS, "A_null": part_a()}
    print("A", json.dumps(res["A_null"]["summary"]), flush=True)
    corpora = load()
    ag = pd.read_parquet(DATA / "agnews_train.parquet")
    jobs = [(n, *corpora[n], "tr") for n in corpora] + [("AG News", ag, "text", "label", "generic")]
    res["B_ngram"], res["C_refit"] = {}, {}
    for name, df, tcol, lcol, lang in jobs:
        T.set_language(lang)
        ta, ya = clean(df, tcol, lcol)
        ga = layer1(ta, ya, 50000, 0.8, SEED)["_groups"]
        idx = np.arange(len(ta))
        if len(idx) > CAP:
            idx = np.sort(np.random.RandomState(SEED).permutation(len(idx))[:CAP])
        texts, y, g = [ta[i] for i in idx], ya[idx], ga[idx]
        res["B_ngram"][name] = ngram_lookup(texts, y, g)
        print("B", name, json.dumps(res["B_ngram"][name]), flush=True)
        if name in ("TurkPhish v2", "TTC-4900", "AG News"):
            ex = [opening_excess(texts, y, g, s) for s in range(NSEEDS)]
            res["C_refit"][name] = {"values": [round(float(e), 4) for e in ex], "mean": round(float(np.mean(ex)), 4),
                                    "min": round(float(np.min(ex)), 4), "max": round(float(np.max(ex)), 4),
                                    "n_above": int(sum(e > OPEN_LIMIT for e in ex))}
            print("C", name, json.dumps(res["C_refit"][name]), flush=True)
        OUT.write_text(json.dumps(res, indent=2))
    OUT.write_text(json.dumps(res, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
