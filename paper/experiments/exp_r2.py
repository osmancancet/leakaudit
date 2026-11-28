"""Experiments added after the second internal review (leakaudit 0.2.0).

R2a  size-matched control for E5 (TTC-4900): on the same splits, remove from the
     training part as many random documents, with the same class composition, as
     the duplicate removal does; the inflation attributable to duplication is
     F1(random removal) - F1(duplicate removal).
R2b  injected near-duplicates: copies of base documents with a share of tokens
     replaced; ground truth is the exact character 5-shingle Jaccard.
R2c  injected label conflicts: exact copies with the opposite label.
R2d  injected sentence inventory: one class-specific sentence (out of 20 per
     class) appended to a share p of the documents of every class (TTC-4900).
R2e  class-balance experiment of E6 repeated with three further random draws.
R2f  short documents: share of documents with <= 10 tokens, and the first-10
     minus random-window excess recomputed on documents longer than 20 tokens.
R2g  TTC-4900 sentence lookup under a random (non-grouped) split.

Usage: python exp_r2.py <data_dir> <out.json>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import clean, load  # noqa: E402
from exp_r1 import SEEDS, _base_v2_body, f1, knn_model, nb_interval  # noqa: E402
from leakaudit import text as T  # noqa: E402
from leakaudit.layers import _full_text_model, layer1, layer2, layer3  # noqa: E402
from leakaudit.minhash import jaccard  # noqa: E402

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
np.seterr(all="ignore")


def r2a_size_matched(texts, y):
    g = layer1(texts, y, near_dup_cap=len(texts))["_groups"]
    res = {"n": len(texts), "runs": []}
    for s in SEEDS:
        rng = np.random.RandomState(1000 + s)
        tr, te = train_test_split(np.arange(len(texts)), test_size=0.2, random_state=s, stratify=y)
        te_groups = set(g[te])
        removed = np.array([i for i in tr if g[i] in te_groups])
        tr_dedup = np.array([i for i in tr if g[i] not in te_groups])
        # random removal with the same number of documents per class
        drop = []
        for c in np.unique(y):
            k = int((y[removed] == c).sum()) if len(removed) else 0
            pool = tr[y[tr] == c]
            drop.extend(rng.choice(pool, k, replace=False).tolist())
        drop = set(drop)
        tr_rand = np.array([i for i in tr if i not in drop])
        row = {"seed": s, "removed": int(len(removed)), "removed_share": round(len(removed) / len(tr), 4)}
        for name, mk in (("lr", _full_text_model), ("knn1", knn_model)):
            yt = y[te]
            Xte = [texts[i] for i in te]
            fa = f1(yt, mk().fit([texts[i] for i in tr], y[tr]).predict(Xte))
            fd = f1(yt, mk().fit([texts[i] for i in tr_dedup], y[tr_dedup]).predict(Xte))
            fr = f1(yt, mk().fit([texts[i] for i in tr_rand], y[tr_rand]).predict(Xte))
            row[name] = {"full": round(fa, 4), "dedup": round(fd, 4), "random_removed": round(fr, 4),
                         "size_effect_pts": round(100 * (fa - fr), 3), "duplication_effect_pts": round(100 * (fr - fd), 3)}
        res["runs"].append(row)
        print("R2a", row, flush=True)
    n_te = len(texts) // 5
    for name in ("lr", "knn1"):
        d = [r[name]["duplication_effect_pts"] for r in res["runs"]]
        e = [r[name]["size_effect_pts"] for r in res["runs"]]
        res[name] = {"duplication_effect_mean_pts": round(float(np.mean(d)), 3),
                     "duplication_effect_ci95_pts": nb_interval(d, len(texts) - n_te, n_te),
                     "size_effect_mean_pts": round(float(np.mean(e)), 3),
                     "size_effect_ci95_pts": nb_interval(e, len(texts) - n_te, n_te)}
    return res


def _perturb(t, share, rng, vocab):
    tok = t.split()
    for j in range(len(tok)):
        if rng.rand() < share:
            tok[j] = vocab[rng.randint(len(vocab))]
    return " ".join(tok)


def r2b_near_dups(texts, y, k=100, seed=7):
    rng = np.random.RandomState(seed)
    vocab = sorted({w for t in texts for w in t.split()})
    out = []
    for share in (0.02, 0.05, 0.10, 0.20):
        pick = rng.choice(len(texts), k, replace=False)
        planted = [_perturb(texts[i], share, rng, vocab) for i in pick]
        true_j = [jaccard(T.shingles(T.mask(texts[i])), T.shingles(T.mask(p))) for i, p in zip(pick, planted)]
        ident = sum(T.mask(texts[i]) == T.mask(p) for i, p in zip(pick, planted))
        r = layer1(texts + planted, np.concatenate([y, y[pick]]), near_dup_cap=len(texts) + k)
        out.append({"token_replacement": share, "planted": k, "identical_after_masking": int(ident),
                    "true_pairs_j08": int(sum(j >= 0.8 for j in true_j)),
                    "mean_true_jaccard": round(float(np.mean(true_j)), 3),
                    "detected_pairs": r["near_dup_pairs"], "detected_non_identical": r["near_dup_pairs_non_identical"]})
        print("R2b", out[-1], flush=True)
    return out


def r2c_conflicts(texts, y, seed=8):
    rng = np.random.RandomState(seed)
    out = []
    for k in (5, 20, 50):
        pick = rng.choice(len(texts), k, replace=False)
        r = layer1(texts + [texts[i] for i in pick], np.concatenate([y, 1 - y[pick]]), near_dup_cap=len(texts) + k)
        out.append({"planted": k, "label_conflicting_groups": r["label_conflicting_duplicate_groups"]})
        print("R2c", out[-1], flush=True)
    return out


def r2d_inventory(texts, y, seed=9):
    rng = np.random.RandomState(seed)
    nc = len(np.unique(y))
    words = sorted({w for t in texts[:500] for w in t.split() if len(w) > 3})
    inv = {c: [" ".join(rng.choice(words, 6)) + " kodx" + str(c) + "." for _ in range(20)] for c in range(nc)}
    out = []
    for p in (0.0, 0.05, 0.1, 0.25, 0.5):
        # sentences are split at end punctuation, so the planted sentence follows a full stop
        t2 = [t.rstrip(" .") + ". " + inv[y[i]][rng.randint(20)] if rng.rand() < p else t for i, t in enumerate(texts)]
        g = layer1(t2, y, near_dup_cap=len(t2))["_groups"]
        L3 = layer3(t2, y, seed=42, groups=g)
        out.append({"p": p, "lookup": L3["lookup_rule_macro_f1"], "fallback": L3["lookup_fallback_rate"],
                    "R_L": L3["normalized_lookup_recovery"], "R_L_ci95": L3["normalized_lookup_recovery_ci95"]})
        print("R2d", out[-1], flush=True)
    return out


def r2e_balance(texts, y, g):
    out = []
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    for seed in (101, 102, 103):
        rng = np.random.RandomState(seed)
        for frac1 in (0.5, 0.2, 0.06):
            n1 = int(len(i0) * frac1 / (1 - frac1))
            keep = np.sort(np.concatenate([i0, rng.choice(i1, min(n1, len(i1)), replace=False)]))
            tk, yk, gk = [texts[i] for i in keep], y[keep], g[keep]
            t2 = [("kodx " + t) if (yk[j] == 1 and rng.rand() < 0.5) else t for j, t in enumerate(tk)]
            v = layer2(t2, yk, first_k=(3,), seed=42, groups=gk)["views"]["first_3_tokens"]
            out.append({"draw": seed, "minority_share": round(float(min(yk.mean(), 1 - yk.mean())), 3),
                        "R": v["normalized_recovery"], "raw_ratio": v["recovery_ratio"]})
            print("R2e", out[-1], flush=True)
    summ = {}
    for share in sorted({o["minority_share"] for o in out}):
        rs = [o["R"] for o in out if o["minority_share"] == share]
        summ[str(share)] = {"R_mean": round(float(np.mean(rs)), 3), "R_min": round(min(rs), 3), "R_max": round(max(rs), 3)}
    return {"runs": out, "summary": summ}


def r2f_short(corpora, cap=10000):
    out = {}
    for name in ("TurkPhish v1", "TurkPhish v2", "TTC-4900", "Product reviews", "Sentiment (test)"):
        texts, y = clean(*corpora[name])
        n = np.array(T.n_tokens(texts))
        row = {"share_le10_tokens": round(float((n <= 10).mean()), 4), "median_tokens": int(np.median(n))}
        if name in ("Product reviews", "Sentiment (test)"):
            idx = np.where(n > 20)[0]
            if len(idx) > cap:
                idx = np.sort(np.random.RandomState(42).choice(idx, cap, replace=False))
            tl, yl = [texts[i] for i in idx], y[idx]
            g = layer1(tl, yl, near_dup_cap=len(tl))["_groups"]
            v = layer2(tl, yl, first_k=(10,), seed=42, groups=g)["views"]
            row["long_docs_n"] = int(len(idx))
            row["long_first10_R"] = v["first_10_tokens"]["normalized_recovery"]
            row["long_random_window_R"] = v["random_window_10_tokens"]["normalized_recovery"]
            row["long_excess10"] = round(row["long_first10_R"] - row["long_random_window_R"], 4)
        out[name] = row
        print("R2f", name, row, flush=True)
    return out


def r2g_random_split_lookup(texts, y):
    tr, te = train_test_split(np.arange(len(texts)), test_size=0.2, random_state=42, stratify=y)
    L3 = layer3(texts, y, tr, te, seed=42)
    g = layer1(texts, y, near_dup_cap=len(texts))["_groups"]
    trg = set(g[tr])
    return {"lookup": L3["lookup_rule_macro_f1"], "fallback": L3["lookup_fallback_rate"],
            "test_overlap": round(float(np.mean([g[i] in trg for i in te])), 4)}


def main():
    corpora = load()
    if len(sys.argv) > 3 and sys.argv[3] == "--only-r2d":
        res = json.loads(OUT.read_text())
        ttc_t, ttc_y = clean(*corpora["TTC-4900"])
        res["R2d_ttc"] = r2d_inventory(ttc_t, ttc_y)
        OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
        print("saved", OUT)
        return
    t0 = time.time()
    res = {"leakaudit_version": "0.2.0"}
    ttc_t, ttc_y = clean(*corpora["TTC-4900"])
    res["R2a_ttc"] = r2a_size_matched(ttc_t, ttc_y)
    base_t, base_y = _base_v2_body(corpora)
    res["R2b"] = r2b_near_dups(base_t, base_y)
    res["R2c"] = r2c_conflicts(base_t, base_y)
    res["R2d_ttc"] = r2d_inventory(ttc_t, ttc_y)
    g = layer1(base_t, base_y, near_dup_cap=len(base_t))["_groups"]
    res["R2e"] = r2e_balance(base_t, base_y, g)
    res["R2g_ttc"] = r2g_random_split_lookup(ttc_t, ttc_y)
    res["R2f"] = r2f_short(corpora)
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print("saved", OUT, res["seconds"], "s")


if __name__ == "__main__":
    main()
