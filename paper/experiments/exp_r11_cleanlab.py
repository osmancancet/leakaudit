"""R11: Cleanlab Datalab on the same corpora, with a TF-IDF feature space (AI-referee round 6).

Cleanlab needs a feature space and predicted probabilities from the user. We give it
the most neutral choice available without a pretrained encoder: word 1-2-gram TF-IDF
reduced to 256 dimensions by truncated SVD and L2-normalized, and out-of-fold
probabilities of a class-balanced logistic regression on the full TF-IDF (5 stratified
folds). On a sample of at most 20,000 documents (seed 42) we run the label and
near-duplicate issue checks and compare the flagged documents with LeakAudit's layer 1
on the same documents: documents in duplicate-connected groups and documents in
label-conflict groups.

Usage: python exp_r11_cleanlab.py <data_dir> <out.json>    (needs cleanlab 2.9.0)
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import clean, load  # noqa: E402
from leakaudit import text as T  # noqa: E402
from leakaudit.layers import layer1  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import cleanlab
from cleanlab import Datalab

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SEED, CAP = 42, 20000


def run(texts, y):
    masked = [T.mask(t) for t in texts]
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(masked)
    X = normalize(TruncatedSVD(256, random_state=SEED).fit_transform(tf))
    P = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), tf, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")
    lab = Datalab(data={"label": y}, label_name="label", verbosity=0)
    lab.find_issues(features=X, pred_probs=P, issue_types={"label": {}, "near_duplicate": {}})
    iss = lab.get_issues()
    return (iss["is_label_issue"].to_numpy(), iss["is_near_duplicate_issue"].to_numpy(),
            list(lab.get_issues("near_duplicate")["near_duplicate_sets"]), masked)


def shingles(t, k=5):
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def jacc(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def main():
    corpora = load()
    res = {"cleanlab": cleanlab.__version__, "seed": SEED, "cap": CAP}
    for name in ("TTC-4900", "Product reviews", "Sentiment (test)"):
        df, tcol, lcol = corpora[name]
        T.set_language("tr")
        ta, ya = clean(df, tcol, lcol)
        idx = np.arange(len(ta))
        if len(idx) > CAP:
            idx = np.sort(np.random.RandomState(SEED).permutation(len(idx))[:CAP])
        texts, y = [ta[i] for i in idx], ya[idx]
        L1 = layer1(texts, y, 50000, 0.8, SEED)
        g = L1["_groups"]
        sizes = np.bincount(g)
        in_dup = sizes[g] > 1
        conf = np.zeros(len(y), bool)
        for gid in np.unique(g[in_dup]):
            m = g == gid
            if len(set(y[m])) > 1:
                conf |= m
        lab_iss, nd_iss, sets, masked = run(texts, y)
        only = np.where(nd_iss & ~in_dup)[0]
        sh = {}
        best = []
        for i in only:
            sh.setdefault(i, shingles(masked[i]))
            m = 0.0
            for j in sets[i]:
                sh.setdefault(j, shingles(masked[j]))
                m = max(m, jacc(sh[i], sh[j]))
            best.append(m)
        best = np.array(best)
        bins = {"ge_0.8": int((best >= 0.8).sum()), "0.5_0.8": int(((best >= 0.5) & (best < 0.8)).sum()),
                "0.2_0.5": int(((best >= 0.2) & (best < 0.5)).sum()), "lt_0.2": int((best < 0.2).sum())}
        cross = int(sum(any(y[j] != y[i] for j in sets[i]) for i in only))
        r = {"n": int(len(y)),
             "leakaudit_dup_docs": int(in_dup.sum()), "leakaudit_conflict_docs": int(conf.sum()),
             "cleanlab_near_dup_docs": int(nd_iss.sum()), "cleanlab_label_issue_docs": int(lab_iss.sum()),
             "near_dup_both": int((nd_iss & in_dup).sum()),
             "cleanlab_near_dup_not_leakaudit": int((nd_iss & ~in_dup).sum()),
             "leakaudit_dup_not_cleanlab": int((in_dup & ~nd_iss).sum()),
             "conflict_docs_flagged_label_issue": int((conf & lab_iss).sum()),
             "cleanlab_only_max_jaccard_bins": bins,
             "cleanlab_only_median_max_jaccard": round(float(np.median(best)), 4) if len(best) else None,
             "cleanlab_only_with_other_label_partner": cross}
        res[name] = r
        print(name, json.dumps(r), flush=True)
        OUT.write_text(json.dumps(res, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
