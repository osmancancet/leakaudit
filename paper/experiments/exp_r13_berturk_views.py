"""R13: the TTC-4900 opening check of layer 2 repeated with fine-tuned BERTurk
(dbmdz/bert-base-turkish-cased, revision b6e1de16c983e0f2c70664591ea3f22810072608).

On duplicate-grouped hold-out splits (the first fold of the layer 2 partition) under three
seeds, BERTurk is fine-tuned on the masked full text and on each first-k view and random
k-token window (k = 3, 5, 10), and scored on the same test part. The normalized recovery
R = (F_view - 1/C) / (F_full - 1/C) and the opening excess (first-k minus window) are
computed with BERTurk as the reference model, and side by side with the TF-IDF models of
LeakAudit on the same splits.

Usage: python exp_r13_berturk_views.py <data_dir> <model_dir> <out.json> [seeds, e.g. 42,43,44]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import clean, load  # noqa: E402
from leakaudit import text as T  # noqa: E402
from leakaudit.layers import layer1, _grouped_holdout, _word_model, _full_text_model, _f1  # noqa: E402

DATA, MODEL, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
SEEDS = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [42, 43, 44]
KS, BS, LR, EPOCHS = (3, 5, 10), 16, 3e-5, 2
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def finetune_eval(tok, X, y, tr, te, nc, seed, maxlen):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=nc).to(DEV)
    enc = lambda idx: {k: v.to(DEV) for k, v in tok([X[i] for i in idx], truncation=True, max_length=maxlen,
                                                    padding=True, return_tensors="pt").items()}
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(tr) / BS))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
    rng = np.random.RandomState(seed)
    model.train()
    for _ in range(EPOCHS):
        order = rng.permutation(tr)
        for b in range(0, len(order), BS):
            idx = order[b:b + BS]
            out = model(**enc(idx), labels=torch.tensor(y[idx]).to(DEV))
            out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
    model.eval(); pred = []
    with torch.no_grad():
        for b in range(0, len(te), 64):
            pred.extend(model(**enc(te[b:b + 64])).logits.argmax(-1).cpu().tolist())
    return _f1(y[te], np.array(pred))


def views(texts, seed):
    rng = np.random.RandomState(seed)
    v = {"full": [T.mask(t) for t in texts]}
    for k in KS:
        v[f"first_{k}"] = [T.first_tokens(t, k) for t in texts]
        w = []
        for t in texts:
            toks = T.mask(t).split(" ")
            s = rng.randint(0, max(1, len(toks) - k + 1))
            w.append(" ".join(toks[s:s + k]))
        v[f"window_{k}"] = w
    return v


def main():
    t0 = time.time()
    texts, y = clean(*load()["TTC-4900"])
    nc = len(np.unique(y)); ch = 1.0 / nc
    g = layer1(texts, y, near_dup_cap=len(texts))["_groups"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    res = json.loads(OUT.read_text()) if OUT.exists() else {
        "model": "dbmdz/bert-base-turkish-cased@b6e1de16", "batch": BS, "lr": LR, "epochs": EPOCHS, "runs": []}
    res["device"] = str(DEV)
    res["runs"] = [r for r in res["runs"] if r["seed"] not in SEEDS]
    for s in SEEDS:
        tr, te = _grouped_holdout(y, g, s, {})
        V = views(texts, s)
        row = {"seed": s, "n_train": int(len(tr)), "n_test": int(len(te)), "berturk": {}, "tfidf": {}}
        for name, X in V.items():
            ml = 128 if name == "full" else 32
            row["berturk"][name] = round(finetune_eval(tok, X, y, tr, te, nc, s, ml), 4)
            m = _full_text_model() if name == "full" else _word_model((1, 2), 1)
            row["tfidf"][name] = round(_f1(y[te], m.fit([texts[i] if name == "full" else X[i] for i in tr], y[tr])
                                           .predict([texts[i] if name == "full" else X[i] for i in te])), 4)
            print(s, name, row["berturk"][name], row["tfidf"][name], round(time.time() - t0), flush=True)
        for mdl in ("berturk", "tfidf"):
            F = row[mdl]; den = F["full"] - ch
            R = {k: round((F[k] - ch) / den, 4) for k in F if k != "full"}
            row[f"{mdl}_R"] = R
            row[f"{mdl}_excess"] = {k: round(R[f"first_{k}"] - R[f"window_{k}"], 4) for k in KS}
        res["runs"].append(row); res["runs"].sort(key=lambda r: r["seed"])
        OUT.write_text(json.dumps(res, indent=2))
    for mdl in ("berturk", "tfidf"):
        ex = [max(r[f"{mdl}_excess"].values()) for r in res["runs"]]
        res[f"{mdl}_max_excess_mean"] = round(float(np.mean(ex)), 4)
        res[f"{mdl}_max_excess_range"] = [round(float(min(ex)), 4), round(float(max(ex)), 4)]
        res[f"{mdl}_full_mean"] = round(float(np.mean([r[mdl]["full"] for r in res["runs"]])), 4)
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2)); print("saved", OUT)


if __name__ == "__main__":
    main()
