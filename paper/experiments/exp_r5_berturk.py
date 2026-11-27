"""R5: the TTC-4900 inflation experiment (R2a) repeated with BERTurk
(dbmdz/bert-base-turkish-cased, revision b6e1de16c983e0f2c70664591ea3f22810072608).

Same splits, same duplicate removal and same size- and class-matched random
removal as R2a; the model is fine-tuned on the two reduced training sets and
scored on the same test part. Duplication effect = F1(random removal) - F1(duplicate removal).

Usage: python exp_r5_berturk.py <data_dir> <model_dir> <out.json> [seeds, e.g. 42,43]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import clean, load  # noqa: E402
from exp_r1 import SEEDS, f1, nb_interval  # noqa: E402
from leakaudit.layers import layer1  # noqa: E402

DATA, MODEL, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
RUN = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else SEEDS  # seeds to run now; earlier results in OUT are kept
MAXLEN, BS, LR, EPOCHS = 128, 16, 3e-5, 2
torch.set_num_threads(max(1, torch.get_num_threads()))


def finetune_eval(tok, texts, y, tr, te, nc, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=nc)
    enc = lambda idx: tok([texts[i] for i in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(tr) / BS)); sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
    rng = np.random.RandomState(seed)
    model.train()
    for _ in range(EPOCHS):
        order = rng.permutation(tr)
        for b in range(0, len(order), BS):
            idx = order[b:b + BS]
            out = model(**enc(idx), labels=torch.tensor(y[idx]))
            out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
    model.eval(); pred = []
    with torch.no_grad():
        for b in range(0, len(te), 64):
            pred.extend(model(**enc(te[b:b + 64])).logits.argmax(-1).tolist())
    return f1(y[te], np.array(pred))


def main():
    t0 = time.time()
    texts, y = clean(*load()["TTC-4900"])
    nc = len(np.unique(y))
    g = layer1(texts, y, near_dup_cap=len(texts))["_groups"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    res = {"model": "dbmdz/bert-base-turkish-cased@b6e1de16", "max_len": MAXLEN, "batch": BS, "lr": LR,
           "epochs": EPOCHS, "n": len(texts), "runs": []}
    if OUT.exists():
        res["runs"] = [r for r in json.loads(OUT.read_text())["runs"] if r["seed"] not in RUN]
    for s in RUN:
        rng = np.random.RandomState(1000 + s)
        tr, te = train_test_split(np.arange(len(texts)), test_size=0.2, random_state=s, stratify=y)
        te_groups = set(g[te])
        removed = np.array([i for i in tr if g[i] in te_groups])
        tr_dedup = np.array([i for i in tr if g[i] not in te_groups])
        drop = []
        for c in np.unique(y):
            k = int((y[removed] == c).sum()) if len(removed) else 0
            drop.extend(rng.choice(tr[y[tr] == c], k, replace=False).tolist())
        drop = set(drop)
        tr_rand = np.array([i for i in tr if i not in drop])
        fd = finetune_eval(tok, texts, y, tr_dedup, te, nc, s)
        fr = finetune_eval(tok, texts, y, tr_rand, te, nc, s)
        row = {"seed": s, "removed": int(len(removed)), "dedup": round(fd, 4), "random_removed": round(fr, 4),
               "duplication_effect_pts": round(100 * (fr - fd), 3), "elapsed_s": round(time.time() - t0)}
        res["runs"].append(row); res["runs"].sort(key=lambda r: r["seed"]); print("R5", row, flush=True)
        OUT.write_text(json.dumps(res, indent=2))
    n_te = len(texts) // 5
    d = [r["duplication_effect_pts"] for r in res["runs"]]
    res["summary"] = {"random_removed_mean": round(float(np.mean([r["random_removed"] for r in res["runs"]])), 4),
                      "duplication_effect_mean_pts": round(float(np.mean(d)), 3),
                      "duplication_effect_ci95_pts": nb_interval(d, len(texts) - n_te, n_te) if len(d) > 1 else None}
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2)); print("saved", OUT, res["summary"])


if __name__ == "__main__":
    main()
