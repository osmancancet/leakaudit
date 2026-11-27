"""R4: the published train/test split of the aggregated sentiment corpus.
Exact (masked) duplication inside the training split, test documents whose masked
text occurs in training (with label agreement), and training documents copied from
the product-review corpus."""
import json, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
from leakaudit.text import mask
D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
tr = pd.read_csv(D / "duygu_train.csv"); te = pd.read_parquet(D / "duygu_test.parquet")
tr["split"], te["split"] = "train", "test"
tr["m"] = [mask(t) for t in tr["text"].astype(str)]; te["m"] = [mask(t) for t in te["text"].astype(str)]
lab = defaultdict(set)
for m, l in zip(tr["m"], tr["label"]): lab[m].add(l)
cnt = tr["m"].value_counts()
dup_groups = cnt[cnt > 1]
conflict_groups = sum(1 for m in dup_groups.index if len(lab[m]) > 1)
te["in_train"] = te["m"].isin(lab)
hit = te[te["in_train"]]
agree = sum(1 for m, l in zip(hit["m"], hit["label"]) if lab[m] == {l})
r = pd.read_parquet(D / "urun_yorum.parquet")
r = r[r["sentence"].notna() & (r["sentence"].astype(str).str.strip() != "")]
rev = {mask(t) for t in r["sentence"].astype(str)}
tr["in_reviews"] = tr["m"].isin(rev)
out = {"n_train": len(tr), "n_test": len(te),
       "train_label_counts": tr["label"].value_counts().to_dict(),
       "train_distinct_masked": int(tr["m"].nunique()),
       "train_uniqueness": round(tr["m"].nunique() / len(tr), 4),
       "train_dup_groups": int(len(dup_groups)), "train_docs_in_dup_groups": int(dup_groups.sum()),
       "train_conflict_groups": int(conflict_groups),
       "test_in_train": int(te["in_train"].sum()), "test_in_train_share": round(float(te["in_train"].mean()), 4),
       "test_in_train_label_agreement_strict": round(agree / max(1, len(hit)), 4),
       "test_in_train_per_source": {k: {"n": int(len(g)), "in_train": int(g["in_train"].sum())} for k, g in te.groupby("dataset")},
       "train_in_reviews": int(tr["in_reviews"].sum()), "train_in_reviews_share": round(float(tr["in_reviews"].mean()), 4),
       "train_per_source": {k: {"n": int(len(g)), "in_reviews": int(g["in_reviews"].sum())} for k, g in tr.groupby("dataset")}}
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
