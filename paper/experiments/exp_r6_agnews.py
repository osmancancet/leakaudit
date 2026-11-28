"""R6: AG News (English, fancyzhx/ag_news revision eb185aade064a813bc0b7f42de02595523103ca4)
as an out-of-sample corpus: audited with --lang generic after the content-aware profile
was fixed. This script adds the exact (masked) overlap between the published test and
training splits and its label agreement."""
import json, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
from leakaudit.text import mask, set_language
D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
set_language("generic")
tr = pd.read_parquet(D / "agnews_train.parquet"); te = pd.read_parquet(D / "agnews_test.parquet")
names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
tr["m"] = [mask(t) for t in tr["text"]]; te["m"] = [mask(t) for t in te["text"]]
lab = defaultdict(set)
for m, l in zip(tr["m"], tr["label"]): lab[m].add(l)
te["in_train"] = te["m"].isin(lab)
hit = te[te["in_train"]]
cnt = tr["m"].value_counts(); dup = cnt[cnt > 1]
out = {"n_train": len(tr), "n_test": len(te),
       "train_uniqueness": round(tr["m"].nunique() / len(tr), 4),
       "train_dup_groups": int(len(dup)), "train_docs_in_dup_groups": int(dup.sum()),
       "train_conflict_groups": int(sum(1 for m in dup.index if len(lab[m]) > 1)),
       "test_in_train": int(te["in_train"].sum()), "test_in_train_share": round(float(te["in_train"].mean()), 4),
       "test_in_train_label_agreement_strict": round(sum(lab[m] == {l} for m, l in zip(hit["m"], hit["label"])) / max(1, len(hit)), 4)}
OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
