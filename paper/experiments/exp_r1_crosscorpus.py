"""E8 cross-corpus overlap: documents of the aggregated sentiment corpus (test split)
whose masked text also occurs in the product-review corpus, per source."""
import json, sys
from pathlib import Path
import pandas as pd
from leakaudit.text import mask
D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
s = pd.read_parquet(D / "duygu_test.parquet")
r = pd.read_parquet(D / "urun_yorum.parquet")
n_rows = len(r)
r = r[r["sentence"].notna() & (r["sentence"].astype(str).str.strip() != "")]  # as in the audit (21 empty rows)
rev = {}
for t, l in zip(r["sentence"].astype(str), r["sentiment"]):
    rev.setdefault(mask(t), set()).add(int(l))
s["in_reviews"] = [mask(t) in rev for t in s["text"].astype(str)]
pol = {"Positive": 1, "Negative": 0}
agree = [pol[l] in rev[mask(t)] for t, l, f in zip(s["text"].astype(str), s["label"], s["in_reviews"]) if f and l in pol]
# strict: every copy of the text in the product-review corpus carries the same polarity
strict = [rev[mask(t)] == {pol[l]} for t, l, f in zip(s["text"].astype(str), s["label"], s["in_reviews"]) if f and l in pol]
out = {"n_sentiment_test": len(s), "n_reviews": len(r), "n_review_rows_incl_empty": n_rows,
       "overlap_total": int(s["in_reviews"].sum()),
       "overlap_label_agreement": round(sum(agree) / max(1, len(agree)), 4), "overlap_with_polarity": len(agree),
       "overlap_label_agreement_strict": round(sum(strict) / max(1, len(strict)), 4),
       "per_source": {k: {"n": int(len(g)), "in_reviews": int(g["in_reviews"].sum())} for k, g in s.groupby("dataset")},
       "label_by_source": {k: g["label"].value_counts().to_dict() for k, g in s.groupby("dataset")}}
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
