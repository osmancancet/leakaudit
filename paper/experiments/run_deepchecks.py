"""Run the Deepchecks NLP data-integrity checks that overlap with LeakAudit layer 1."""
import json, sys, time
from pathlib import Path
import pandas as pd
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import TextDuplicates, ConflictingLabels, TrainTestSamplesMix
from sklearn.model_selection import train_test_split

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
corp = {"TTC-4900": (pd.read_parquet(D / "ttc4900.parquet"), "text", "category"),
        "Sentiment (test)": (pd.read_parquet(D / "duygu_test.parquet"), "text", "label"),
        "Product reviews (50k sample)": (pd.read_parquet(D / "urun_yorum.parquet").sample(50000, random_state=42), "sentence", "sentiment")}
res = {"deepchecks": __import__("deepchecks").__version__}
for name, (df, t, l) in corp.items():
    df = df[df[t].notna() & (df[t].astype(str).str.strip() != "")].reset_index(drop=True)
    td = TextData(df[t].astype(str).tolist(), label=df[l].astype(str).tolist(), task_type="text_classification")
    r = {}
    t0 = time.time()
    v = TextDuplicates().run(td).value
    r["TextDuplicates"] = {"percent_of_duplicates": float(v["percent_of_duplicates"]),
                           "n_duplicate_groups": int(len(v["duplicates"].index.get_level_values(0).unique())) if hasattr(v["duplicates"], "index") else None}
    v = ConflictingLabels().run(td).value
    r["ConflictingLabels"] = {"percent_of_conflicting_samples": float(v["percent_of_conflicting_samples"]),
                              "n_conflicting_groups": int(len(v["samples_per_conflict"])) if "samples_per_conflict" in v else None}
    tr, te = train_test_split(range(len(df)), test_size=0.2, random_state=42, stratify=df[l].astype(str))
    a = TextData([df[t].astype(str)[i] for i in tr], label=[df[l].astype(str)[i] for i in tr], task_type="text_classification")
    b = TextData([df[t].astype(str)[i] for i in te], label=[df[l].astype(str)[i] for i in te], task_type="text_classification")
    v = TrainTestSamplesMix().run(a, b).value
    r["TrainTestSamplesMix_random_split"] = {k: float(v[k]) for k in v if isinstance(v[k], (int, float))}
    r["seconds"] = round(time.time() - t0, 1)
    res[name] = r
    print(name, r, flush=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))
