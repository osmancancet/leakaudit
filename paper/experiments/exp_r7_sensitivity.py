"""R7: sensitivity of the content-aware verdicts to each threshold.
For every check, the value each audited corpus has on the gated quantity, and the
interval of the threshold within which no verdict changes."""
import json, sys
from pathlib import Path
from leakaudit.audit import ContentAwareThresholds, near_dup_docs_per_1000
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
reps = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "reports").glob("*.json"))}
for k in ("ttc4900_clean", "urun_yorum_clean"):
    reps[k] = json.loads((ROOT / "paper/experiments" / f"{k}_report.json").read_text())
reps.pop("turkphish_v2_body", None)
th = ContentAwareThresholds()


def quantities(r):
    L1, v, L3 = r["layer1"], r["layer2"]["views"], r["layer3"]
    def win(k):
        kk = k.split("_")[1]
        return v.get(f"random_window_{kk}_tokens", v["random_window_10_tokens"])["normalized_recovery"]
    exc = max(d["normalized_recovery"] - win(k) for k, d in v.items() if k.startswith("first_"))
    oth = max(d["normalized_recovery"] for k, d in v.items() if not k.startswith(("first_", "random_window")))
    n = max(1, r["n_documents"])
    q = {"uniqueness": (L1["template_uniqueness"], th.min_template_uniqueness, ">="),
         "near_dup_docs_per_1000": (near_dup_docs_per_1000(L1), th.max_near_dup_docs_per_1000, "<="),
         "conflict_groups_per_1000": (1000 * L1["label_conflicting_duplicate_groups"] / n, th.max_label_conflicts_per_1000, "<="),
         "opening_excess": (exc, th.max_opening_excess, "<="),
         "other_views_R": (oth, th.max_normalized_recovery, "<=")}
    if L3["lookup_fallback_rate"] <= th.max_lookup_fallback:
        q["lookup_R"] = (L3["normalized_lookup_recovery"], th.max_lookup_recovery, "<=")
    return q


Q = {k: quantities(r) for k, r in reps.items()}
out = {"corpora": {k: {c: round(v[0], 4) for c, v in q.items()} for k, q in Q.items()}, "stable_interval": {}}
for c in ("uniqueness", "near_dup_docs_per_1000", "conflict_groups_per_1000", "opening_excess", "other_views_R", "lookup_R"):
    vals = sorted(q[c][0] for q in Q.values() if c in q)
    t = next(q[c][1] for q in Q.values() if c in q)
    # no verdict changes while the threshold stays strictly between the nearest values on either side
    lo = max([x for x in vals if x <= t], default=None); hi = min([x for x in vals if x > t], default=None)
    out["stable_interval"][c] = {"threshold": t, "nearest_below_or_at": None if lo is None else round(lo, 4),
                                 "nearest_above": None if hi is None else round(hi, 4)}
OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
