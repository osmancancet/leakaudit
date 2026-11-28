"""R9: recovery anchored on the empirical permuted-label null (AI-referee round 5).

R_v uses F0 = 1/C. Here every view is also scored against its own permuted-label
null, R0_v = (F_v - N_v) / (F_full - N_v), where N_v is the macro-F1 of the same view
model refitted on permuted labels with the same folds (stored in every report). The
script reports the range of both scores over all gated views and whether any
content-aware verdict changes when R0 replaces R. No model is refitted.

Usage: python exp_r9_null_anchor.py <repo_root> <out.json>
"""
import copy, json, sys
from pathlib import Path
from leakaudit.audit import content_aware_certificate

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
REPS = {k: ROOT / "reports" / f"{k}.json" for k in
        ("turkphish_v1", "turkphish_v2", "ttc4900", "urun_yorum", "duygu", "agnews")}
REPS.update({k: ROOT / "paper/experiments" / f"{k}_report.json" for k in ("ttc4900_clean", "urun_yorum_clean")})


def r0(v, full):
    n = v["permuted_label_macro_f1"]
    return (v["macro_f1"] - n) / (full - n) if full != n else float("nan")


res, allR, allR0, changed = {}, [], [], []
for k, p in REPS.items():
    rep = json.loads(p.read_text())
    full = rep["layer2"]["full_text_macro_f1"]
    alt = copy.deepcopy(rep)
    per = {}
    for name, v in alt["layer2"]["views"].items():
        a = r0(v, full)
        per[name] = {"R": v["normalized_recovery"], "R0": round(a, 4)}
        v["normalized_recovery"] = a
        if not name.startswith("random_window"):
            allR.append(per[name]["R"]); allR0.append(a)
    before = content_aware_certificate(rep)["checks"]
    after = content_aware_certificate(alt)["checks"]
    res[k] = {"views": per, "verdict_R": before, "verdict_R0": after, "changed": before != after}
    if before != after:
        changed.append(k)
res["summary"] = {"R_min": round(min(allR), 4), "R_max": round(max(allR), 4),
                  "R0_min": round(min(allR0), 4), "R0_max": round(max(allR0), 4),
                  "n_negative_R": sum(x < 0 for x in allR), "n_negative_R0": sum(x < 0 for x in allR0),
                  "n_views": len(allR), "verdicts_changed": changed}
OUT.write_text(json.dumps(res, indent=2)); print(json.dumps(res["summary"], indent=1))
