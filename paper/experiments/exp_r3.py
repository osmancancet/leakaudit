"""Positive control for the certificate (added after the AI-referee stress test).

R3a  TTC-4900 and the product-review corpus are cleaned the way the audit
     recommends: every duplicate-connected group (exact and verified
     near-duplicates, Jaccard >= 0.8, searched over all documents) is reduced to
     one document, and groups whose members carry different labels are removed.
     The cleaned corpora are audited with the default configuration, giving both
     the strict and the content-aware verdict.

Usage: python exp_r3.py <data_dir> <out.json>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import clean, load  # noqa: E402
from leakaudit import audit  # noqa: E402
from leakaudit.layers import layer1  # noqa: E402

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
np.seterr(all="ignore")


def cleaned(texts, y):
    g = layer1(texts, y, near_dup_cap=len(texts))["_groups"]
    keep, dropped_conflict = [], 0
    by = {}
    for i, gi in enumerate(g):
        by.setdefault(gi, []).append(i)
    for gi, idx in by.items():
        if len(set(y[idx])) > 1:
            dropped_conflict += len(idx)
            continue
        keep.append(idx[0])
    keep = np.sort(np.array(keep))
    return keep, dropped_conflict


def main():
    corpora = load()
    res = {"leakaudit_version": "0.2.0"}
    t0 = time.time()
    for name, key in (("TTC-4900", "ttc4900_clean"), ("Product reviews", "urun_yorum_clean")):
        texts, y = clean(*corpora[name])
        keep, dc = cleaned(texts, y)
        df = pd.DataFrame({"text": [texts[i] for i in keep], "label": y[keep]})
        rep = audit(df, text="text", label="label", name=f"{name} (cleaned)")
        Path(OUT.parent / f"{key}_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
        res[key] = {"n_before": len(texts), "n_after": int(len(keep)), "removed_in_conflict_groups": int(dc),
                    "strict": rep["certificate"]["checks"], "strict_passed": rep["certificate"]["passed"],
                    "content_aware": rep["certificate_content_aware"]["checks"],
                    "content_aware_passed": rep["certificate_content_aware"]["passed"],
                    "uniqueness": rep["layer1"]["template_uniqueness"], "near_dup_pairs": rep["layer1"]["near_dup_pairs"],
                    "conflicts": rep["layer1"]["label_conflicting_duplicate_groups"],
                    "full_f1": rep["layer2"]["full_text_macro_f1"],
                    "first10_R": rep["layer2"]["views"]["first_10_tokens"]["normalized_recovery"],
                    "random_window_R": rep["layer2"]["views"]["random_window_10_tokens"]["normalized_recovery"],
                    "warnings": rep["certificate"]["warnings"]}
        print("R3a", key, res[key], flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print("saved", OUT, res["seconds"], "s")


if __name__ == "__main__":
    main()
