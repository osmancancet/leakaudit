"""Human-readable Markdown summary of an audit report."""
from __future__ import annotations

from typing import Dict


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def to_markdown(r: Dict) -> str:
    c, L1, L2, L3 = r["certificate"], r["layer1"], r["layer2"], r["layer3"]
    th = r["config"]["thresholds"]
    lines = [
        f"# LeakAudit report{': ' + r['corpus'] if r.get('corpus') else ''}",
        "",
        f"**Certificate: {'PASSED' if c['passed'] else 'FAILED'}**  ",
        f"{r['n_documents']:,} documents, {len(r['classes'])} classes; "
        f"layers 2-3 on {r['n_audited_layers_2_3']:,} documents; seed {r['config']['seed']}; "
        f"leakaudit {r['version']}.",
        "",
    ]
    if c["reasons"]:
        lines += ["Reasons:", ""] + [f"- {x}" for x in c["reasons"]] + [""]
    ch = c["checks"]
    lines += [
        "## Layer 1: documents", "",
        "| Check | Value | Threshold | Result |", "|---|---:|---:|---|",
        f"| Template uniqueness after masking | {L1['template_uniqueness']:.4f} | >= {th['min_template_uniqueness']} | {_mark(ch['L1_template_uniqueness'])} |",
        f"| Near-duplicate pairs (Jaccard >= {L1['near_dup_jaccard_threshold']}) | {L1['near_dup_pairs']:,} | <= {th['max_near_dup_pairs']} | {_mark(ch['L1_near_duplicates'])} |",
        f"| Duplicate groups with conflicting labels | {L1['label_conflicting_duplicate_groups']:,} | <= {th['max_label_conflicting_groups']} | {_mark(ch['L1_label_conflicts'])} |",
        "",
        f"Distinct masked templates: {L1['distinct_masked_templates']:,}; duplicate groups: {L1['duplicate_groups']:,} "
        f"covering {L1['documents_in_duplicate_groups']:,} documents; near-duplicate pairs crossing labels: "
        f"{L1['near_dup_pairs_cross_label']:,} (sample of {L1['near_dup_sample_size']:,}).",
        "",
        "## Layer 2: surface views", "",
        f"Full-text macro-F1 (5-fold CV): {L2['full_text_macro_f1']:.4f}; chance: {L2['chance_macro_f1']:.4f}. "
        "Normalized recovery = (view - chance) / (full text - chance).", "",
        "| View | Macro-F1 | Recovery ratio | Normalized recovery |", "|---|---:|---:|---:|",
    ]
    for k, v in L2["views"].items():
        lines.append(f"| {k} | {v['macro_f1']:.4f} | {v['recovery_ratio']:.3f} | {v['normalized_recovery']:.3f} |")
    lines += ["", f"Max normalized recovery: {L2['max_normalized_recovery']:.3f} ({L2['max_view']}); "
              f"threshold <= {th['max_normalized_recovery']}: {_mark(ch['L2_surface_views'])}."]
    if "length_ks_p" in L2:
        lines.append(f"Length distribution by class: KS D = {L2['length_ks_D']}, p = {L2['length_ks_p']}.")
    lines += [
        "", "## Layer 3: sentences", "",
        f"Zero-learning lookup over class-exclusive sentences: **{L3['lookup_rule_macro_f1']:.4f}** macro-F1; "
        f"trained full-text reference: {L3['trained_reference_macro_f1']:.4f}; margin {L3['margin']:.4f} "
        f"(>= {th['min_lookup_margin']} required): {_mark(ch['L3_lookup_margin'])}.",
        "", "| Class | Sentence types | Class-exclusive | Top-40 exclusive coverage |", "|---|---:|---:|---:|",
    ]
    names = list(r["classes"])
    for k, v in L3["sentence_inventory"].items():
        cov = [x for key, x in v.items() if key.endswith("_coverage")][0]
        lines.append(f"| {names[int(k)]} | {v['sentence_types']:,} | {v['exclusive_types']:,} | {cov:.3f} |")
    lines.append("")
    return "\n".join(lines)
