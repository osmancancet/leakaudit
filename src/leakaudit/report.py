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
    ca = r.get("certificate_content_aware")
    if ca:
        lines += [f"Content-aware profile (rates per 1,000 documents, opening views relative to a random window): "
                  f"{'PASSED' if ca['passed'] else 'FAILED'}; failed checks: "
                  f"{', '.join(k for k, v in ca['checks'].items() if not v) or 'none'}", ""]
    if c.get("warnings"):
        lines += ["Warnings:", ""] + [f"- {x}" for x in c["warnings"]] + [""]
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
        f"{L1['near_dup_pairs_cross_label']:,}; identical / non-identical pairs: {L1['near_dup_pairs_identical']:,} / "
        f"{L1['near_dup_pairs_non_identical']:,} (sample of {L1['near_dup_sample_size']:,}; "
        f"{L1['near_dup_pairs_per_1000_docs']} pairs per 1,000 documents; {L1['split_groups']:,} duplicate-connected groups used for splits).",
        "",
        "## Layer 2: surface views", "",
        f"Full-text macro-F1 (5-fold CV, {L2['folds']} folds): {L2['full_text_macro_f1']:.4f} "
        f"(random folds: {L2['full_text_macro_f1_random_folds']:.4f}); chance: {L2['chance_macro_f1']:.4f}. "
        "Normalized recovery = (view - chance) / (full text - chance), unclipped; the random-window view is a "
        "control and is not gated.", "",
        "| View | Macro-F1 | Recovery ratio | Normalized recovery | 95% CI |", "|---|---:|---:|---:|---|",
    ]
    for k, v in L2["views"].items():
        ci = v["normalized_recovery_ci95"]
        lines.append(f"| {k} | {v['macro_f1']:.4f} | {v['recovery_ratio']:.3f} | {v['normalized_recovery']:.3f} | "
                     f"{ci[0]:.3f} to {ci[1]:.3f} |")
    lines += ["", f"Max normalized recovery: {L2['max_normalized_recovery']:.3f} ({L2['max_view']}); "
              f"threshold <= {th['max_normalized_recovery']}: {_mark(ch['L2_surface_views'])}."]
    if "length_ks_p" in L2:
        lines.append(f"Length distribution by class: KS D = {L2['length_ks_D']}, p = {L2['length_ks_p']}.")
    lines += [
        "", "## Layer 3: sentences", "",
        f"Learning-free lookup over class-exclusive sentences ({L3['split']} split): "
        f"**{L3['lookup_rule_macro_f1']:.4f}** macro-F1 (fallback to majority class on "
        f"{L3['lookup_fallback_rate']:.1%} of test documents; majority constant {L3['majority_constant_macro_f1']:.4f}); "
        f"trained full-text reference: {L3['trained_reference_macro_f1']:.4f}; margin {L3['margin']:.4f}; "
        f"normalized lookup recovery {L3['normalized_lookup_recovery']:.3f} "
        f"(<= {th['max_lookup_recovery']} required): {_mark(ch['L3_lookup'])}.",
        "", "| Class | Sentence types | Class-exclusive | Test documents with a top-40 exclusive sentence |",
        "|---|---:|---:|---:|",
    ]
    names = list(r["classes"])
    for k, v in L3["sentence_inventory"].items():
        cov = [x for key, x in v.items() if key.endswith("_coverage")][0]
        lines.append(f"| {names[int(k)]} | {v['sentence_types']:,} | {v['exclusive_types']:,} | {cov:.3f} |")
    lines.append("")
    return "\n".join(lines)
