"""Build every number, table and the manuscript body from the result files.

  reports/*.json, paper/experiments/{results,exp_r1,exp_r1_crosscorpus}.json
    -> paper/numbers.json, paper/tables/*.tex, paper/manuscript/leakaudit_main.tex

The manuscript source paper/manuscript/leakaudit_main.src.tex contains {{key}}
placeholders; each is replaced by the value of that key in numbers.json, so no
number in the paper is typed by hand.

Usage: python paper/make_tables.py <repo_root>
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
REP, EXP = ROOT / "reports", ROOT / "paper" / "experiments"
TAB, MS = ROOT / "paper" / "tables", ROOT / "paper" / "manuscript"
TAB.mkdir(parents=True, exist_ok=True)

CORP = [("turkphish_v1", "TurkPhish v1"), ("turkphish_v2", "TurkPhish v2"), ("turkphish_v2_body", "TurkPhish v2 (body only)"),
        ("ttc4900", "TTC-4900"), ("urun_yorum", "Product reviews"), ("duygu", "Sentiment (test)")]
R = {k: json.load(open(REP / f"{k}.json")) for k, _ in CORP}
RES = json.load(open(EXP / "results.json"))
X = json.load(open(EXP / "exp_r1.json"))
CC = json.load(open(EXP / "exp_r1_crosscorpus.json"))
X2 = json.load(open(EXP / "exp_r2.json"))
N = {}


def f(x, d=3):
    if isinstance(x, (int, float)) and not math.isnan(x) and abs(x) < 0.5 * 10 ** -d:
        x = 0.0
    return f"{x:.{d}f}".replace("-", "$-$") if isinstance(x, (int, float)) and not math.isnan(x) else "n/a"


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def thou(x):
    return f"{int(x):,}"


def ci(c, d=2):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


# ------------------------------------------------------------------ headline numbers
for k, name in CORP:
    r = R[k]
    L1, L2, L3 = r["layer1"], r["layer2"], r["layer3"]
    N[f"{k}.n"] = thou(r["n_documents"])
    N[f"{k}.seconds"] = f"{sum(r['seconds'].values()):.0f}"
    N[f"{k}.uniq"] = f(L1["template_uniqueness"], 4)
    N[f"{k}.templates"] = thou(L1["distinct_masked_templates"])
    N[f"{k}.dupgroups"] = thou(L1["duplicate_groups"])
    N[f"{k}.dupdocs"] = thou(L1["documents_in_duplicate_groups"])
    N[f"{k}.dupshare"] = pct(L1["documents_in_duplicate_groups"] / r["n_documents"], 0)
    N[f"{k}.redundant"] = pct((L1["documents_in_duplicate_groups"] - L1["duplicate_groups"]) / r["n_documents"], 1)
    N[f"{k}.conflicts"] = thou(L1["label_conflicting_duplicate_groups"])
    N[f"{k}.pairs"] = thou(L1["near_dup_pairs"])
    N[f"{k}.pairs_nonid"] = thou(L1["near_dup_pairs_non_identical"])
    N[f"{k}.pairs_cross"] = thou(L1["near_dup_pairs_cross_label"])
    N[f"{k}.splitgroups"] = thou(L1["split_groups"])
    N[f"{k}.full"] = f(L2["full_text_macro_f1"])
    N[f"{k}.full_rand"] = f(L2["full_text_macro_f1_random_folds"])
    for v, d in L2["views"].items():
        key = v.replace("metadata:", "meta_").replace("_tokens", "").replace("punctuation_emoji_case_profile", "profile")
        N[f"{k}.{key}.f1"] = f(d["macro_f1"])
        N[f"{k}.{key}.R"] = f(d["normalized_recovery"], 2)
        N[f"{k}.{key}.Rpct"] = f"{100 * d['normalized_recovery']:.0f}\\%"
        N[f"{k}.{key}.ci"] = ci(d["normalized_recovery_ci95"])
    N[f"{k}.lookup"] = f(L3["lookup_rule_macro_f1"])
    N[f"{k}.majority"] = f(L3["majority_constant_macro_f1"])
    N[f"{k}.fallback"] = pct(L3["lookup_fallback_rate"])
    N[f"{k}.trained"] = f(L3["trained_reference_macro_f1"])
    N[f"{k}.margin"] = f(L3["margin"])
    N[f"{k}.margin_ci"] = ci(L3["margin_ci95"], 3)
    N[f"{k}.RL"] = f(L3["normalized_lookup_recovery"], 2)
    N[f"{k}.RL_ci"] = ci(L3["normalized_lookup_recovery_ci95"])
    cov = [list(v.values())[2] for v in L3["sentence_inventory"].values()]
    N[f"{k}.coverage_max"] = pct(max(cov), 0)
    N[f"{k}.split"] = L3["split"]
    if "length_ks_p" in L2:
        N[f"{k}.ks_p"] = f(L2["length_ks_p"])

first10 = [R[k]["layer2"]["views"]["first_10_tokens"]["normalized_recovery"] for k in
           ("turkphish_v1", "turkphish_v2", "ttc4900", "urun_yorum", "duygu")]
N["first10.min"], N["first10.max"] = f"{100 * min(first10):.0f}\\%", f"{100 * min(1, max(first10)):.0f}\\%"
for k in ("turkphish_v1", "turkphish_v2", "ttc4900", "urun_yorum", "duygu"):
    v = R[k]["layer2"]["views"]
    N[f"{k}.excess10"] = f(v["first_10_tokens"]["normalized_recovery"] - v["random_window_10_tokens"]["normalized_recovery"], 2)
gen = ("ttc4900", "urun_yorum", "duygu")
N["gen.fallback_min"] = pct(min(R[k]["layer3"]["lookup_fallback_rate"] for k in gen), 0)
runs = [sum(R[k]["seconds"].values()) for k, _ in CORP]
N["runtime.min"], N["runtime.max"] = f"{min(runs):.0f}", f"{max(runs):.0f}"
rt = RES["runtime_layer1_product_reviews"]
N["rt.first_n"], N["rt.first_s"] = thou(rt[0]["n"]), f"{rt[0]['seconds']:.0f}"
N["rt.last_n"], N["rt.last_s"] = thou(rt[-1]["n"]), f"{rt[-1]['seconds']:.0f}"
N["rt.exponent"] = f"{math.log(rt[-1]['seconds'] / rt[0]['seconds']) / math.log(rt[-1]['n'] / rt[0]['n']):.1f}"
for k, name in (("TurkPhish v1", "v1"), ("TTC-4900", "ttc"), ("Product reviews", "rev"), ("Sentiment (test)", "sen")):
    m = RES[k]["masking"]
    N[f"mask.{name}.lower"], N[f"mask.{name}.masked"] = f(m["lowercase_only"]), f(m["masked"])
N["v1.pairs_old"] = thou(336640)  # value reported by leakaudit 0.1.0 (reports of 2026-11-25, archived)

# E5
for k, key in (("TTC-4900", "ttc"), ("Product reviews", "rev"), ("Sentiment (test)", "sen")):
    e = X["E5"][k]
    N[f"e5.{key}.overlap"] = pct(sum(r["overlap"] for r in e["runs"]) / len(e["runs"]), 1)
    for m in ("lr", "knn1"):
        N[f"e5.{key}.{m}.mean"] = f(e[m]["inflation_mean_pts"], 2)
        N[f"e5.{key}.{m}.ci"] = ci(e[m]["nb_ci95_pts"])
        N[f"e5.{key}.{m}.full"] = f(sum(r[m]["full_train"] for r in e["runs"]) / 5)
        N[f"e5.{key}.{m}.dedup"] = f(sum(r[m]["dedup_train"] for r in e["runs"]) / 5)
    ov = [r["lr"]["acc_overlapping"] for r in e["runs"]]
    nov = [r["lr"]["acc_non_overlapping"] for r in e["runs"]]
    N[f"e5.{key}.acc_ov"], N[f"e5.{key}.acc_nov"] = f(sum(ov) / 5), f(sum(nov) / 5)
N["e5.v1.overlap"] = pct(sum(r["overlap"] for r in X["E5"]["TurkPhish v1"]["runs"]) / 5, 2)

# E6
E6 = X["E6"]
N["e6.n"] = thou(E6["base_n"])
for d in E6["planted_duplicates"]:
    N[f"e6.dup{int(100 * d['rate'])}.inj"], N[f"e6.dup{int(100 * d['rate'])}.pairs"] = thou(d["injected"]), thou(d["near_dup_pairs"])
    N[f"e6.dup{int(100 * d['rate'])}.uniq"] = f(d["template_uniqueness"], 3)
for d in E6["planted_opening"]:
    N[f"e6.open{int(100 * d['p'])}.R"], N[f"e6.open{int(100 * d['p'])}.ci"] = f(d["first3_R"], 2), ci(d["first3_R_ci95"])
for d in E6["planted_metadata"]:
    N[f"e6.meta{int(100 * d['agreement'])}.R"], N[f"e6.meta{int(100 * d['agreement'])}.ci"] = f(d["R"], 2), ci(d["R_ci95"])
    N[f"e6.meta{int(100 * d['agreement'])}.raw"] = f(d["raw_ratio"], 2)
for d in E6["balance"]:
    s = f"{round(100 * d['minority_share']):d}"
    N[f"e6.bal{s}.R"], N[f"e6.bal{s}.raw"], N[f"e6.bal{s}.maj"] = f(d["R"], 2), f(d["raw_ratio"], 2), f(d["majority_based"], 2)

# E7 / E8
E7 = X["E7"]
N["e7.src_f1"], N["e7.src_chance"] = f(E7["source_from_text_macro_f1"]), f(E7["source_chance"])
for s, v in E7["sources"].items():
    if "within_source_macro_f1" in v:
        N[f"e7.{s}.f1"] = f(v["within_source_macro_f1"])
N["cc.total"] = thou(CC["overlap_total"])
N["cc.share"] = pct(CC["overlap_total"] / CC["n_sentiment_test"], 0)
N["cc.urun.n"] = thou(CC["per_source"]["urun_yorumlari"]["n"])
N["cc.urun.in"] = thou(CC["per_source"]["urun_yorumlari"]["in_reviews"])
N["cc.agree"] = pct(CC["overlap_label_agreement"], 2)
neutral = sum(v.get("Notr", 0) for v in CC["label_by_source"].values())
wiki_neutral = CC["label_by_source"]["wiki"].get("Notr", 0) + CC["label_by_source"]["random"].get("Notr", 0)
N["cc.neutral_from_wiki"] = pct(wiki_neutral / neutral, 0)
N["cc.wiki_share_neutral"] = pct(CC["label_by_source"]["wiki"]["Notr"] / neutral, 1)

N["cc.agree_strict"] = pct(CC["overlap_label_agreement_strict"], 1)
N["cc.rev_rows"] = thou(CC["n_review_rows_incl_empty"])
N["gen.fallback_floor"] = f"{math.floor(1000 * min(R[k]['layer3']['lookup_fallback_rate'] for k in gen)) / 10:.1f}\\%"

# R2 experiments
A = X2["R2a_ttc"]
N["r2a.removed"] = pct(sum(r["removed_share"] for r in A["runs"]) / len(A["runs"]), 1)
for m in ("lr", "knn1"):
    N[f"r2a.{m}.dup"] = f(A[m]["duplication_effect_mean_pts"], 2)
    N[f"r2a.{m}.dup_ci"] = ci(A[m]["duplication_effect_ci95_pts"])
    N[f"r2a.{m}.size"] = f(A[m]["size_effect_mean_pts"], 2)
for d in X2["R2b"]:
    s = f"{round(100 * d['token_replacement'])}"
    N[f"r2b.{s}.true"], N[f"r2b.{s}.det"] = thou(d["true_pairs_j08"]), thou(d["detected_pairs"])
for d in X2["R2c"]:
    N[f"r2c.{d['planted']}.found"] = thou(d["label_conflicting_groups"])
for d in X2["R2d_ttc"]:
    N[f"r2d.{round(100 * d['p'])}.RL"] = f(d["R_L"], 2)
for share, v in X2["R2e"]["summary"].items():
    s = f"{round(100 * float(share))}"
    N[f"r2e.{s}.mean"] = f(v["R_mean"], 2)
    N[f"r2e.{s}.range"] = f"{f(v['R_min'], 2)}--{f(v['R_max'], 2)}"
N["r2g.lookup"], N["r2g.overlap"] = f(X2["R2g_ttc"]["lookup"]), pct(X2["R2g_ttc"]["test_overlap"], 1)
for k, key in (("Product reviews", "rev"), ("Sentiment (test)", "sen"), ("TTC-4900", "ttc")):
    v = X2["R2f"][k]
    N[f"r2f.{key}.short"] = pct(v["share_le10_tokens"], 0)
    if "long_excess10" in v:
        N[f"r2f.{key}.excess"] = f(v["long_excess10"], 2)

(ROOT / "paper" / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False))

# ------------------------------------------------------------------ tables
def letters(r):
    c = r["certificate"]["checks"]
    m = [("L1_template_uniqueness", "D"), ("L1_near_duplicates", "N"), ("L1_label_conflicts", "C"),
         ("L2_surface_views", "S"), ("L3_lookup", "L")]
    return ", ".join(x for k, x in m if not c[k]) or "none"


rows = []
for k, name in CORP:
    if k == "turkphish_v2_body":
        continue
    L1 = R[k]["layer1"]
    sw = RES.get({"turkphish_v1": "TurkPhish v1", "ttc4900": "TTC-4900", "urun_yorum": "Product reviews",
                  "duygu": "Sentiment (test)"}.get(k, ""), {}).get("threshold_sweep")
    sweep = " / ".join(thou(sw[f"pairs_j{t}"]) for t in (0.7, 0.8, 0.9)) if sw else "0 / 0 / 0"
    m = RES.get({"turkphish_v1": "TurkPhish v1", "turkphish_v2": "TurkPhish v2", "ttc4900": "TTC-4900",
                 "urun_yorum": "Product reviews", "duygu": "Sentiment (test)"}[k], {}).get("masking")
    rows.append(f"{name} & {f(m['lowercase_only'])} / {f(m['masked'])} & {thou(L1['duplicate_groups'])} ({thou(L1['documents_in_duplicate_groups'])}) & "
                f"{thou(L1['label_conflicting_duplicate_groups'])} & {sweep} & {thou(L1['near_dup_pairs_non_identical'])} & "
                f"{thou(L1['near_dup_pairs_cross_label'])} & {thou(L1['split_groups'])} \\\\")
(TAB / "layer1.tex").write_text("\n".join(rows) + "\n")

rows = []
for k, name in CORP:
    L3 = R[k]["layer3"]
    mark = ("$^{b}$" if L3["split"] != "grouped" and L3["split"] != "user" else "") + ("$^{c}$" if L3["lookup_fallback_rate"] > 0.9 else "")
    rows.append(f"{name}{mark} & {f(L3['lookup_rule_macro_f1'])} & {f(L3['majority_constant_macro_f1'])} & {pct(L3['lookup_fallback_rate'])} & "
                f"{f(L3['trained_reference_macro_f1'])} & {f(L3['normalized_lookup_recovery'], 2)} {ci(L3['normalized_lookup_recovery_ci95'])} & "
                f"{letters(R[k])} \\\\")
(TAB / "layer3.tex").write_text("\n".join(rows) + "\n")

rows = []
for k, key, name in (("TTC-4900", "ttc", "TTC-4900"), ("Product reviews", "rev", "Product reviews"), ("Sentiment (test)", "sen", "Sentiment (test)")):
    rows.append(f"{name} & {N[f'e5.{key}.overlap']} & {N[f'e5.{key}.lr.full']} & {N[f'e5.{key}.lr.dedup']} & {N[f'e5.{key}.lr.mean']} {N[f'e5.{key}.lr.ci']} & "
                f"{N[f'e5.{key}.knn1.full']} & {N[f'e5.{key}.knn1.dedup']} & {N[f'e5.{key}.knn1.mean']} {N[f'e5.{key}.knn1.ci']} \\\\")
(TAB / "inflation.tex").write_text("\n".join(rows) + "\n")

rows = []
for s, v in E7["sources"].items():
    cc = CC["per_source"][s]
    lab = CC["label_by_source"][s]
    tot = sum(lab.values())
    shares = " / ".join(pct(lab.get(c, 0) / tot, 0) for c in ("Positive", "Notr", "Negative"))
    rows.append(f"\\texttt{{{s.replace('_', chr(92) + '_')}}} & {thou(cc['n'])} & {shares} & {f(v['within_source_macro_f1']) if 'within_source_macro_f1' in v else 'n/a'} & "
                f"{thou(cc['in_reviews'])} \\\\")
(TAB / "sources.tex").write_text("\n".join(rows) + "\n")

# ------------------------------------------------------------------ certificate profiles
sys.path.insert(0, str(ROOT / "src"))
from leakaudit.audit import content_aware_certificate  # noqa: E402

X3 = json.load(open(EXP / "exp_r3.json"))
RC = {"ttc4900_clean": json.load(open(EXP / "ttc4900_clean_report.json")),
      "urun_yorum_clean": json.load(open(EXP / "urun_yorum_clean_report.json"))}
LET = [("L1_template_uniqueness", "D"), ("L1_near_duplicates", "N"), ("L1_label_conflicts", "C"),
       ("L2_surface_views", "S"), ("L3_lookup", "L")]


def prof_row(name, r):
    L1, L2, L3 = r["layer1"], r["layer2"], r["layer3"]
    v = L2["views"]
    rw = v["random_window_10_tokens"]["normalized_recovery"]
    exc = max(d["normalized_recovery"] - rw for k, d in v.items() if k.startswith("first_"))
    oth = max(d["normalized_recovery"] for k, d in v.items() if not k.startswith("first_") and k != "random_window_10_tokens")
    ca = content_aware_certificate(r)
    strict = ", ".join(x for k, x in LET if not r["certificate"]["checks"][k]) or "none"
    cont = ", ".join(x for k, x in LET if not ca["checks"][k]) or "none"
    inf = "" if ca["lookup_informative"] else "$^{c}$"
    return (f"{name} & {L1['near_dup_pairs_per_1000_docs']:,.1f} & {f(1000 * L1['label_conflicting_duplicate_groups'] / r['n_documents'], 2)} & "
            f"{f(exc, 2)} & {f(oth, 2)} & {f(L3['normalized_lookup_recovery'], 2)}{inf} & {strict} & {cont} \\\\"), ca["passed"]


rows = []
for k, name in CORP:
    rows.append(prof_row(name, R[k])[0])
rows.append("\\hline")
for k, name in (("ttc4900_clean", "TTC-4900, cleaned"), ("urun_yorum_clean", "Product reviews, cleaned")):
    rows.append(prof_row(name, RC[k])[0])
(TAB / "profiles.tex").write_text("\n".join(rows) + "\n")
for k, key in (("ttc4900_clean", "ttc"), ("urun_yorum_clean", "rev")):
    d = X3[k]
    N[f"r3.{key}.before"], N[f"r3.{key}.after"] = thou(d["n_before"]), thou(d["n_after"])
    N[f"r3.{key}.conflictdocs"] = thou(d["removed_in_conflict_groups"])
    N[f"r3.{key}.full"] = f(d["full_f1"])
    N[f"r3.{key}.excess10"] = f(d["first10_R"] - d["random_window_R"], 2)
DC = json.load(open(EXP / "deepchecks_r3.json"))
N["dc.ttc.groups"] = thou(DC["TTC-4900"]["TextDuplicates"]["n_duplicate_groups"])
N["dc.sen.groups"] = thou(DC["Sentiment (test)"]["TextDuplicates"]["n_duplicate_groups"])
N["dc.ttc.mix"] = pct(DC["TTC-4900"]["TrainTestSamplesMix_random_split"]["ratio"], 1)
N["dc.ttc.conf"] = thou(round(DC["TTC-4900"]["ConflictingLabels"]["percent_of_conflicting_samples"] * 4900))
N["ca.th.excess"] = "0.25"
X5 = json.load(open(EXP / "exp_r5.json"))
N["r5.n_seeds"] = str(len(X5["runs"]))
N["r5.f1"] = f(X5["summary"]["random_removed_mean"])
N["r5.dup"] = f(X5["summary"]["duplication_effect_mean_pts"], 2)
N["r5.dup_ci"] = ci(X5["summary"]["duplication_effect_ci95_pts"])
N["r5.range"] = f"{f(min(r['duplication_effect_pts'] for r in X5['runs']), 2)} to {f(max(r['duplication_effect_pts'] for r in X5['runs']), 2)}"
X4 = json.load(open(EXP / "exp_r4.json"))
N["r4.n_train"] = thou(X4["n_train"])
N["r4.uniq"] = f(X4["train_uniqueness"])
N["r4.dupgroups"] = thou(X4["train_dup_groups"])
N["r4.dupdocs"] = thou(X4["train_docs_in_dup_groups"])
N["r4.conf"] = thou(X4["train_conflict_groups"])
N["r4.te_in_tr"] = thou(X4["test_in_train"])
N["r4.te_in_tr_share"] = pct(X4["test_in_train_share"], 2)
N["r4.te_in_tr_agree"] = pct(X4["test_in_train_label_agreement_strict"], 1)
N["r4.tr_in_rev"] = thou(X4["train_in_reviews"])
N["r4.tr_in_rev_share"] = pct(X4["train_in_reviews_share"], 1)
N["r4.urun.n"] = thou(X4["train_per_source"]["urun_yorumlari"]["n"])
N["r4.urun.in"] = thou(X4["train_per_source"]["urun_yorumlari"]["in_reviews"])
N["ca.passed_raw"] = str(sum(content_aware_certificate(R[k])["passed"] for k, _ in CORP))
(ROOT / "paper" / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False))

# ------------------------------------------------------------------ manuscript
src = MS / "leakaudit_main.src.tex"
if src.exists():
    text = src.read_text()
    missing = sorted(set(re.findall(r"\{\{([^}]+)\}\}", text)) - set(N))
    if missing:
        sys.exit("unknown placeholders: " + ", ".join(missing))
    text = re.sub(r"\{\{([^}]+)\}\}", lambda m: N[m.group(1)], text)
    # inline the generated table rows (\input inside tabular breaks a following \hline)
    text = re.sub(r"\\input\{tables/([a-z0-9]+)\.tex\}\n", lambda m: (TAB / f"{m.group(1)}.tex").read_text(), text)
    (MS / "leakaudit_main.tex").write_text(text)
print(f"{len(N)} numbers; tables written to {TAB}")

tsrc = MS / "leakaudit_titlepage.src.tex"
main = MS / "leakaudit_main.tex"
if tsrc.exists() and main.exists():
    m = main.read_text()
    ab = m[m.index("\\begin{abstract}"):m.index("\\end{abstract}") + len("\\end{abstract}")]
    (MS / "leakaudit_titlepage.tex").write_text(tsrc.read_text().replace("%%ABSTRACT%%", ab))
