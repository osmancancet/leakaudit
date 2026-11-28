# LeakAudit report: TurkPhish v1

**Certificate: FAILED**  
7,504 documents, 2 classes; layers 2-3 on 7,504 documents; seed 42; leakaudit 0.2.0.

Reasons:

- template uniqueness 0.0079 < 0.95
- 4478487 near-duplicate pairs (Jaccard >= 0.8)
- view 'first_5_tokens' recovers 100% of above-chance full-text performance
- learning-free sentence lookup reaches 1.0 macro-F1 and recovers 100% of the trained model's above-chance performance

Content-aware profile (rates per 1,000 documents, opening views relative to a random window): FAILED; failed checks: L1_template_uniqueness, L1_near_duplicates, L2_surface_views, L3_lookup

Warnings:

- no grouped partition places every class in every fold; layers 2 and 3 used random folds, so their scores include memorization of duplicates

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 0.0079 | >= 0.95 | FAIL |
| Near-duplicate pairs (Jaccard >= 0.8) | 4,478,487 | <= 0 | FAIL |
| Duplicate groups with conflicting labels | 0 | <= 0 | PASS |

Distinct masked templates: 59; duplicate groups: 55 covering 7,500 documents; near-duplicate pairs crossing labels: 0; identical / non-identical pairs: 639,934 / 3,838,553 (sample of 7,504; 596813.3 pairs per 1,000 documents; 6 duplicate-connected groups used for splits).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV, random (grouped partition infeasible) folds): 1.0000 (random folds: 1.0000); chance: 0.5000. Normalized recovery = (view - chance) / (full text - chance), unclipped; the random-window view is a control and is not gated.

| View | Macro-F1 | Recovery ratio | Normalized recovery | 95% CI |
|---|---:|---:|---:|---|
| first_3_tokens | 0.9206 | 0.921 | 0.841 | -0.200 to 1.000 |
| first_5_tokens | 1.0000 | 1.000 | 1.000 | 1.000 to 1.000 |
| first_10_tokens | 1.0000 | 1.000 | 1.000 | 1.000 to 1.000 |
| random_window_3_tokens | 0.9936 | 0.994 | 0.987 | -0.200 to 0.992 |
| random_window_5_tokens | 0.9996 | 1.000 | 0.999 | -0.500 to 1.000 |
| random_window_10_tokens | 0.9998 | 1.000 | 1.000 | -0.200 to 1.000 |
| punctuation_emoji_case_profile | 0.6074 | 0.607 | 0.215 | -0.714 to 0.262 |
| length_only | 0.8248 | 0.825 | 0.650 | -0.714 to 0.732 |
| metadata:Gönderen | 0.9998 | 1.000 | 1.000 | -0.200 to 1.000 |

Max normalized recovery: 1.000 (first_5_tokens); threshold <= 0.5: FAIL.
Length distribution by class: KS D = 0.7661, p = 0.0.

## Layer 3: sentences

Learning-free lookup over class-exclusive sentences (random (grouped partition infeasible) split): **1.0000** macro-F1 (fallback to majority class on 0.1% of test documents; majority constant 0.4445); trained full-text reference: 1.0000; margin 0.0000; normalized lookup recovery 1.000 (<= 0.5 required): FAIL.

| Class | Sentence types | Class-exclusive | Test documents with a top-40 exclusive sentence |
|---|---:|---:|---:|
| Güvenilir | 11 | 11 | 1.000 |
| Oltalama | 39 | 39 | 0.999 |
