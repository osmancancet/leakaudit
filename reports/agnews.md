# LeakAudit report: AG News (training split)

**Certificate: FAILED**  
120,000 documents, 4 classes; layers 2-3 on 20,000 documents; seed 42; leakaudit 0.2.0.

Reasons:

- 903 near-duplicate pairs (Jaccard >= 0.8)
- 22 duplicate groups carry conflicting labels
- view 'first_10_tokens' recovers 87% of above-chance full-text performance

Content-aware profile (rates per 1,000 documents, opening views relative to a random window): FAILED; failed checks: L1_near_duplicates, L2_surface_views

Warnings:

- the sentence lookup fell back to the majority class for 99% of test documents; check L3 is not informative here

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 0.9983 | >= 0.95 | PASS |
| Near-duplicate pairs (Jaccard >= 0.8) | 903 | <= 0 | FAIL |
| Duplicate groups with conflicting labels | 22 | <= 0 | FAIL |

Distinct masked templates: 119,791; duplicate groups: 196 covering 405 documents; near-duplicate pairs crossing labels: 89; identical / non-identical pairs: 41 / 862 (sample of 50,000; 18.06 pairs per 1,000 documents; 118,955 duplicate-connected groups used for splits).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV, grouped folds): 0.9001 (random folds: 0.9022); chance: 0.2500. Normalized recovery = (view - chance) / (full text - chance), unclipped; the random-window view is a control and is not gated.

| View | Macro-F1 | Recovery ratio | Normalized recovery | 95% CI |
|---|---:|---:|---:|---|
| first_3_tokens | 0.7030 | 0.781 | 0.697 | 0.687 to 0.706 |
| first_5_tokens | 0.7494 | 0.833 | 0.768 | 0.759 to 0.777 |
| first_10_tokens | 0.8162 | 0.907 | 0.871 | 0.863 to 0.878 |
| random_window_3_tokens | 0.5337 | 0.593 | 0.436 | 0.426 to 0.446 |
| random_window_5_tokens | 0.6223 | 0.691 | 0.573 | 0.563 to 0.582 |
| random_window_10_tokens | 0.7322 | 0.814 | 0.742 | 0.734 to 0.751 |
| punctuation_emoji_case_profile | 0.2936 | 0.326 | 0.067 | 0.058 to 0.076 |
| length_only | 0.2784 | 0.309 | 0.044 | 0.035 to 0.053 |

Max normalized recovery: 0.871 (first_10_tokens); threshold <= 0.5: FAIL.

## Layer 3: sentences

Learning-free lookup over class-exclusive sentences (grouped split): **0.1128** macro-F1 (fallback to majority class on 99.2% of test documents; majority constant 0.1013); trained full-text reference: 0.8917; margin 0.7789; normalized lookup recovery -0.214 (<= 0.5 required): PASS.

| Class | Sentence types | Class-exclusive | Test documents with a top-40 exclusive sentence |
|---|---:|---:|---:|
| 0 | 5,151 | 5,145 | 0.000 |
| 1 | 5,988 | 5,986 | 0.002 |
| 2 | 6,079 | 6,067 | 0.000 |
| 3 | 6,021 | 6,007 | 0.010 |
