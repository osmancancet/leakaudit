# LeakAudit report: Turkish sentiment analysis dataset (test split)

**Certificate: FAILED**  
48,965 documents, 3 classes; layers 2-3 on 20,000 documents; seed 42; leakaudit 0.2.0.

Reasons:

- 135 near-duplicate pairs (Jaccard >= 0.8)
- 8 duplicate groups carry conflicting labels
- view 'metadata:dataset' recovers 88% of above-chance full-text performance

Content-aware profile (rates per 1,000 documents, opening views relative to a random window): FAILED; failed checks: L1_near_duplicates, L2_surface_views

Warnings:

- the sentence lookup fell back to the majority class for 99% of test documents; check L3 is not informative here

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 0.9991 | >= 0.95 | PASS |
| Near-duplicate pairs (Jaccard >= 0.8) | 135 | <= 0 | FAIL |
| Duplicate groups with conflicting labels | 8 | <= 0 | FAIL |

Distinct masked templates: 48,920; duplicate groups: 39 covering 84 documents; near-duplicate pairs crossing labels: 15; identical / non-identical pairs: 53 / 82 (sample of 48,965; 2.757 pairs per 1,000 documents; 48,851 duplicate-connected groups used for splits).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV, grouped folds): 0.8880 (random folds: 0.8880); chance: 0.3333. Normalized recovery = (view - chance) / (full text - chance), unclipped; the random-window view is a control and is not gated.

| View | Macro-F1 | Recovery ratio | Normalized recovery | 95% CI |
|---|---:|---:|---:|---|
| first_3_tokens | 0.6915 | 0.779 | 0.646 | 0.632 to 0.658 |
| first_5_tokens | 0.7363 | 0.829 | 0.727 | 0.714 to 0.738 |
| first_10_tokens | 0.8078 | 0.910 | 0.855 | 0.846 to 0.866 |
| random_window_3_tokens | 0.6430 | 0.724 | 0.558 | 0.545 to 0.572 |
| random_window_5_tokens | 0.7073 | 0.796 | 0.674 | 0.662 to 0.686 |
| random_window_10_tokens | 0.7867 | 0.886 | 0.817 | 0.806 to 0.828 |
| punctuation_emoji_case_profile | 0.2439 | 0.275 | -0.161 | -0.173 to -0.150 |
| length_only | 0.5319 | 0.599 | 0.358 | 0.343 to 0.371 |
| metadata:dataset | 0.8224 | 0.926 | 0.882 | 0.871 to 0.892 |

Max normalized recovery: 0.882 (metadata:dataset); threshold <= 0.5: FAIL.

## Layer 3: sentences

Learning-free lookup over class-exclusive sentences (grouped split): **0.2340** macro-F1 (fallback to majority class on 99.1% of test documents; majority constant 0.2326); trained full-text reference: 0.8934; margin 0.6594; normalized lookup recovery -0.177 (<= 0.5 required): PASS.

| Class | Sentence types | Class-exclusive | Test documents with a top-40 exclusive sentence |
|---|---:|---:|---:|
| Negative | 3,510 | 3,506 | 0.002 |
| Notr | 5,489 | 5,489 | 0.000 |
| Positive | 15,984 | 15,980 | 0.003 |
