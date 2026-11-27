# LeakAudit report: Turkish product reviews (sentiment)

**Certificate: FAILED**  
235,144 documents, 2 classes; layers 2-3 on 20,000 documents; seed 42; leakaudit 0.2.0.

Reasons:

- 652 near-duplicate pairs (Jaccard >= 0.8)
- 77 duplicate groups carry conflicting labels
- view 'first_10_tokens' recovers 62% of above-chance full-text performance

Content-aware profile (rates per 1,000 documents, opening views relative to a random window): FAILED; failed checks: L1_near_duplicates

Warnings:

- the sentence lookup fell back to the majority class for 98% of test documents; check L3 is not informative here

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 0.9920 | >= 0.95 | PASS |
| Near-duplicate pairs (Jaccard >= 0.8) | 652 | <= 0 | FAIL |
| Duplicate groups with conflicting labels | 77 | <= 0 | FAIL |

Distinct masked templates: 233,257; duplicate groups: 1,145 covering 3,032 documents; near-duplicate pairs crossing labels: 26; identical / non-identical pairs: 175 / 477 (sample of 50,000; 13.04 pairs per 1,000 documents; 233,017 duplicate-connected groups used for splits).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV, grouped folds): 0.7453 (random folds: 0.7408); chance: 0.5000. Normalized recovery = (view - chance) / (full text - chance), unclipped; the random-window view is a control and is not gated.

| View | Macro-F1 | Recovery ratio | Normalized recovery | 95% CI |
|---|---:|---:|---:|---|
| first_3_tokens | 0.5668 | 0.760 | 0.273 | 0.237 to 0.306 |
| first_5_tokens | 0.6026 | 0.809 | 0.418 | 0.382 to 0.460 |
| first_10_tokens | 0.6523 | 0.875 | 0.621 | 0.582 to 0.665 |
| random_window_10_tokens | 0.6412 | 0.860 | 0.576 | 0.535 to 0.617 |
| punctuation_emoji_case_profile | 0.1578 | 0.212 | -1.395 | -1.468 to -1.323 |
| length_only | 0.4119 | 0.553 | -0.359 | -0.393 to -0.328 |

Max normalized recovery: 0.621 (first_10_tokens); threshold <= 0.5: FAIL.
Length distribution by class: KS D = 0.0226, p = 0.5778.

## Layer 3: sentences

Learning-free lookup over class-exclusive sentences (grouped split): **0.4836** macro-F1 (fallback to majority class on 98.0% of test documents; majority constant 0.4838); trained full-text reference: 0.7435; margin 0.2599; normalized lookup recovery -0.067 (<= 0.5 required): PASS.

| Class | Sentence types | Class-exclusive | Test documents with a top-40 exclusive sentence |
|---|---:|---:|---:|
| 0 | 1,828 | 1,823 | 0.000 |
| 1 | 28,035 | 28,030 | 0.006 |
