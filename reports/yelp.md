# LeakAudit report: Yelp polarity (training split)

**Certificate: FAILED**  
560,000 documents, 2 classes; layers 2-3 on 20,000 documents; seed 42; leakaudit 0.2.0.

Reasons:

- 7 near-duplicate pairs (Jaccard >= 0.8)
- 3 duplicate groups carry conflicting labels
- view 'first_10_tokens' recovers 52% of above-chance full-text performance

Content-aware profile (rates per 1,000 documents, opening views relative to a random window): PASSED; failed checks: none

Warnings:

- the sentence lookup fell back to the majority class for 95% of test documents; check L3 is not informative here

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 0.9999 | >= 0.95 | PASS |
| Near-duplicate pairs (Jaccard >= 0.8) | 7 | <= 0 | FAIL |
| Duplicate groups with conflicting labels | 3 | <= 0 | FAIL |

Distinct masked templates: 559,922; duplicate groups: 71 covering 149 documents; near-duplicate pairs crossing labels: 0; identical / non-identical pairs: 0 / 7 (sample of 50,000; 0.14 pairs per 1,000 documents; 559,915 duplicate-connected groups used for splits).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV, grouped folds): 0.9219 (random folds: 0.9202); chance: 0.5000. Normalized recovery = (view - chance) / (full text - chance), unclipped; the random-window view is a control and is not gated.

| View | Macro-F1 | Recovery ratio | Normalized recovery | 95% CI |
|---|---:|---:|---:|---|
| first_3_tokens | 0.6588 | 0.715 | 0.377 | 0.361 to 0.393 |
| first_5_tokens | 0.6854 | 0.744 | 0.440 | 0.425 to 0.456 |
| first_10_tokens | 0.7190 | 0.780 | 0.519 | 0.504 to 0.534 |
| random_window_3_tokens | 0.6112 | 0.663 | 0.264 | 0.246 to 0.278 |
| random_window_5_tokens | 0.6422 | 0.697 | 0.337 | 0.321 to 0.352 |
| random_window_10_tokens | 0.6971 | 0.756 | 0.467 | 0.454 to 0.482 |
| punctuation_emoji_case_profile | 0.4521 | 0.490 | -0.114 | -0.128 to -0.099 |
| length_only | 0.5683 | 0.617 | 0.162 | 0.144 to 0.178 |

Max normalized recovery: 0.519 (first_10_tokens); threshold <= 0.5: FAIL.
Length distribution by class: KS D = 0.1378, p = 0.0.

## Layer 3: sentences

Learning-free lookup over class-exclusive sentences (grouped split): **0.3777** macro-F1 (fallback to majority class on 94.9% of test documents; majority constant 0.3352); trained full-text reference: 0.9195; margin 0.5418; normalized lookup recovery -0.291 (<= 0.5 required): PASS.

| Class | Sentence types | Class-exclusive | Test documents with a top-40 exclusive sentence |
|---|---:|---:|---:|
| 0 | 71,346 | 71,300 | 0.010 |
| 1 | 56,181 | 56,135 | 0.019 |
