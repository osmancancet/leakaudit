# LeakAudit report: DBpedia (training split)

**Certificate: FAILED**  
560,000 documents, 14 classes; layers 2-3 on 20,000 documents; seed 42; leakaudit 0.2.0.

Reasons:

- 1575 near-duplicate pairs (Jaccard >= 0.8)
- view 'first_10_tokens' recovers 83% of above-chance full-text performance

Content-aware profile (rates per 1,000 documents, opening views relative to a random window): FAILED; failed checks: L1_near_duplicates

Warnings:

- the sentence lookup fell back to the majority class for 93% of test documents; check L3 is not informative here

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 0.9995 | >= 0.95 | PASS |
| Near-duplicate pairs (Jaccard >= 0.8) | 1,575 | <= 0 | FAIL |
| Duplicate groups with conflicting labels | 0 | <= 0 | PASS |

Distinct masked templates: 559,720; duplicate groups: 174 covering 454 documents; near-duplicate pairs crossing labels: 0; identical / non-identical pairs: 7 / 1,568 (sample of 50,000; 31.5 pairs per 1,000 documents; 558,913 duplicate-connected groups used for splits).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV, grouped folds): 0.9695 (random folds: 0.9702); chance: 0.0714. Normalized recovery = (view - chance) / (full text - chance), unclipped; the random-window view is a control and is not gated.

| View | Macro-F1 | Recovery ratio | Normalized recovery | 95% CI |
|---|---:|---:|---:|---|
| first_3_tokens | 0.4933 | 0.509 | 0.470 | 0.462 to 0.477 |
| first_5_tokens | 0.6479 | 0.668 | 0.642 | 0.635 to 0.649 |
| first_10_tokens | 0.8150 | 0.841 | 0.828 | 0.823 to 0.834 |
| random_window_3_tokens | 0.4542 | 0.469 | 0.426 | 0.418 to 0.433 |
| random_window_5_tokens | 0.5816 | 0.600 | 0.568 | 0.561 to 0.576 |
| random_window_10_tokens | 0.7360 | 0.759 | 0.740 | 0.733 to 0.746 |
| punctuation_emoji_case_profile | 0.0863 | 0.089 | 0.017 | 0.013 to 0.020 |
| length_only | 0.0888 | 0.092 | 0.019 | 0.015 to 0.022 |

Max normalized recovery: 0.828 (first_10_tokens); threshold <= 0.5: FAIL.

## Layer 3: sentences

Learning-free lookup over class-exclusive sentences (grouped split): **0.1177** macro-F1 (fallback to majority class on 93.2% of test documents; majority constant 0.0099); trained full-text reference: 0.9647; margin 0.8470; normalized lookup recovery 0.052 (<= 0.5 required): PASS.

| Class | Sentence types | Class-exclusive | Test documents with a top-40 exclusive sentence |
|---|---:|---:|---:|
| 0 | 2,970 | 2,968 | 0.000 |
| 1 | 2,764 | 2,761 | 0.015 |
| 10 | 2,776 | 2,763 | 0.105 |
| 11 | 2,767 | 2,765 | 0.000 |
| 12 | 2,701 | 2,699 | 0.018 |
| 13 | 2,852 | 2,849 | 0.011 |
| 2 | 2,721 | 2,720 | 0.000 |
| 3 | 2,511 | 2,510 | 0.000 |
| 4 | 2,916 | 2,915 | 0.003 |
| 5 | 3,034 | 3,033 | 0.017 |
| 6 | 3,311 | 3,309 | 0.055 |
| 7 | 2,458 | 2,457 | 0.011 |
| 8 | 1,702 | 1,702 | 0.440 |
| 9 | 2,491 | 2,478 | 0.079 |
