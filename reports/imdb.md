# LeakAudit report: IMDB (training split)

**Certificate: FAILED**  
25,000 documents, 2 classes; layers 2-3 on 20,000 documents; seed 42; leakaudit 0.2.0.

Reasons:

- 123 near-duplicate pairs (Jaccard >= 0.8)

Content-aware profile (rates per 1,000 documents, opening views relative to a random window): FAILED; failed checks: L1_near_duplicates

Warnings:

- the sentence lookup fell back to the majority class for 96% of test documents; check L3 is not informative here

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 0.9962 | >= 0.95 | PASS |
| Near-duplicate pairs (Jaccard >= 0.8) | 123 | <= 0 | FAIL |
| Duplicate groups with conflicting labels | 0 | <= 0 | PASS |

Distinct masked templates: 24,904; duplicate groups: 92 covering 188 documents; near-duplicate pairs crossing labels: 0; identical / non-identical pairs: 100 / 23 (sample of 25,000; 4.92 pairs per 1,000 documents; 24,881 duplicate-connected groups used for splits).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV, grouped folds): 0.8923 (random folds: 0.8933); chance: 0.5000. Normalized recovery = (view - chance) / (full text - chance), unclipped; the random-window view is a control and is not gated.

| View | Macro-F1 | Recovery ratio | Normalized recovery | 95% CI |
|---|---:|---:|---:|---|
| first_3_tokens | 0.5992 | 0.672 | 0.253 | 0.237 to 0.269 |
| first_5_tokens | 0.6269 | 0.703 | 0.324 | 0.308 to 0.340 |
| first_10_tokens | 0.6648 | 0.745 | 0.420 | 0.406 to 0.436 |
| random_window_3_tokens | 0.5551 | 0.622 | 0.140 | 0.122 to 0.157 |
| random_window_5_tokens | 0.5751 | 0.644 | 0.192 | 0.173 to 0.208 |
| random_window_10_tokens | 0.6166 | 0.691 | 0.297 | 0.280 to 0.313 |
| punctuation_emoji_case_profile | 0.4920 | 0.551 | -0.020 | -0.038 to -0.004 |
| length_only | 0.5024 | 0.563 | 0.006 | -0.010 to 0.024 |

Max normalized recovery: 0.420 (first_10_tokens); threshold <= 0.5: PASS.
Length distribution by class: KS D = 0.0238, p = 0.0069.

## Layer 3: sentences

Learning-free lookup over class-exclusive sentences (grouped split): **0.3591** macro-F1 (fallback to majority class on 96.3% of test documents; majority constant 0.3341); trained full-text reference: 0.8932; margin 0.5341; normalized lookup recovery -0.358 (<= 0.5 required): PASS.

| Class | Sentence types | Class-exclusive | Test documents with a top-40 exclusive sentence |
|---|---:|---:|---:|
| 0 | 82,869 | 82,808 | 0.009 |
| 1 | 80,620 | 80,559 | 0.007 |
