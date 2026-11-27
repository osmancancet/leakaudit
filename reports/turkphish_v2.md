# LeakAudit report: TurkPhish v2

**Certificate: FAILED**  
2,653 documents, 2 classes; layers 2-3 on 2,653 documents; seed 42; leakaudit 0.2.0.

Reasons:

- view 'first_10_tokens' recovers 100% of above-chance full-text performance
- learning-free sentence lookup reaches 0.9008 macro-F1 and recovers 80% of the trained model's above-chance performance

Content-aware profile (rates per 1,000 documents, opening views relative to a random window): FAILED; failed checks: L2_surface_views, L3_lookup

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 1.0000 | >= 0.95 | PASS |
| Near-duplicate pairs (Jaccard >= 0.8) | 0 | <= 0 | PASS |
| Duplicate groups with conflicting labels | 0 | <= 0 | PASS |

Distinct masked templates: 2,653; duplicate groups: 0 covering 0 documents; near-duplicate pairs crossing labels: 0; identical / non-identical pairs: 0 / 0 (sample of 2,653; 0.0 pairs per 1,000 documents; 2,653 duplicate-connected groups used for splits).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV, grouped folds): 0.9989 (random folds: 0.9985); chance: 0.5000. Normalized recovery = (view - chance) / (full text - chance), unclipped; the random-window view is a control and is not gated.

| View | Macro-F1 | Recovery ratio | Normalized recovery | 95% CI |
|---|---:|---:|---:|---|
| first_3_tokens | 0.5281 | 0.529 | 0.056 | 0.019 to 0.098 |
| first_5_tokens | 0.8573 | 0.858 | 0.716 | 0.689 to 0.742 |
| first_10_tokens | 0.9985 | 1.000 | 0.999 | 0.996 to 1.002 |
| random_window_3_tokens | 0.6056 | 0.606 | 0.212 | 0.172 to 0.249 |
| random_window_5_tokens | 0.6433 | 0.644 | 0.287 | 0.252 to 0.323 |
| random_window_10_tokens | 0.7320 | 0.733 | 0.465 | 0.430 to 0.497 |
| punctuation_emoji_case_profile | 0.4978 | 0.498 | -0.004 | -0.044 to 0.029 |
| length_only | 0.4881 | 0.489 | -0.024 | -0.067 to 0.011 |

Max normalized recovery: 0.999 (first_10_tokens); threshold <= 0.5: FAIL.
Length distribution by class: KS D = 0.0153, p = 0.9968.

## Layer 3: sentences

Learning-free lookup over class-exclusive sentences (grouped split): **0.9008** macro-F1 (fallback to majority class on 10.0% of test documents; majority constant 0.3387); trained full-text reference: 1.0000; margin 0.0992; normalized lookup recovery 0.802 (<= 0.5 required): FAIL.

| Class | Sentence types | Class-exclusive | Test documents with a top-40 exclusive sentence |
|---|---:|---:|---:|
| Guvenilir | 1,736 | 1,542 | 0.768 |
| Oltalama | 1,904 | 1,710 | 0.868 |
