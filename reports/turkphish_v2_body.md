# LeakAudit report: TurkPhish v2 (body only, subject as metadata)

**Certificate: FAILED**  
2,653 documents, 2 classes; layers 2-3 on 2,653 documents; seed 42; leakaudit 0.2.0.

Reasons:

- view 'metadata:konu' recovers 105% of above-chance full-text performance
- learning-free sentence lookup reaches 0.8918 macro-F1 and recovers 82% of the trained model's above-chance performance

Content-aware profile (rates per 1,000 documents, opening views relative to a random window): FAILED; failed checks: L2_surface_views, L3_lookup

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 1.0000 | >= 0.95 | PASS |
| Near-duplicate pairs (Jaccard >= 0.8) | 0 | <= 0 | PASS |
| Duplicate groups with conflicting labels | 0 | <= 0 | PASS |

Distinct masked templates: 2,653; duplicate groups: 0 covering 0 documents; near-duplicate pairs crossing labels: 0; identical / non-identical pairs: 0 / 0 (sample of 2,653; 0.0 pairs per 1,000 documents; 2,653 duplicate-connected groups used for splits).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV, grouped folds): 0.9763 (random folds: 0.9755); chance: 0.5000. Normalized recovery = (view - chance) / (full text - chance), unclipped; the random-window view is a control and is not gated.

| View | Macro-F1 | Recovery ratio | Normalized recovery | 95% CI |
|---|---:|---:|---:|---|
| first_3_tokens | 0.5030 | 0.515 | 0.006 | -0.034 to 0.047 |
| first_5_tokens | 0.4883 | 0.500 | -0.025 | -0.065 to 0.013 |
| first_10_tokens | 0.5612 | 0.575 | 0.129 | 0.088 to 0.167 |
| random_window_3_tokens | 0.5909 | 0.605 | 0.191 | 0.154 to 0.228 |
| random_window_5_tokens | 0.6195 | 0.635 | 0.251 | 0.214 to 0.290 |
| random_window_10_tokens | 0.7144 | 0.732 | 0.450 | 0.415 to 0.487 |
| punctuation_emoji_case_profile | 0.4881 | 0.500 | -0.025 | -0.067 to 0.015 |
| length_only | 0.5000 | 0.512 | 0.000 | -0.043 to 0.039 |
| metadata:konu | 0.9992 | 1.024 | 1.048 | 1.036 to 1.061 |

Max normalized recovery: 1.048 (metadata:konu); threshold <= 0.5: FAIL.
Length distribution by class: KS D = 0.0145, p = 0.9986.

## Layer 3: sentences

Learning-free lookup over class-exclusive sentences (grouped split): **0.8918** macro-F1 (fallback to majority class on 7.3% of test documents; majority constant 0.3387); trained full-text reference: 0.9793; margin 0.0875; normalized lookup recovery 0.817 (<= 0.5 required): FAIL.

| Class | Sentence types | Class-exclusive | Test documents with a top-40 exclusive sentence |
|---|---:|---:|---:|
| Guvenilir | 1,496 | 1,045 | 0.768 |
| Oltalama | 1,663 | 1,212 | 0.868 |
