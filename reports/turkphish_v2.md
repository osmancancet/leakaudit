# LeakAudit report: TurkPhish v2

**Certificate: FAILED**  
2,653 documents, 2 classes; layers 2-3 on 2,653 documents; seed 42; leakaudit 0.1.0.

Reasons:

- view 'first_10_tokens' recovers 100% of above-chance full-text performance

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 1.0000 | >= 0.95 | PASS |
| Near-duplicate pairs (Jaccard >= 0.8) | 0 | <= 0 | PASS |
| Duplicate groups with conflicting labels | 0 | <= 0 | PASS |

Distinct masked templates: 2,653; duplicate groups: 0 covering 0 documents; near-duplicate pairs crossing labels: 0 (sample of 2,653).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV): 0.9985; chance: 0.5000. Normalized recovery = (view - chance) / (full text - chance).

| View | Macro-F1 | Recovery ratio | Normalized recovery |
|---|---:|---:|---:|
| first_3_tokens | 0.5281 | 0.529 | 0.056 |
| first_5_tokens | 0.8608 | 0.862 | 0.724 |
| first_10_tokens | 0.9989 | 1.000 | 1.001 |
| punctuation_emoji_case_profile | 0.4764 | 0.477 | 0.000 |
| length_only | 0.4963 | 0.497 | 0.000 |

Max normalized recovery: 1.001 (first_10_tokens); threshold <= 0.5: FAIL.
Length distribution by class: KS D = 0.0153, p = 0.9968.

## Layer 3: sentences

Zero-learning lookup over class-exclusive sentences: **0.8913** macro-F1; trained full-text reference: 0.9981; margin 0.1068 (>= 0.1 required): PASS.

| Class | Sentence types | Class-exclusive | Top-40 exclusive coverage |
|---|---:|---:|---:|
| Guvenilir | 1,734 | 1,535 | 0.825 |
| Oltalama | 1,867 | 1,668 | 1.000 |
