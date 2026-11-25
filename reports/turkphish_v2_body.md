# LeakAudit report: TurkPhish v2 (yalnız gövde, konu üstveri)

**Certificate: FAILED**  
2,653 documents, 2 classes; layers 2-3 on 2,653 documents; seed 42; leakaudit 0.1.0.

Reasons:

- view 'metadata:konu' recovers 105% of above-chance full-text performance
- zero-learning sentence lookup reaches 0.8863 macro-F1; trained model margin only 0.0911

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 1.0000 | >= 0.95 | PASS |
| Near-duplicate pairs (Jaccard >= 0.8) | 0 | <= 0 | PASS |
| Duplicate groups with conflicting labels | 0 | <= 0 | PASS |

Distinct masked templates: 2,653; duplicate groups: 0 covering 0 documents; near-duplicate pairs crossing labels: 0 (sample of 2,653).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV): 0.9755; chance: 0.5000. Normalized recovery = (view - chance) / (full text - chance).

| View | Macro-F1 | Recovery ratio | Normalized recovery |
|---|---:|---:|---:|
| first_3_tokens | 0.4915 | 0.504 | 0.000 |
| first_5_tokens | 0.4883 | 0.500 | 0.000 |
| first_10_tokens | 0.5510 | 0.565 | 0.107 |
| punctuation_emoji_case_profile | 0.4682 | 0.480 | 0.000 |
| length_only | 0.4896 | 0.502 | 0.000 |
| metadata:konu | 0.9989 | 1.024 | 1.049 |

Max normalized recovery: 1.049 (metadata:konu); threshold <= 0.5: FAIL.
Length distribution by class: KS D = 0.0145, p = 0.9986.

## Layer 3: sentences

Zero-learning lookup over class-exclusive sentences: **0.8863** macro-F1; trained full-text reference: 0.9774; margin 0.0911 (>= 0.1 required): FAIL.

| Class | Sentence types | Class-exclusive | Top-40 exclusive coverage |
|---|---:|---:|---:|
| Guvenilir | 1,513 | 1,064 | 0.825 |
| Oltalama | 1,623 | 1,174 | 1.000 |
