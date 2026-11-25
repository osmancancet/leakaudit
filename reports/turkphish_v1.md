# LeakAudit report: TurkPhish v1

**Certificate: FAILED**  
7,504 documents, 2 classes; layers 2-3 on 7,504 documents; seed 42; leakaudit 0.1.0.

Reasons:

- template uniqueness 0.0079 < 0.95
- 336640 near-duplicate pairs (Jaccard >= 0.8)
- view 'first_5_tokens' recovers 100% of above-chance full-text performance
- zero-learning sentence lookup reaches 1.0 macro-F1; trained model margin only 0.0

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 0.0079 | >= 0.95 | FAIL |
| Near-duplicate pairs (Jaccard >= 0.8) | 336,640 | <= 0 | FAIL |
| Duplicate groups with conflicting labels | 0 | <= 0 | PASS |

Distinct masked templates: 59; duplicate groups: 55 covering 7,500 documents; near-duplicate pairs crossing labels: 0 (sample of 7,504).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV): 1.0000; chance: 0.5000. Normalized recovery = (view - chance) / (full text - chance).

| View | Macro-F1 | Recovery ratio | Normalized recovery |
|---|---:|---:|---:|
| first_3_tokens | 0.9206 | 0.921 | 0.841 |
| first_5_tokens | 1.0000 | 1.000 | 1.000 |
| first_10_tokens | 1.0000 | 1.000 | 1.000 |
| punctuation_emoji_case_profile | 0.6074 | 0.607 | 0.215 |
| length_only | 0.8249 | 0.825 | 0.650 |
| metadata:Gönderen | 0.9998 | 1.000 | 1.000 |

Max normalized recovery: 1.000 (first_5_tokens); threshold <= 0.5: FAIL.
Length distribution by class: KS D = 0.7661, p = 0.0.

## Layer 3: sentences

Zero-learning lookup over class-exclusive sentences: **1.0000** macro-F1; trained full-text reference: 1.0000; margin 0.0000 (>= 0.1 required): FAIL.

| Class | Sentence types | Class-exclusive | Top-40 exclusive coverage |
|---|---:|---:|---:|
| Güvenilir | 11 | 11 | 1.000 |
| Oltalama | 39 | 39 | 1.000 |
