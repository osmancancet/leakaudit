# LeakAudit report: Turkish sentiment analysis dataset (test split)

**Certificate: FAILED**  
48,965 documents, 3 classes; layers 2-3 on 20,000 documents; seed 42; leakaudit 0.1.0.

Reasons:

- 120 near-duplicate pairs (Jaccard >= 0.8)
- 6 duplicate groups carry conflicting labels
- view 'metadata:dataset' recovers 88% of above-chance full-text performance

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 0.9992 | >= 0.95 | PASS |
| Near-duplicate pairs (Jaccard >= 0.8) | 120 | <= 0 | FAIL |
| Duplicate groups with conflicting labels | 6 | <= 0 | FAIL |

Distinct masked templates: 48,925; duplicate groups: 34 covering 74 documents; near-duplicate pairs crossing labels: 12 (sample of 48,965).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV): 0.8882; chance: 0.3333. Normalized recovery = (view - chance) / (full text - chance).

| View | Macro-F1 | Recovery ratio | Normalized recovery |
|---|---:|---:|---:|
| first_3_tokens | 0.6909 | 0.778 | 0.644 |
| first_5_tokens | 0.7365 | 0.829 | 0.727 |
| first_10_tokens | 0.8062 | 0.908 | 0.852 |
| punctuation_emoji_case_profile | 0.2421 | 0.273 | 0.000 |
| length_only | 0.2911 | 0.328 | 0.000 |
| metadata:dataset | 0.8224 | 0.926 | 0.881 |

Max normalized recovery: 0.881 (metadata:dataset); threshold <= 0.5: FAIL.

## Layer 3: sentences

Zero-learning lookup over class-exclusive sentences: **0.2340** macro-F1; trained full-text reference: 0.8855; margin 0.6515 (>= 0.1 required): PASS.

| Class | Sentence types | Class-exclusive | Top-40 exclusive coverage |
|---|---:|---:|---:|
| Negative | 3,491 | 3,488 | 0.020 |
| Notr | 5,490 | 5,490 | 0.006 |
| Positive | 16,092 | 16,089 | 0.009 |
