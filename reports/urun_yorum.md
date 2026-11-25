# LeakAudit report: Turkish product reviews (sentiment)

**Certificate: FAILED**  
235,144 documents, 2 classes; layers 2-3 on 20,000 documents; seed 42; leakaudit 0.1.0.

Reasons:

- 571 near-duplicate pairs (Jaccard >= 0.8)
- 80 duplicate groups carry conflicting labels
- view 'first_10_tokens' recovers 64% of above-chance full-text performance

## Layer 1: documents

| Check | Value | Threshold | Result |
|---|---:|---:|---|
| Template uniqueness after masking | 0.9926 | >= 0.95 | PASS |
| Near-duplicate pairs (Jaccard >= 0.8) | 571 | <= 0 | FAIL |
| Duplicate groups with conflicting labels | 80 | <= 0 | FAIL |

Distinct masked templates: 233,393; duplicate groups: 1,112 covering 2,863 documents; near-duplicate pairs crossing labels: 22 (sample of 50,000).

## Layer 2: surface views

Full-text macro-F1 (5-fold CV): 0.7413; chance: 0.5000. Normalized recovery = (view - chance) / (full text - chance).

| View | Macro-F1 | Recovery ratio | Normalized recovery |
|---|---:|---:|---:|
| first_3_tokens | 0.5668 | 0.765 | 0.277 |
| first_5_tokens | 0.5943 | 0.802 | 0.391 |
| first_10_tokens | 0.6540 | 0.882 | 0.638 |
| punctuation_emoji_case_profile | 0.1586 | 0.214 | 0.000 |
| length_only | 0.3488 | 0.470 | 0.000 |

Max normalized recovery: 0.638 (first_10_tokens); threshold <= 0.5: FAIL.
Length distribution by class: KS D = 0.0226, p = 0.5778.

## Layer 3: sentences

Zero-learning lookup over class-exclusive sentences: **0.4837** macro-F1; trained full-text reference: 0.7357; margin 0.2520 (>= 0.1 required): PASS.

| Class | Sentence types | Class-exclusive | Top-40 exclusive coverage |
|---|---:|---:|---:|
| 0 | 1,813 | 1,808 | 0.032 |
| 1 | 27,862 | 27,857 | 0.009 |
