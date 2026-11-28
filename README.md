# LeakAudit

**Leakage and shortcut audit for labelled text corpora.** LeakAudit tells you, before you train anything, whether a text classification dataset can be "solved" by duplicates, templates, surface cues or a lookup table instead of by learning the task. It ends with a pass/fail **leakage certificate** you can publish alongside the dataset.

It grew out of the TurkPhish v2 study, where a widely used Turkish phishing dataset turned out to collapse to 7 templates and to be perfectly separable by its first three words ([Çetlenbik & Süzen, 2026](https://doi.org/10.3390/app16168274)).

```bash
pip install leakaudit
leakaudit data.csv --text text --label label --md report.md
```

## What it checks

| Layer | Check | Fails when |
|---|---|---|
| 1. Documents | Template uniqueness after masking URLs, e-mails, phones and numbers | fewer than 95 % of documents are distinct templates |
| | MinHash-LSH near duplicates, verified with exact Jaccard | any pair has Jaccard ≥ 0.8 |
| | Exact duplicates carrying different labels | any such group exists (label noise) |
| 2. Surface views | First 3/5/10 tokens, punctuation/emoji/case profile, length alone (deciles), every metadata column alone; a random 10-token window as an ungated control | a view recovers more than 50 % of the above-chance full-text macro-F1 |
| 3. Sentences | Learning-free lookup over class-exclusive sentences vs. a trained TF-IDF model | the lookup recovers more than 50 % of the trained model's above-chance macro-F1 |

**Normalized recovery** = (view macro-F1 − chance) / (full-text macro-F1 − chance), with chance = 1 / number of classes. A view at chance scores 0; a view as good as the full text scores 1. The value is not clipped, so it can exceed 1 or fall below 0, and it comes with a 95 % bootstrap interval. The raw ratio (view / full text) is reported as well.

Layers 2 and 3 use folds that keep duplicate groups (exact duplicates and verified near-duplicates, joined by union-find) on one side of every split, so the audit does not leak itself. When no grouped partition can place every class in every fold, plain stratified folds are used and the report says so.

All thresholds are configurable. Every number in the report is reproducible from the seed.

## Usage

```bash
# join several columns into the text
leakaudit emails.csv --text subject+body --label category --out report.json --md report.md

# test metadata columns as stand-alone predictors, use your own split for layer 3
leakaudit data.parquet --text text --label label --metadata source author --split split
```

```python
import pandas as pd
from leakaudit import audit, to_markdown

df = pd.read_csv("data.csv")
report = audit(df, text="text", label="label", metadata=["source"])
print(report["certificate"])        # {'passed': False, 'checks': {...}, 'reasons': [...]}
print(to_markdown(report))
```

The command exits with status 1 when the certificate fails, so it can gate a CI pipeline or a dataset release.

Text is normalised to Unicode NFKC and lower-cased with Turkish rules (dotted/dotless *i*) by default; `--lang generic` switches to Unicode default lower-casing. No other step depends on the language.

With `--split`, the report also gives the share of test documents that have a duplicate in the training part of your split. The certificate carries warnings when grouped folds were infeasible or when the sentence lookup falls back to the majority class for more than 90 % of test documents (the lookup check is then not informative).

## Example audits

The [`reports/`](reports) folder holds full reports for five public Turkish corpora. All five fail certification, each for a different reason:

| Corpus | Documents | Main finding |
|---|---:|---|
| [TurkPhish v1](https://www.kaggle.com/datasets/osmancancet/turkish-phishing-email-dataset) | 7,504 | 59 masked templates; first 5 tokens and the sender column each separate the classes |
| [TurkPhish v2](https://huggingface.co/datasets/OsmanCan/turkphish-v2) | 2,653 | Clean at document level; the subject line alone separates the classes |
| [TTC-4900](https://huggingface.co/datasets/savasy/ttc4900) | 4,900 | 14 % of documents sit in duplicate groups; removing training copies of test documents lowers macro-F1 |
| [Turkish product reviews](https://huggingface.co/datasets/fthbrmnby/turkish_product_reviews) | 235,144 | 77 duplicate groups carry contradictory labels |
| [Turkish sentiment (test)](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset) | 48,965 | The `dataset` provenance column alone recovers 88 % of above-chance macro-F1; half of the test split also occurs in the product-review corpus |

`pip install -r requirements-paper.txt` pins the exact versions used. `run_all.sh` first checks the input files against their SHA-256 digests, then reproduces every report, experiment, table and figure of the article from the files in `data/` (their SHA-256 digests are in `paper/experiments/data_sha256.txt`); `run_corpora.sh` re-runs only the audits. The training split of the sentiment corpus is `train.csv` of revision `4bd9103ad5f24d44db806d40c94c8b9ec116ad05`, saved as `data/duygu_train.csv`. AG News is `data/{train,test}-00000-of-00001.parquet` of `fancyzhx/ag_news` at revision `eb185aade064a813bc0b7f42de02595523103ca4`, saved as `data/agnews_train.parquet` and `data/agnews_test.parquet`. The BERTurk experiments (`exp_r5_berturk.py`, and `exp_r13_berturk_views.py` for the TTC-4900 opening check with BERTurk as reference model) use `dbmdz/bert-base-turkish-cased` at revision `b6e1de16c983e0f2c70664591ea3f22810072608` (model.safetensors SHA-256 `18e29c6c61a046ab18fa48aeec3dce1285b09c5648d9ba68da1474940b9a9fcd`) with torch 2.5.1 and transformers 4.46.3. A surface view flagged in layer 2 is a shortcut the task *can* be solved by, not necessarily an error: in news categorization the first tokens are the headline. Read layer 2 as shortcut susceptibility.

## Changes in 0.2.0

- Grouped (duplicate-aware) folds for layers 2 and 3; random-fold full-text score reported for comparison.
- Identical pairs counted exactly; hashing only for distinct texts, skipped LSH buckets reported (0.1.0 undercounted large identical clusters).
- Unclipped recovery with bootstrap intervals; random-window control; length in deciles; metadata as value plus character n-grams.
- Layer 3 gate on normalized lookup recovery; fallback rate and majority baseline reported.
- NFKC normalisation; `--lang` option.
- Overlap of a user-supplied split; certificate warnings.
- Content-aware certificate profile (rate-based; opening views judged against a random window of the same length; near-duplicates as affected documents per 1,000).
- Random-window controls for every first-k view; bootstrap intervals over duplicate groups; permuted-label null for every view.
- `near_dup_documents_affected` counts sampled documents with an identical or near-duplicate partner inside the near-duplicate sample (earlier builds of 0.2.0 counted duplicate-group members over the whole corpus and divided by the sample size).
- Paper experiments R8 (position-shuffled negative control and bootstrap interval for the opening excess) and R9 (recovery anchored on the permuted-label null).
- Paper experiments R10 (permuted-label nulls scored against the layer 2 gates, a class-exclusive 4-gram lookup, the opening excess refitted under ten seeds) and R11 (comparison with Cleanlab Datalab 2.9.0 in a TF-IDF feature space).
- Audits of three further English benchmarks (DBpedia, Yelp polarity, IMDB) and paper experiment R12 (size-matched inflation on IMDB and AG News samples).

## Citation

If you use LeakAudit, please cite:

> Çetlenbik, O. C., & Süzen, A. A. (2026). TurkPhish v2: A Template-Composed Generation Matrix for Turkish Phishing E-Mail Corpora, with a Leakage Audit and a Detection Benchmark. *Applied Sciences, 16*(16), 8274. https://doi.org/10.3390/app16168274

## Türkçe özet

LeakAudit, etiketli bir metin veri setinin görevi öğrenmeden, yalnızca kopyalar, şablonlar, yüzeysel ipuçları ya da bir arama tablosuyla "çözülüp çözülemeyeceğini" eğitimden önce ölçer ve veri setiyle birlikte yayımlanabilecek bir geçti/kaldı **sızıntı sertifikası** üretir. Türkçe büyük/küçük harf dönüşümünü (ı/i) doğru yapar; denetimlerin kendisi dilden bağımsızdır.

## License

MIT. Contact: osman.cetlenbik@cbu.edu.tr
