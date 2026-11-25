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
| 2. Surface views | First 3/5/10 tokens, punctuation/emoji/case profile, length alone, every metadata column alone | a view recovers more than 50 % of the above-chance full-text macro-F1 |
| 3. Sentences | Zero-learning lookup over class-exclusive sentences vs. a trained TF-IDF model | the trained model beats the lookup by less than 0.10 macro-F1 |

**Normalized recovery** = (view macro-F1 − chance) / (full-text macro-F1 − chance), with chance = 1 / number of classes. A view at chance scores 0; a view as good as the full text scores 1. The raw ratio (view / full text) is reported as well.

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

Text normalisation is Turkish-aware (dotted/dotless *i*), but every check is language-independent.

## Example audits

The [`reports/`](reports) folder holds full reports for five public Turkish corpora. All five fail certification, each for a different reason:

| Corpus | Documents | Main finding |
|---|---:|---|
| [TurkPhish v1](https://www.kaggle.com/datasets/osmancancet/turkish-phishing-email-dataset) | 7,504 | 59 masked templates; first 5 tokens and the sender column each separate the classes |
| [TurkPhish v2](https://huggingface.co/datasets/OsmanCan/turkphish-v2) | 2,653 | Clean at document level; the subject line alone separates the classes |
| [TTC-4900](https://huggingface.co/datasets/savasy/ttc4900) | 4,900 | Template uniqueness 0.925; 717 near-duplicate pairs |
| [Turkish product reviews](https://huggingface.co/datasets/fthbrmnby/turkish_product_reviews) | 235,144 | 80 duplicate groups carry contradictory labels |
| [Turkish sentiment (test)](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset) | 48,965 | The `dataset` provenance column alone recovers 93 % of full-text macro-F1 |

`run_corpora.sh` reproduces the last three after downloading the Parquet files into `data/`. A surface view flagged in layer 2 is a shortcut the task *can* be solved by, not necessarily an error: in news categorization the first tokens are the headline. Read layer 2 as shortcut susceptibility.

## Citation

If you use LeakAudit, please cite:

> Çetlenbik, O. C., & Süzen, A. A. (2026). TurkPhish v2: A Template-Composed Generation Matrix for Turkish Phishing E-Mail Corpora, with a Leakage Audit and a Detection Benchmark. *Applied Sciences, 16*(16), 8274. https://doi.org/10.3390/app16168274

## Türkçe özet

LeakAudit, etiketli bir metin veri setinin görevi öğrenmeden, yalnızca kopyalar, şablonlar, yüzeysel ipuçları ya da bir arama tablosuyla "çözülüp çözülemeyeceğini" eğitimden önce ölçer ve veri setiyle birlikte yayımlanabilecek bir geçti/kaldı **sızıntı sertifikası** üretir. Türkçe büyük/küçük harf dönüşümünü (ı/i) doğru yapar; denetimlerin kendisi dilden bağımsızdır.

## License

MIT. Contact: osman.cetlenbik@cbu.edu.tr
