#!/bin/sh
# Reproduces every report and experiment in the article (leakaudit 0.2.0).
# Expects the corpora in data/ (see README) and leakaudit installed (pip install -e .).
set -e
# 1. the input files must be exactly the audited ones
(cd data && sha256sum -c ../paper/experiments/data_sha256.txt)
LA=leakaudit
# leakaudit exits 1 when the certificate fails (expected here); any other non-zero status is an error
audit() { set +e; "$LA" "$@" > /dev/null; rc=$?; set -e; [ "$rc" -le 1 ] || { echo "audit failed: $*"; exit "$rc"; }; }
PY=${PYTHON:-python3}
mkdir -p reports paper/experiments
audit data/turkish_phishing_dataset.csv --text "Konu+İçerik" --label Kategori --metadata Gönderen --name "TurkPhish v1" --out reports/turkphish_v1.json --md reports/turkphish_v1.md
audit data/turkphish_v2.csv --text konu+icerik --label kategori --name "TurkPhish v2" --out reports/turkphish_v2.json --md reports/turkphish_v2.md
audit data/turkphish_v2.csv --text icerik --label kategori --metadata konu --name "TurkPhish v2 (body only, subject as metadata)" --out reports/turkphish_v2_body.json --md reports/turkphish_v2_body.md
audit data/ttc4900.parquet --text text --label category --name "TTC-4900 (news categorization)" --out reports/ttc4900.json --md reports/ttc4900.md
audit data/urun_yorum.parquet --text sentence --label sentiment --name "Turkish product reviews (sentiment)" --out reports/urun_yorum.json --md reports/urun_yorum.md
audit data/duygu_test.parquet --text text --label label --metadata dataset --name "Turkish sentiment analysis dataset (test split)" --out reports/duygu.json --md reports/duygu.md
audit data/agnews_train.parquet --text text --label label --lang generic --name "AG News (training split)" --out reports/agnews.json --md reports/agnews.md
# DBpedia, Yelp polarity and IMDB: fast.ai copies (s3.amazonaws.com/fast-ai-nlp: dbpedia_csv.tgz, yelp_review_polarity_csv.tgz, imdb.tgz), converted by paper/experiments/convert_fastai.py
audit data/dbpedia_train.parquet --text text --label label --lang generic --name "DBpedia (training split)" --out reports/dbpedia.json --md reports/dbpedia.md
audit data/yelp_polarity_train.parquet --text text --label label --lang generic --name "Yelp polarity (training split)" --out reports/yelp.json --md reports/yelp.md
audit data/imdb_train.parquet --text text --label label --lang generic --name "IMDB (training split)" --out reports/imdb.json --md reports/imdb.md
$PY paper/experiments/run_experiments.py data paper/experiments/results.json
$PY paper/experiments/exp_r1.py data paper/experiments/exp_r1.json
$PY paper/experiments/exp_r1_crosscorpus.py data paper/experiments/exp_r1_crosscorpus.json
$PY paper/experiments/exp_r2.py data paper/experiments/exp_r2.json
$PY paper/experiments/exp_r3.py data paper/experiments/exp_r3.json
# Deepchecks comparison needs a separate environment (deepchecks 0.19.1, scikit-learn 1.5.2):
#   python paper/experiments/run_deepchecks.py data paper/experiments/deepchecks_r3.json
$PY paper/experiments/exp_r4_train.py data paper/experiments/exp_r4.json
# R5 (BERTurk, ~45 min per seed on 4 CPU cores; needs torch and transformers==4.46.3 and the model files in models/berturk):
# $PY paper/experiments/exp_r5_berturk.py data models/berturk paper/experiments/exp_r5.json
$PY paper/experiments/exp_r6_agnews.py data paper/experiments/exp_r6.json
$PY paper/experiments/exp_r7_sensitivity.py . paper/experiments/exp_r7.json
$PY paper/experiments/exp_r8_position_control.py data paper/experiments/exp_r8.json
$PY paper/experiments/exp_r9_null_anchor.py . paper/experiments/exp_r9.json
$PY paper/experiments/exp_r10_specificity.py data paper/experiments/exp_r10.json
$PY paper/experiments/exp_r12_inflation_en.py data paper/experiments/exp_r12_imdb.json IMDB
$PY paper/experiments/exp_r12_inflation_en.py data paper/experiments/exp_r12_ag.json "AG News"
# R13 (BERTurk as layer 2 reference for the TTC-4900 opening check, ~50 min per seed on CPU; same environment as R5):
# $PY paper/experiments/exp_r13_berturk_views.py data models/berturk paper/experiments/exp_r13.json 42,43,44
# R11 (Cleanlab Datalab; run in a separate environment with cleanlab[datalab]==2.9.0):
# python paper/experiments/exp_r11_cleanlab.py data paper/experiments/exp_r11.json
$PY paper/experiments/make_figures.py .
$PY paper/make_tables.py .
$PY -c "import sys, platform, sklearn, numpy, pandas, scipy; print(platform.platform(), platform.processor(), sys.version.split()[0], 'sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'scipy', scipy.__version__)" > paper/experiments/environment.txt
sha256sum src/leakaudit/*.py paper/experiments/*.py requirements-paper.txt > paper/experiments/code_sha256.txt
echo DONE
