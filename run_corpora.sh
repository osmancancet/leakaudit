#!/bin/sh
set -e
LA=.venv/bin/leakaudit
$LA data/ttc4900.parquet --text text --label category --name "TTC-4900 (news categorization)" --out reports/ttc4900.json --md reports/ttc4900.md > /dev/null || true
$LA data/urun_yorum.parquet --text sentence --label sentiment --name "Turkish product reviews (sentiment)" --out reports/urun_yorum.json --md reports/urun_yorum.md > /dev/null || true
$LA data/duygu_test.parquet --text text --label label --metadata dataset --name "Turkish sentiment analysis dataset (test split)" --out reports/duygu.json --md reports/duygu.md > /dev/null || true
echo DONE
