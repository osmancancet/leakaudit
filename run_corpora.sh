#!/bin/sh
set -e
LA=${LA:-leakaudit}
audit() { set +e; "$LA" "$@" > /dev/null; rc=$?; set -e; [ "$rc" -le 1 ] || { echo "audit failed: $*"; exit "$rc"; }; }
audit data/ttc4900.parquet --text text --label category --name "TTC-4900 (news categorization)" --out reports/ttc4900.json --md reports/ttc4900.md
audit data/urun_yorum.parquet --text sentence --label sentiment --name "Turkish product reviews (sentiment)" --out reports/urun_yorum.json --md reports/urun_yorum.md
audit data/duygu_test.parquet --text text --label label --metadata dataset --name "Turkish sentiment analysis dataset (test split)" --out reports/duygu.json --md reports/duygu.md
echo DONE
