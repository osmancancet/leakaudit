"""Convert the fast.ai copies of DBpedia, Yelp polarity and IMDB to parquet.
Usage: python convert_fastai.py <extracted_dir> <data_dir>"""
import sys, pathlib
import pandas as pd
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
db = pd.read_csv(src / "dbpedia_csv/train.csv", header=None, names=["label", "title", "text"]); db["label"] -= 1
db[["text", "label", "title"]].to_parquet(out / "dbpedia_train.parquet")
y = pd.read_csv(src / "yelp_review_polarity_csv/train.csv", header=None, names=["label", "text"]); y["label"] -= 1
y.to_parquet(out / "yelp_polarity_train.parquet")
rows = [(p.read_text(encoding="utf-8"), c) for lab, c in (("neg", 0), ("pos", 1))
        for p in sorted((src / "imdb/train" / lab).glob("*.txt"))]
pd.DataFrame(rows, columns=["text", "label"]).to_parquet(out / "imdb_train.parquet")
