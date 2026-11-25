import random

import pandas as pd

from leakaudit import audit
from leakaudit.text import mask, tr_lower


def test_turkish_lower_and_mask():
    assert tr_lower("IŞIK İzmir") == "ışık izmir"
    m = mask("Hesabınız için https://x.co/a adresine gidin, 0532 123 45 67 arayın: 250 TL")
    assert "[url]" in m and "[phone]" in m and "[num]" in m


def _templated_corpus(n=400, seed=0):
    """Pathological corpus: one template per class, only numbers/links vary."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        if i % 2:
            t = f"Hesabınız askıya alındı. Doğrulamak için https://x{rng.randint(0, 9999)}.com adresine gidin. Kod {rng.randint(0, 99999)}."
            rows.append((t, "phish"))
        else:
            t = f"Toplantı yarın saat {rng.randint(8, 18)}:00 da başlayacak. Gündem ekte yer almaktadır. Ref {rng.randint(0, 99999)}."
            rows.append((t, "legit"))
    return pd.DataFrame(rows, columns=["text", "label"])


def _diverse_corpus(n=600, seed=1):
    """Label depends on content words placed at random positions; no fixed openings."""
    rng = random.Random(seed)
    vocab = [f"kelime{i}" for i in range(400)]
    cues = {"phish": [f"tehdit{i}" for i in range(60)], "legit": [f"olagan{i}" for i in range(60)]}
    rows = []
    for i in range(n):
        lab = "phish" if i % 2 else "legit"
        words = rng.sample(vocab, rng.randint(25, 45)) + rng.sample(cues[lab], 3)
        rng.shuffle(words)
        rows.append((" ".join(words) + ".", lab))
    return pd.DataFrame(rows, columns=["text", "label"])


def test_templated_corpus_fails():
    r = audit(_templated_corpus(), "text", "label")
    c = r["certificate"]
    assert not c["passed"]
    assert r["layer1"]["distinct_masked_templates"] == 2
    assert not c["checks"]["L1_template_uniqueness"]


def test_diverse_corpus_layer1_and_layer2_pass():
    r = audit(_diverse_corpus(), "text", "label")
    assert r["certificate"]["checks"]["L1_template_uniqueness"]
    assert r["certificate"]["checks"]["L1_label_conflicts"]
    assert r["layer2"]["views"]["first_3_tokens"]["normalized_recovery"] < 0.5


def test_metadata_shortcut_detected():
    df = _diverse_corpus()
    df["source"] = df["label"].map({"phish": "siteA", "legit": "siteB"})
    r = audit(df, "text", "label", metadata=["source"])
    assert r["layer2"]["max_view"] == "metadata:source"
    assert not r["certificate"]["checks"]["L2_surface_views"]


def test_label_conflict_detected():
    df = _diverse_corpus()
    df = pd.concat([df, pd.DataFrame([[df.text[0], "phish" if df.label[0] == "legit" else "legit"]],
                                     columns=["text", "label"])], ignore_index=True)
    r = audit(df, "text", "label")
    assert r["layer1"]["label_conflicting_duplicate_groups"] == 1
    assert not r["certificate"]["checks"]["L1_label_conflicts"]
