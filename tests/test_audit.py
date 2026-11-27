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


def test_clean_corpus_passes_certificate():
    """Negative control: no duplicates, no fixed openings, cues spread over the text."""
    r = audit(_diverse_corpus(), "text", "label")
    assert r["certificate"]["passed"], r["certificate"]["reasons"]


def test_identical_pairs_counted_analytically():
    df = _diverse_corpus()
    extra = pd.concat([df.iloc[[0]]] * 300, ignore_index=True)  # one cluster larger than the LSH bucket cap
    r = audit(pd.concat([df, extra], ignore_index=True), "text", "label")
    assert r["layer1"]["near_dup_pairs_identical"] == 301 * 300 // 2
    assert r["layer1"]["lsh_buckets_skipped"] == 0


def test_language_option_lowercasing():
    from leakaudit import text as T
    try:
        T.set_language("generic")
        assert T.tr_lower("ISTANBUL") == "istanbul"
        T.set_language("tr")
        assert T.tr_lower("ISTANBUL") == "ıstanbul"
        assert T.tr_lower("İSTANBUL") == "istanbul"
    finally:
        T.set_language("tr")


def test_user_split_overlap_reported():
    import pandas as pd
    from leakaudit import audit
    rows = []
    for i in range(60):
        rows.append({"text": f"belge numara {i} konu alpha beta gamma kelime{i} ek{i*7}", "label": i % 2,
                     "split": "train" if i < 48 else "test"})
    rows.append({"text": rows[50]["text"], "label": rows[50]["label"], "split": "train"})
    r = audit(pd.DataFrame(rows), text="text", label="label", split="split")
    assert r["layer3"]["user_split_test_overlap"] > 0
    assert any("given test split" in w for w in r["certificate"]["warnings"])


def test_content_aware_certificate_is_reported():
    import pandas as pd
    from leakaudit import audit
    rows = [{"text": f"metin {i} kelime{i} başka{i*3} sözcük{i*5} ve devamı", "label": i % 2} for i in range(80)]
    r = audit(pd.DataFrame(rows), text="text", label="label")
    ca = r["certificate_content_aware"]
    assert set(ca["checks"]) == set(r["certificate"]["checks"])
    assert ca["checks"]["L1_near_duplicates"]


def test_matched_windows_null_and_group_bootstrap():
    import numpy as np
    from leakaudit.layers import _boot_index
    rep = audit(_diverse_corpus(), "text", "label")
    views = rep["layer2"]["views"]
    for k in (3, 5, 10):
        assert f"random_window_{k}_tokens" in views
    chance = rep["layer2"]["chance_macro_f1"]
    for v in views.values():  # permuted labels carry no signal: the null never clearly exceeds 1/C
        assert v["permuted_label_macro_f1"] <= chance + 0.05
    # a group bootstrap draws whole groups: members of a group appear equally often
    groups = np.array([0, 0, 0, 1, 2, 3])
    b = _boot_index(np.random.RandomState(0), len(groups), groups)
    counts = np.bincount(b, minlength=6)
    assert counts[0] == counts[1] == counts[2]
