"""Figure 1 of the LeakAudit article (PDF and 1200 dpi TIFF, Times New Roman, 2-D only)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(sys.argv[1])
REP = ROOT / "reports"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"], "font.size": 9,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6})

corpora = [("turkphish_v1", "TurkPhish v1"), ("turkphish_v2", "TurkPhish v2"), ("ttc4900", "TTC-4900"),
           ("urun_yorum", "Product\nreviews"), ("duygu", "Sentiment\n(test)")]
views = [("first_3_tokens", "First 3 tokens"), ("first_5_tokens", "First 5 tokens"),
         ("first_10_tokens", "First 10 tokens"), ("random_window_10_tokens", "Random 10-token window"),
         ("punctuation_emoji_case_profile", "Punct./emoji/case profile"),
         ("length_only", "Length only"), ("metadata", "Metadata column")]

M = np.full((len(views), len(corpora)), np.nan)
for j, (f, _) in enumerate(corpora):
    r = json.load(open(REP / f"{f}.json"))["layer2"]["views"]
    for i, (k, _) in enumerate(views):
        if k == "metadata":
            md = [v["normalized_recovery"] for kk, v in r.items() if kk.startswith("metadata:")]
            M[i, j] = max(md) if md else np.nan
        elif k in r:
            M[i, j] = r[k]["normalized_recovery"]
# colours saturate at 0 and 1; the printed numbers are the unclipped values

fig, ax = plt.subplots(figsize=(6.3, 3.2))
cmap = plt.get_cmap("Greys").copy()
cmap.set_bad("white")
im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(corpora)))
ax.set_xticklabels([c for _, c in corpora])
ax.set_yticks(range(len(views)))
ax.set_yticklabels([v for _, v in views])
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isnan(M[i, j]):
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="0.45")
        else:
            ax.text(j, i, f"{round(M[i, j], 2) + 0.0:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.55 else "black", fontweight="bold" if M[i, j] > 0.5 else "normal")
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Normalized recovery")
cb.ax.axhline(0.5, color="white", lw=2.6)
cb.ax.axhline(0.5, color="black", lw=1.2)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.set_ticklabels(["0", "0.25", "0.5 (gate)", "0.75", "1"])
fig.tight_layout()
fig.savefig(OUT / "figure1_recovery.pdf")

fig.savefig(OUT / "figure1_recovery.tiff", dpi=1200, pil_kwargs={"compression": "tiff_lzw"})
print("saved figures to", OUT)
