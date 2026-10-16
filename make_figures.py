"""Figures for the audit. Run after audit.py.

images/audit_v2_ncc_null.png      cross-patient null vs observed Kaggle->Figshare NCC
images/audit_v2_containment.png   containment heatmap (fraction of row in column)
images/audit_v2_orientation.png   one Figshare slice as h5py returns it, transposed, and its Kaggle copy
images/audit_v2_verify_sheet.png  matched pairs sampled at the weakest end (just above threshold)
                                  and the strongest non-matches (just below), for visual checking
"""

import csv
import json
import os
import random

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from audit import (CACHE_DIR, FIGSHARE_MAT_DIR, IMAGES_DIR, KAGGLE_DIR, NCC_THRESHOLD,
                   RESULTS_DIR, to_uint8)

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
NULL_C, MATCH_C, THR_C = "#9ca3af", "#2563eb", "#dc2626"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})


def kaggle_path(p):
    """The pairs CSV stores paths relative to KAGGLE_DIR; the feature cache stores absolute ones."""
    return p if os.path.isabs(p) else os.path.join(KAGGLE_DIR, *p.split("/"))


def fig_slice(fid, transpose=True):
    with h5py.File(os.path.join(FIGSHARE_MAT_DIR, f"{fid}.mat"), "r") as f:
        a = to_uint8(np.array(f["cjdata"]["image"]))
    return a.T if transpose else a


def null_plot(z, res):
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    bins = np.linspace(0.3, 1.0, 141)
    ax.hist(z["cross"], bins=bins, color=NULL_C, alpha=0.85,
            label=f"Figshare vs. other-patient Figshare (null, patient groups, n={len(z['cross'])})")
    ax.hist(z["br35h_no"], bins=bins, color="#d1d5db", alpha=0.9, histtype="step", linewidth=1.2,
            label=f"Br35H no-tumor vs. Figshare (null, n={len(z['br35h_no'])})")
    ax.hist(z["nick_best"], bins=bins, color=MATCH_C, alpha=0.55,
            label="Nickparvar v1 vs. Figshare (observed)")
    ax.axvline(NCC_THRESHOLD, color=THR_C, linewidth=1.2)
    ax.text(NCC_THRESHOLD - 0.005, ax.get_ylim()[1] * 0.92, f"threshold {NCC_THRESHOLD}",
            color=THR_C, ha="right", fontsize=8)
    ax.set_xlabel("best NCC to any Figshare slice (64×64, brain-cropped)")
    ax.set_ylabel("images")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "audit_v2_ncc_null.png"), dpi=200)
    plt.close(fig)


def containment_plot(res):
    c = res["containment"]
    names = ["Figshare"] + [n for n in c]
    rows = [n for n in c]
    M = np.full((len(rows), len(names)), np.nan)
    for i, r in enumerate(rows):
        for j, col in enumerate(names):
            if col != r and col in c[r]:
                M[i, j] = c[r][col] * 100
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(names)):
            if not np.isnan(M[i, j]):
                v = M[i, j]
                ax.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 55 else INK)
            elif names[j] == rows[i]:
                ax.text(j, i, "—", ha="center", va="center", color=MUTED)
    ax.set_xlabel("contained in (column)")
    ax.set_ylabel("% of this corpus (row)")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("% of row with a near-duplicate")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "audit_v2_containment.png"), dpi=200)
    plt.close(fig)


def load_pairs():
    with open(os.path.join(RESULTS_DIR, "overlap_audit_v2_pairs.csv"), newline="") as f:
        return list(csv.DictReader(f))


def orientation_plot(pairs):
    """A Figshare slice with one upright Kaggle copy and one left-right mirrored
    copy (the audit's 'rot270' match = MATLAB orientation flipped left-right)."""
    nick = [r for r in pairs if r["corpus"] == "Nickparvar-v1" and float(r["ncc"]) > 0.99]
    by_fid = {}
    for r in nick:
        by_fid.setdefault(r["figshare_id"], {})[r["orientation"]] = r
    fid, d = next((f, d) for f, d in by_fid.items() if "transpose" in d and "rot270" in d)
    up, mir = d["transpose"], d["rot270"]
    panels = [(fig_slice(fid, transpose=False), "Figshare as h5py reads it\n(the initial audit's input)"),
              (fig_slice(fid, transpose=True), "Figshare transposed\n(MATLAB orientation)"),
              (np.asarray(Image.open(kaggle_path(up["path"])).convert("L")), f"Nickparvar copy\nNCC {float(up['ncc']):.3f}"),
              (np.asarray(Image.open(kaggle_path(mir["path"])).convert("L")), f"Nickparvar copy, mirrored\nNCC {float(mir['ncc']):.3f}")]
    p = up
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.2))
    for ax, (img, t) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "audit_v2_orientation.png"), dpi=200)
    plt.close(fig)
    return p


def strongest_rejected(k=3):
    """Merged-dataset (v1) images whose best Figshare NCC is highest while still
    *below* the threshold: the hardest negatives the threshold turns away.
    Recomputed from the feature cache (the pairs CSV holds accepted matches only)."""
    import pickle
    from audit import DIHEDRAL, FIGSHARE_LABELS, best_match
    with open(os.path.join(CACHE_DIR, "features.pkl"), "rb") as f:
        fig, corpora = pickle.load(f)
    c = corpora["Nickparvar-v1"]
    keys = list(DIHEDRAL)
    res = [best_match(c["vecs"], fig["vecs"][o]) for o in keys]
    S = np.stack([r[0] for r in res])
    which = S.argmax(0)
    best = S.max(0)
    order = [i for i in np.argsort(-best) if best[i] < NCC_THRESHOLD]
    out, seen = [], set()
    for i in order:
        j = res[which[i]][1][i]
        if j in seen:
            continue
        seen.add(j)
        out.append({"path": c["paths"][i], "kaggle_label": c["labels"][i], "ncc": f"{best[i]:.4f}",
                    "figshare_id": fig["ids"][j], "figshare_label": fig["labels"][j],
                    "orientation": keys[which[i]]})
        if len(out) == k:
            break
    return out


def fig_slice_oriented(fid, orientation):
    """Figshare slice in the orientation its Kaggle copy was matched in, so a
    mirrored copy is shown next to a mirrored Figshare slice."""
    from audit import DIHEDRAL
    return np.ascontiguousarray(DIHEDRAL[orientation](fig_slice(fid, transpose=False)))


def verify_sheet(pairs, n=6):
    """Two groups of n pairs, Kaggle image above its Figshare match.
    Top: the n weakest accepted matches (lowest NCC above threshold, one per
    Figshare slice). Bottom: every cross-class match, then the strongest
    rejected pairs (highest NCC below threshold) to fill the row."""
    acc = sorted((r for r in pairs if r["corpus"] == "Nickparvar-v1"), key=lambda r: float(r["ncc"]))
    weakest, seen = [], set()
    for r in acc:
        if r["figshare_id"] not in seen:
            seen.add(r["figshare_id"])
            weakest.append(r)
        if len(weakest) == n:
            break
    cross = [r for r in acc if r["kaggle_label"] != r["figshare_label"]][:n]
    rejected = strongest_rejected(n - len(cross))
    bottom = [(r, "mislabeled") for r in cross] + [(r, "rejected") for r in rejected]
    fig, axes = plt.subplots(4, n, figsize=(1.15 * n, 5.3))
    groups = ((0, [(r, "accepted") for r in weakest]), (2, bottom))
    tag_color = {"accepted": MATCH_C, "mislabeled": "#b45309", "rejected": THR_C}
    for row, group in groups:
        for j, (r, tag) in enumerate(group):
            axes[row, j].imshow(np.asarray(Image.open(kaggle_path(r["path"])).convert("L")), cmap="gray")
            axes[row, j].set_title(f"{tag}\nKaggle {r['kaggle_label'][:4]}, NCC {float(r['ncc']):.3f}",
                                   fontsize=6, color=tag_color[tag])
            axes[row + 1, j].imshow(fig_slice_oriented(r["figshare_id"], r["orientation"]), cmap="gray")
            axes[row + 1, j].set_title(f"Figshare #{r['figshare_id']} {r['figshare_label'][:4]}", fontsize=6)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(h_pad=0.6)
    fig.savefig(os.path.join(IMAGES_DIR, "audit_v2_verify_sheet.png"), dpi=200)
    plt.close(fig)
    return weakest, [r for r, _ in bottom]


if __name__ == "__main__":
    res = json.load(open(os.path.join(RESULTS_DIR, "overlap_audit_v2.json")))
    z = np.load(os.path.join(CACHE_DIR, "null.npz"))
    null_plot(z, res)
    containment_plot(res)
    pairs = load_pairs()
    orientation_plot(pairs)
    w, o = verify_sheet(pairs)
    for r in w + o:
        print(r["kaggle_label"], r["figshare_label"], r["ncc"], os.path.basename(r["path"]), r["figshare_id"])
