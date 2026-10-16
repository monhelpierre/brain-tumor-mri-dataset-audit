"""Build the one-example-per-dataset gallery (paper Figure 1).

Uses Figshare straight from the raw .mat files and the Kaggle collections
downloaded by audit.py (run it first), plus three single-file downloads via
the Kaggle CLI (SARTAJ, Br35H, BRISC), fetched into results/sample_dl/ if
missing. That step needs a Kaggle API token (~/.kaggle/kaggle.json); pass
--skip-download to reuse files already there.

BraTS (brats.jpg) and MU-Glioma-Post (tcia_mu_glioma_post.jpg) are not
downloaded here. BraTS requires a Synapse account and accepting the
challenge's data use terms, and MU-Glioma-Post (CC BY 4.0, no account) is
distributed only through IBM Aspera Connect. Download one case of each and
run extract_brats_slice.py / extract_mu_glioma_post_slice.py. Missing
samples are drawn as a hatched placeholder instead of failing. REMBRANDT
requires a signed TCIA Restricted License Agreement and is not shown."""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyBboxPatch
from PIL import Image


def draw_lock(ax, cx=0.5, cy=0.62, scale=0.09, color="#777777"):
    """Draw a simple padlock glyph from primitives (no font glyph needed,
    so it renders identically regardless of the system's installed fonts)."""
    body = FancyBboxPatch((cx - scale, cy - scale * 0.9), 2 * scale, 1.5 * scale,
                           boxstyle="round,pad=0,rounding_size=0.02",
                           transform=ax.transAxes, facecolor=color, edgecolor="none", zorder=3)
    shackle = Arc((cx, cy + scale * 0.6), 1.3 * scale, 1.6 * scale, theta1=0, theta2=180,
                  transform=ax.transAxes, color=color, linewidth=2.2, zorder=2)
    ax.add_patch(body)
    ax.add_patch(shackle)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
SAMPLE_DL = os.path.join(SCRIPT_DIR, "results", "sample_dl")  # single-file downloads fetched for this figure only

KAGGLE_EXE = "kaggle"  # the Kaggle CLI, from `pip install kaggle`


def ensure_sample(dataset, remote_path, local_rel_path):
    """Download one file from a Kaggle dataset into SAMPLE_DL/local_rel_path
    if it isn't already there."""
    dest = os.path.join(SAMPLE_DL, local_rel_path)
    if os.path.isfile(dest):
        return dest
    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading {dataset}:{remote_path} ...")
    subprocess.run(
        [KAGGLE_EXE, "datasets", "download", dataset, "-f", remote_path, "-p", dest_dir, "--force"],
        check=True,
    )
    downloaded = os.path.join(dest_dir, os.path.basename(remote_path))
    if downloaded != dest:
        os.replace(downloaded, dest)
    return dest


SAMPLE_FILES = {
    "sartaj_glioma": ("sartajbhuvaji/brain-tumor-classification-mri", "Testing/glioma_tumor/image(1).jpg", "sartaj_glioma.jpg"),
    "br35h": ("ahmedhamada0/brain-tumor-detection", "Br35H-Mask-RCNN/TEST/y701.jpg", "br35h.jpg"),
    "brisc": ("rezwanrudro/brisc2025", "brisc2025/classification_task/test/glioma/brisc2025_test_00001_gl_ax_t1.jpg", "brisc.jpg"),
}
if "--skip-download" not in sys.argv:
    for key, (ds, remote, local) in SAMPLE_FILES.items():
        try:
            ensure_sample(ds, remote, local)
        except Exception as e:
            print(f"  could not fetch {key}: {e}")

from audit import KAGGLE_DIR  # noqa: E402  (the kagglehub layout audit.py downloads into)

MASOUD = os.path.join(KAGGLE_DIR, "masoudnickparvar", "brain-tumor-mri-dataset", "versions", "1")
JAKESH = os.path.join(KAGGLE_DIR, "jakeshbohaju", "brain-tumor", "versions", "3", "Brain Tumor", "Brain Tumor")

def first_real_image(dir_path):
    """First non-mask image file in dir_path, sorted for determinism."""
    for fname in sorted(os.listdir(dir_path)):
        if "mask" not in fname.lower() and fname.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(dir_path, fname)
    raise FileNotFoundError(f"No non-mask image found in {dir_path}")


def figshare_glioma_png():
    """First glioma slice (label 2) straight from the raw .mat files, in MATLAB
    orientation (h5py returns MATLAB's column-major cjdata.image transposed)."""
    import h5py
    import numpy as np
    from audit import FIGSHARE_MAT_DIR, to_uint8
    out = os.path.join(SAMPLE_DL, "figshare_glioma.png")
    if not os.path.isfile(out):
        mats = sorted(os.listdir(FIGSHARE_MAT_DIR), key=lambda f: int(os.path.splitext(f)[0]))
        for f in mats:
            with h5py.File(os.path.join(FIGSHARE_MAT_DIR, f), "r") as h:
                if int(np.array(h["cjdata"]["label"]).ravel()[0]) == 2:
                    Image.fromarray(to_uint8(np.array(h["cjdata"]["image"])).T).save(out)
                    break
    return out


entries = [
    ("Figshare", figshare_glioma_png(), "glioma"),
    ("SARTAJ", os.path.join(SAMPLE_DL, "sartaj_glioma.jpg"), "glioma_tumor"),
    ("Br35H", os.path.join(SAMPLE_DL, "br35h.jpg"), "tumor (annotated)"),
    ("Kaggle\nBrain Tumor MRI", first_real_image(os.path.join(MASOUD, "Training", "glioma")), "glioma"),
    ("Kaggle\n\"Brain Tumor\"", os.path.join(JAKESH, "Image3.jpg"), "tumor=1 (CSV label)"),
    ("BraTS", os.path.join(SAMPLE_DL, "brats.jpg"), "glioma, T1c (BraTS2023-GLI val.)"),
    ("TCIA", os.path.join(SAMPLE_DL, "tcia_mu_glioma_post.jpg"), "MU-Glioma-POST, T1c\n(REMBRANDT: restricted, not shown)"),
    ("BRISC", os.path.join(SAMPLE_DL, "brisc.jpg"), "glioma, axial T1"),
]

n = len(entries)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.15 * rows + 0.3))
axes = axes.flatten()

missing = []
for i, (name, path, sublabel) in enumerate(entries):
    ax = axes[i]
    ax.set_xticks([])
    ax.set_yticks([])
    have_image = path is not None and os.path.isfile(path)
    if not have_image:
        # Deliberately NOT a plain blank box: a diagonal hatch fill, a
        # dashed (not solid) border, and a lock glyph make clear this is
        # an intentional "not fetched" marker rather than a missing/broken
        # image. Two different reasons land here: genuinely access-gated
        # (path is None), or a real, fetchable sample whose extraction
        # script just hasn't been run on this machine yet (path given but
        # missing) -- see the README for which is
        # which and why neither is auto-downloaded by this script alone.
        if path is not None:
            missing.append((name, path))
            sublabel = sublabel + "\n[not found locally --\nsee README.md]"
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#999999")
            spine.set_linestyle((0, (5, 4)))
            spine.set_linewidth(1.3)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    facecolor="none", edgecolor="#cfcfcf",
                                    hatch="////", linewidth=0))
        draw_lock(ax)
        ax.text(0.5, 0.26, sublabel, ha="center", va="center", fontsize=6.5,
                color="#444444", style="italic", transform=ax.transAxes)
    else:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#888888")
        img = Image.open(path).convert("L")
        ax.imshow(img, cmap="gray")
        ax.text(0.5, -0.08, sublabel, ha="center", va="top", fontsize=6.8, transform=ax.transAxes)
    ax.set_title(name, fontsize=8.3)

if missing:
    print("\nNot regenerated (sample file missing locally):")
    for name, path in missing:
        print(f"  {name}: expected {path}")
    print("See the README for how to obtain these.")

for j in range(n, len(axes)):
    axes[j].axis("off")

fig.suptitle("One example per cataloged dataset (Table 1)", fontsize=10, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out_path = os.path.join(IMAGES_DIR, "dataset_example_gallery.png")
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Saved {out_path}")
