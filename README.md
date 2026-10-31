# Brain tumor MRI dataset audit

Code and results for the paper *Patient-Level Leakage and Cross-Dataset Duplication in Public
Brain Tumor MRI Benchmarks: A Provenance Audit and Selection Framework*
(under review).

Most brain tumor MRI classification papers train and test on a few public
datasets: the Figshare release of Cheng et al. and Kaggle collections
built from it (Nickparvar, SARTAJ, Br35H, Navoneel, Bohaju), plus BRISC.
They have different names, so they are often treated as different data.
This repository measures how much they actually share.

## Main findings

| Finding | Number |
|---|---|
| Kaggle Brain Tumor MRI Dataset (v1) that is Figshare | 4,441 of 7,023 images (63%), all 209 patients |
| SARTAJ that is Figshare | 66% |
| BRISC that is Figshare | 71% |
| Test images with a near copy in training | v1: 38%, v2: 29% (v2 is described as deduplicated), BRISC: 13% |
| Traceable tumor test images whose patient is also in training | v1: 742/742, v2: 877/879, BRISC: 700/700 |
| Figshare patient IDs vs. actual patients | 233 IDs, 209 people |
| Corpus sharing no image with any other | Bohaju only |

**A pitfall worth knowing.** Figshare ships MATLAB v7.3 `.mat` files.
Reading `cjdata.image` with `h5py` returns the slice **transposed**,
because MATLAB stores arrays column by column. Nothing crashes, but every
comparison against an upright copy fails. An audit run on transposed input
finds 3 shared images instead of 4,441. Undo the transpose (`image.T`), and
test all 8 flips and rotations when matching, since Kaggle copies also
circulate mirrored.

## Method in short

1. Rescale each image to 0 to 255, crop to the head, resize to 64x64
   (Lanczos), subtract the mean and normalize to unit length.
2. Compare images by normalized cross correlation (NCC), the dot product
   of two such vectors. It ignores linear changes in brightness and
   contrast.
3. Search every Figshare slice in all 8 orientations and keep each image's
   best score.
4. Call two images the same slice at NCC >= 0.93. This threshold sits
   above the highest score between Figshare slices of *different*
   patients (0.929), with patient IDs that differ only by a trailing
   letter merged into one patient.
5. A Kaggle image matched to Figshare inherits that slice's patient, which
   makes patient leakage across a train/test split measurable.

All counts are lower bounds: heavily cropped, augmented or nonlinearly
edited copies are not matched.

## Repository layout

| Path | What it is |
|---|---|
| `audit.py` | The full audit. Downloads missing data, computes every number in the paper. |
| `make_figures.py` | Figures from the audit: orientation example, NCC null distributions, containment matrix, visual verification sheet. Run after `audit.py`. |
| `make_gallery.py` | One example image per dataset (paper Figure 1). |
| `extract_brats_slice.py`, `extract_mu_glioma_post_slice.py` | Turn one manually downloaded BraTS or MU-Glioma-Post case into the 2D sample used by the gallery. |
| `results/overlap_audit_v2.json` | Every number reported in the paper. |
| `results/overlap_audit_v2_pairs.csv` | Every Kaggle to Figshare match: corpus, file, split, labels, Figshare slice and patient ID, NCC, orientation. |
| `legacy/` | The initial pHash audit, kept only to reproduce the transposed input result (3 matches). Do not use it for anything else. |

## Running it

Requires Python 3 with numpy, Pillow, h5py, imagehash, kagglehub,
matplotlib and nibabel (plus the `kaggle` CLI for the gallery).

```bash
pip install -r requirements.txt
python audit.py          # about 20 min on first run, seconds afterwards (cached)
python make_figures.py
```

`audit.py` downloads whatever is missing on first run:

- **Figshare** from its public archive (article 1512427, version 5).
- **Kaggle collections** through `kagglehub` at pinned versions, so a
  later re-upload cannot silently change the result. Public datasets, no
  API key needed.

| Dataset | Kaggle handle | Version |
|---|---|---|
| Brain Tumor MRI Dataset v1 / v2 | `masoudnickparvar/brain-tumor-mri-dataset` | 1, 2 |
| SARTAJ | `sartajbhuvaji/brain-tumor-classification-mri` | 3 |
| Br35H | `ahmedhamada0/brain-tumor-detection` | 12 |
| Navoneel | `navoneel/brain-mri-images-for-brain-tumor-detection` | 1 |
| Bohaju | `jakeshbohaju/brain-tumor` | 3 |
| BRISC | `briscdataset/brisc2025` | 6 |

Data locations can be changed with the `FIGSHARE_MAT_DIR` and
`KAGGLE_ROOT` environment variables.

`make_gallery.py` also needs a Kaggle API token (`~/.kaggle/kaggle.json`)
for three single file downloads. The BraTS and MU-Glioma-Post samples
cannot be fetched automatically: BraTS requires a Synapse account and
accepting the challenge's data use terms, and MU-Glioma-Post is
distributed only through IBM Aspera. Download one case yourself and set
the path at the top of the matching `extract_*.py` script.

## Data

No dataset images are redistributed here. Each dataset keeps its own
license and terms of use. Please cite the original sources:

- Cheng, J. (2017). Brain tumor dataset. Figshare. doi:10.6084/m9.figshare.1512427
- Fateh, A. et al. (2026). BRISC: Annotated dataset for brain tumor
  segmentation and classification. *Scientific Data* 13, 361.
  doi:10.1038/s41597-026-06753-y
- The Kaggle collections listed above, at their Kaggle pages.

## Citation

Citation details will be added after review.

## License

Code: Apache License 2.0 (see `LICENSE`). Results files in `results/`:
CC BY 4.0.
