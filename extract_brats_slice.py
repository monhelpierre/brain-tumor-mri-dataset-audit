"""Extract a single representative 2D slice from a local BraTS NIfTI
volume, for the dataset example gallery (paper Figure 1).

BraTS data itself is not redistributed by this script and is not fetched
automatically: it requires a free Synapse account and accepting the
challenge's data-use terms (https://www.synapse.org/brats), a consent
step this repo does not automate. Point
BRATS_CASE_DIR at wherever you downloaded one case's NIfTI files to
(e.g. the BraTS2023-GLI challenge validation set) and rerun.
"""
import os

import nibabel as nib
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(SCRIPT_DIR, "results", "sample_dl", "brats.jpg")

# Local path to one downloaded BraTS case folder (contains *-t1c.nii.gz,
# *-t1n.nii.gz, *-t2f.nii.gz, *-t2w.nii.gz). Set the BRATS_CASE_DIR
# environment variable or edit the default below. The paper used case
# BraTS-GLI-00015-001 of the BraTS2023-GLI challenge validation set.
BRATS_CASE_DIR = os.environ.get(
    "BRATS_CASE_DIR",
    os.path.join("path", "to", "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData", "BraTS-GLI-00015-001"))
MODALITY_SUFFIX = "-t1c.nii.gz"  # T1 contrast-enhanced, for comparability with the other gallery entries

# Fraction through the z-axis to slice at; 0.55 was chosen by eye after
# scanning several candidate cases for one with a clearly visible
# ring-enhancing lesion at this level.
Z_FRACTION = 0.55


def main():
    case_name = os.path.basename(BRATS_CASE_DIR.rstrip("\\/"))
    nii_path = os.path.join(BRATS_CASE_DIR, f"{case_name}{MODALITY_SUFFIX}")
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(
            f"{nii_path} not found -- set BRATS_CASE_DIR to a real downloaded case folder."
        )

    img = nib.load(nii_path)
    data = img.get_fdata()
    z = int(data.shape[2] * Z_FRACTION)
    sl = np.rot90(data[:, :, z])
    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8) * 255
    sl = sl.astype(np.uint8)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    Image.fromarray(sl).save(OUT_PATH, quality=95)
    print(f"Saved {OUT_PATH} from {nii_path} (slice {z}/{data.shape[2]})")


if __name__ == "__main__":
    main()
