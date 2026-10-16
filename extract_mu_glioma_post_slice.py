"""Extract a single representative 2D slice from a local MU-Glioma-Post
NIfTI case, for the dataset-example gallery (paper Figure 1).

Unlike BraTS, MU-Glioma-Post is CC BY 4.0 and openly downloadable with no
account (verified directly on its TCIA collection page); it is not
fetched automatically here only because distribution goes through the
IBM Aspera Connect client. Point
MU_GLIOMA_POST_ROOT at wherever you downloaded the collection to (the
TCIA page's "Images (skull-stripped) and Segmentations" row) and rerun.

The tumor mask shipped with each case is used to pick the axial slice
with the largest tumor cross-section, rather than guessing a fixed
z-fraction as the BraTS extraction script does (that collection's
validation release ships no mask).
"""
import os

import nibabel as nib
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(SCRIPT_DIR, "results", "sample_dl", "tcia_mu_glioma_post.jpg")

# Set the MU_GLIOMA_POST_ROOT environment variable or edit the default below.
MU_GLIOMA_POST_ROOT = os.environ.get(
    "MU_GLIOMA_POST_ROOT", os.path.join("path", "to", "PKG - MU-Glioma-Post", "MU-Glioma-Post"))
PATIENT_ID = "PatientID_0009"
TIMEPOINT = "Timepoint_2"
MODALITY_SUFFIX = "_brain_t1c.nii.gz"  # T1 contrast-enhanced, for comparability with the other gallery entries


def main():
    case_dir = os.path.join(MU_GLIOMA_POST_ROOT, PATIENT_ID, TIMEPOINT)
    prefix = f"{PATIENT_ID}_{TIMEPOINT}"
    t1c_path = os.path.join(case_dir, f"{prefix}{MODALITY_SUFFIX}")
    mask_path = os.path.join(case_dir, f"{prefix}_tumorMask.nii.gz")
    if not (os.path.isfile(t1c_path) and os.path.isfile(mask_path)):
        raise FileNotFoundError(
            f"{t1c_path} or its tumorMask not found -- set MU_GLIOMA_POST_ROOT/"
            "PATIENT_ID/TIMEPOINT to a real downloaded case."
        )

    img = nib.load(t1c_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    areas = (mask > 0).sum(axis=(0, 1))
    z = int(np.argmax(areas))
    sl = np.rot90(img[:, :, z])
    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8) * 255
    sl = sl.astype(np.uint8)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    Image.fromarray(sl).save(OUT_PATH, quality=95)
    print(f"Saved {OUT_PATH} from {t1c_path} (slice {z}/{img.shape[2]}, tumor area {int(areas[z])} px)")


if __name__ == "__main__":
    main()
