"""Provenance and leakage audit of public brain tumor MRI datasets.

Reads the raw Figshare .mat files, tests all 8 dihedral orientations of every
Figshare slice (h5py returns MATLAB's column-major cjdata.image transposed, and
Kaggle copies also circulate mirrored), and matches by normalized
cross-correlation (NCC) on brain-cropped 64x64 thumbnails. It also reruns the
earlier pHash method on transposed and corrected input, to show how the
orientation pitfall hides almost every match ("phash_v1_method" in the JSON).

Outputs (results/):
  overlap_audit_v2.json       every number reported in the paper
  overlap_audit_v2_pairs.csv  every Kaggle->Figshare match with NCC, PID, labels
                              (paths relative to KAGGLE_DIR)
Figures: run make_figures.py afterwards.

Dataset locations: data/figshare_brain_tumor_mat and data/kaggle (kagglehub's
KAGGLEHUB_CACHE layout) next to this script. Override with the DATA_DIR,
FIGSHARE_MAT_DIR or KAGGLE_ROOT environment variables. Anything missing is
downloaded on first run (ensure_datasets): Kaggle collections via kagglehub at
the pinned versions below (public datasets, no API key needed), Figshare from
its public article archive.
"""

import csv
import glob
import hashlib
import io
import json
import os
import pickle
import re
import sys
import urllib.request
import zipfile
from collections import Counter

import h5py
import imagehash
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(SCRIPT_DIR, "data"))
FIGSHARE_MAT_DIR = os.environ.get(
    "FIGSHARE_MAT_DIR", os.path.join(DATA_DIR, "figshare_brain_tumor_mat"))
KAGGLE_ROOT = os.environ.get("KAGGLE_ROOT", os.path.join(DATA_DIR, "kaggle"))
KAGGLE_DIR = os.path.join(KAGGLE_ROOT, "datasets")
# Cheng's Figshare release (article 1512427, v5): a zip of 4 zips of .mat files.
FIGSHARE_URL = "https://figshare.com/ndownloader/articles/1512427/versions/5"
# Pinned so the audit is reproducible even if uploaders publish new versions
# (Nickparvar v2, a re-balanced/de-duplicated release, appeared after v1 was
# the version used throughout the literature; both are audited).
KAGGLE_PINNED = [
    "masoudnickparvar/brain-tumor-mri-dataset/versions/1",
    "masoudnickparvar/brain-tumor-mri-dataset/versions/2",
    "sartajbhuvaji/brain-tumor-classification-mri/versions/3",
    "ahmedhamada0/brain-tumor-detection/versions/12",
    "navoneel/brain-mri-images-for-brain-tumor-detection/versions/1",
    "jakeshbohaju/brain-tumor/versions/3",
    "briscdataset/brisc2025/versions/6",
]
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")

THUMB = 64
NCC_THRESHOLD = 0.93  # above the cross-patient Figshare null's maximum (see figshare_null)
LOWER_BAND = 0.85     # sub-threshold band reported separately: mostly true copies, but overlaps the null
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

FIGSHARE_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}
CANON = {  # folder name -> canonical class
    "glioma": "glioma", "glioma_tumor": "glioma",
    "meningioma": "meningioma", "meningioma_tumor": "meningioma",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary",
    "notumor": "notumor", "no_tumor": "notumor", "no": "notumor",
    "yes": "tumor",
}

# The 8 dihedral transforms, as functions on a 2D array. "identity" is the
# array exactly as h5py returns it; "transpose" is MATLAB's own orientation.
DIHEDRAL = {
    "identity": lambda a: a,
    "transpose": lambda a: a.T,
    "rot90": lambda a: np.rot90(a, 1),
    "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3),
    "flip_lr": lambda a: a[:, ::-1],
    "flip_ud": lambda a: a[::-1, :],
    "anti_transpose": lambda a: np.rot90(a, 2).T,
}


def ensure_datasets():
    """Download whatever is missing; a no-op when everything is already present."""
    if len(glob.glob(os.path.join(FIGSHARE_MAT_DIR, "*.mat"))) < 3064:
        print(f"downloading Figshare release to {FIGSHARE_MAT_DIR}", flush=True)
        os.makedirs(FIGSHARE_MAT_DIR, exist_ok=True)
        with urllib.request.urlopen(FIGSHARE_URL) as r:
            outer = zipfile.ZipFile(io.BytesIO(r.read()))
        for name in outer.namelist():
            if not name.lower().endswith(".zip"):
                continue
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                for member in inner.namelist():
                    if member.lower().endswith(".mat"):
                        with open(os.path.join(FIGSHARE_MAT_DIR, os.path.basename(member)), "wb") as f:
                            f.write(inner.read(member))
    for handle in KAGGLE_PINNED:
        owner, slug, _, ver = handle.split("/")
        if not os.path.isdir(os.path.join(KAGGLE_DIR, owner, slug, "versions", ver)):
            print(f"downloading {handle}", flush=True)
            os.environ["KAGGLEHUB_CACHE"] = KAGGLE_ROOT
            import kagglehub
            kagglehub.dataset_download(handle)


def to_uint8(a):
    a = a.astype(np.float64)
    lo, hi = a.min(), a.max()
    return ((a - lo) / (hi - lo + 1e-12) * 255).astype(np.uint8)


def thumb_vector(gray_u8):
    """Crop to the head's bounding box, resize to THUMB x THUMB, mean-center,
    L2-normalize. The crop removes padding/border differences between exports;
    mean-centering + normalization make the dot product an NCC that is invariant
    to linear contrast/brightness changes (independent per-image rescaling)."""
    a = gray_u8
    mask = a > max(10, 0.1 * a.max())
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    if len(rows) > 8 and len(cols) > 8:
        a = a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    t = np.asarray(Image.fromarray(a).resize((THUMB, THUMB), Image.LANCZOS), dtype=np.float32).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def load_figshare():
    """Returns dict with per-slice id, label, PID, and the vector/pHash for every
    dihedral orientation."""
    files = sorted(glob.glob(os.path.join(FIGSHARE_MAT_DIR, "*.mat")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    ids, labels, pids = [], [], []
    vecs = {k: [] for k in DIHEDRAL}
    phashes = {"identity": [], "transpose": []}
    for i, p in enumerate(files):
        if i % 500 == 0:
            print(f"  figshare {i}/{len(files)}", flush=True)
        with h5py.File(p, "r") as f:
            cj = f["cjdata"]
            img = np.array(cj["image"])
            label = int(np.array(cj["label"]).ravel()[0])
            pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).ravel())
        u8 = to_uint8(img)
        ids.append(os.path.splitext(os.path.basename(p))[0])
        labels.append(FIGSHARE_LABELS[label])
        pids.append(pid)
        for k, fn in DIHEDRAL.items():
            vecs[k].append(thumb_vector(np.ascontiguousarray(fn(u8))))
        for k in phashes:
            phashes[k].append(imagehash.phash(Image.fromarray(np.ascontiguousarray(DIHEDRAL[k](u8)))))
    return {
        "ids": ids, "labels": labels, "pids": pids,
        "vecs": {k: np.stack(v) for k, v in vecs.items()},
        "phash": phashes,
    }


def corpus_specs():
    k = KAGGLE_DIR
    nick = os.path.join(k, "masoudnickparvar", "brain-tumor-mri-dataset", "versions")
    return {
        # name: (root, layout) layout: "split/class" or "class" or "bohaju"
        "Nickparvar-v1": (os.path.join(nick, "1"), "split/class"),
        "Nickparvar-v2": (os.path.join(nick, "2"), "split/class"),
        "SARTAJ": (os.path.join(k, "sartajbhuvaji", "brain-tumor-classification-mri", "versions", "3"), "split/class"),
        "Br35H": (os.path.join(k, "ahmedhamada0", "brain-tumor-detection", "versions", "12"), "class:yes,no"),
        "Navoneel": (os.path.join(k, "navoneel", "brain-mri-images-for-brain-tumor-detection", "versions", "1", "brain_tumor_dataset"), "class:yes,no"),
        "Bohaju": (os.path.join(k, "jakeshbohaju", "brain-tumor", "versions", "3"), "bohaju"),
        "BRISC": (os.path.join(k, "briscdataset", "brisc2025", "versions", "6", "brisc2025", "classification_task"), "split/class"),
    }


def list_corpus(root, layout):
    """Returns list of (path, split, class)."""
    items = []
    if layout == "split/class":
        for split in sorted(os.listdir(root)):
            sd = os.path.join(root, split)
            if not os.path.isdir(sd):
                continue
            for cls in sorted(os.listdir(sd)):
                cd = os.path.join(sd, cls)
                if not os.path.isdir(cd):
                    continue
                for fn in sorted(os.listdir(cd)):
                    if fn.lower().endswith(IMG_EXT):
                        items.append((os.path.join(cd, fn), "test" if split.lower().startswith("test") else "train", CANON[cls.lower()]))
    elif layout.startswith("class:"):
        for cls in layout.split(":", 1)[1].split(","):
            cd = os.path.join(root, cls)
            for fn in sorted(os.listdir(cd)):
                if fn.lower().endswith(IMG_EXT):
                    items.append((os.path.join(cd, fn), "all", CANON[cls]))
    elif layout == "bohaju":
        with open(os.path.join(root, "Brain Tumor.csv"), newline="") as f:
            lab = {r["Image"]: ("tumor" if r["Class"] == "1" else "notumor") for r in csv.DictReader(f)}
        img_dir = os.path.join(root, "Brain Tumor", "Brain Tumor")
        for fn in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(fn)[0]
            if fn.lower().endswith(IMG_EXT) and stem in lab:
                items.append((os.path.join(img_dir, fn), "all", lab[stem]))
    return items


def load_corpus(name, root, layout):
    items = list_corpus(root, layout)
    vecs, md5s, ph = [], [], []
    for i, (p, _, _) in enumerate(items):
        if i % 1000 == 0:
            print(f"  {name} {i}/{len(items)}", flush=True)
        with open(p, "rb") as f:
            md5s.append(hashlib.md5(f.read()).hexdigest())
        with Image.open(p) as im:
            g = np.asarray(im.convert("L"))
        vecs.append(thumb_vector(g))
        ph.append(imagehash.phash(Image.fromarray(g)))
    return {
        "paths": [it[0] for it in items], "splits": [it[1] for it in items],
        "labels": [it[2] for it in items], "vecs": np.stack(vecs), "md5": md5s, "phash": ph,
    }


def best_match(Q, R, chunk=2048, exclude_self=False):
    """For each row of Q, best NCC and argmax over rows of R."""
    best = np.full(len(Q), -1.0, dtype=np.float32)
    arg = np.zeros(len(Q), dtype=np.int64)
    for s in range(0, len(Q), chunk):
        S = Q[s:s + chunk] @ R.T
        if exclude_self:
            idx = np.arange(s, min(s + chunk, len(Q)))
            S[idx - s, idx] = -1
        arg[s:s + chunk] = S.argmax(1)
        best[s:s + chunk] = S[np.arange(len(S)), arg[s:s + chunk]]
    return best, arg


def patient_group(pid):
    """Figshare PIDs such as MR024780B / MR024780D / MR024780E differ only by a
    trailing letter; where inspected, they are the same person in different
    studies (same lesion, same location). 233 PIDs -> 209 groups. Patient-level
    analyses use these groups, not raw PIDs."""
    return re.sub(r"[A-Z]$", "", pid)


def figshare_null(fig, key=patient_group):
    """For each Figshare slice (true orientation), the best NCC against slices
    from a *different* patient: guaranteed non-duplicates from the same domain,
    same scanner pool and same contrast, i.e. the hardest realistic negatives.
    Also the best NCC against *same*-patient other slices, for reference.
    key=str gives the raw-PID version, which is contaminated by repeat studies."""
    V = fig["vecs"]["transpose"]
    pids = np.array([key(p) for p in fig["pids"]])
    cross = np.full(len(V), -1.0, dtype=np.float32)
    same = np.full(len(V), np.nan, dtype=np.float32)
    for s in range(0, len(V), 1024):
        S = V[s:s + 1024] @ V.T
        for j in range(len(S)):
            i = s + j
            m = pids == pids[i]
            cross[i] = S[j][~m].max()
            m[i] = False
            if m.any():
                same[i] = S[j][m].max()
    return cross, same


def main():
    ensure_datasets()
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "features.pkl")
    if os.path.exists(cache):
        print("loading cached features", flush=True)
        with open(cache, "rb") as f:
            fig, corpora = pickle.load(f)
    else:
        print("hashing Figshare (.mat, all 8 orientations)", flush=True)
        fig = load_figshare()
        corpora = {}
        for name, (root, layout) in corpus_specs().items():
            print(f"hashing {name}", flush=True)
            corpora[name] = load_corpus(name, root, layout)
        with open(cache, "wb") as f:
            pickle.dump((fig, corpora), f)

    out = {"ncc_threshold": NCC_THRESHOLD, "thumb": THUMB,
           "figshare": {"slices": len(fig["ids"]), "pids": len(set(fig["pids"])),
                        "patient_groups": len({patient_group(p) for p in fig["pids"]}),
                        "multi_pid_groups": {g: sorted(v) for g, v in
                                             ((g, {p for p in set(fig["pids"]) if patient_group(p) == g})
                                              for g in {patient_group(p) for p in fig["pids"]}) if len(v) > 1},
                        "per_class": dict(Counter(fig["labels"]))},
           "corpora": {n: {"images": len(c["paths"]), "per_split_class": {f"{s}/{l}": v for (s, l), v in
                           Counter(zip(c["splits"], c["labels"])).items()}} for n, c in corpora.items()}}

    # ---- 1. Null distribution and threshold justification -------------------
    print("null distribution", flush=True)
    cross, same = figshare_null(fig)
    cross_raw, _ = figshare_null(fig, key=str)
    out["null"] = {
        "cross_raw_pid_best_ncc": {"max": float(cross_raw.max()), "p999": float(np.percentile(cross_raw, 99.9)),
                                   "n_at_or_above_threshold": int((cross_raw >= NCC_THRESHOLD).sum())},
        "cross_patient_best_ncc": {"p50": float(np.median(cross)), "p99": float(np.percentile(cross, 99)),
                                   "p999": float(np.percentile(cross, 99.9)), "max": float(cross.max()),
                                   "n_at_or_above_threshold": int((cross >= NCC_THRESHOLD).sum())},
        "same_patient_best_ncc": {"p50": float(np.nanmedian(same)), "max": float(np.nanmax(same)),
                                  "n_at_or_above_threshold": int(np.nansum(same >= NCC_THRESHOLD))},
    }
    # Second null from a different source: Br35H "no tumor" images can't be in Figshare.
    br = corpora["Br35H"]
    no_idx = np.array([i for i, l in enumerate(br["labels"]) if l == "notumor"])
    bn, _ = best_match(br["vecs"][no_idx], fig["vecs"]["transpose"])
    out["null"]["br35h_notumor_vs_figshare"] = {"n": int(len(no_idx)), "max": float(bn.max()),
                                                 "p999": float(np.percentile(bn, 99.9)),
                                                 "n_at_or_above_threshold": int((bn >= NCC_THRESHOLD).sum())}

    # ---- 2. Orientation: which dihedral transform do Kaggle copies match? ---
    print("orientation analysis", flush=True)
    orient = {}
    per_corpus_fig = {}
    for name, c in corpora.items():
        bests = {}
        for k, V in fig["vecs"].items():
            bests[k] = best_match(c["vecs"], V)
        stack = np.stack([bests[k][0] for k in DIHEDRAL])
        which = stack.argmax(0)
        keys = list(DIHEDRAL)
        best_ncc = stack.max(0)
        best_arg = np.array([bests[keys[w]][1][i] for i, w in enumerate(which)])
        hit = best_ncc >= NCC_THRESHOLD
        orient[name] = {
            "matches_any_orientation": int(hit.sum()),
            "matches_by_orientation": {keys[w]: int(n) for w, n in Counter(which[hit]).items()},
            "matches_if_only_identity": int((bests["identity"][0] >= NCC_THRESHOLD).sum()),
            "matches_if_only_transpose": int((bests["transpose"][0] >= NCC_THRESHOLD).sum()),
        }
        per_corpus_fig[name] = (best_ncc, best_arg, which)
    out["orientation"] = orient

    # pHash, as in the v1 audit, with and without the transpose fix.
    ph = {}
    for name in ("Nickparvar-v1", "Bohaju"):
        c = corpora[name]
        res = {}
        for k in ("identity", "transpose"):
            F = np.array([np.unpackbits(np.packbits(h.hash.ravel())) for h in fig["phash"][k]], dtype=np.int16)
            Q = np.array([np.unpackbits(np.packbits(h.hash.ravel())) for h in c["phash"]], dtype=np.int16)
            # Hamming distance = (#bits) - matches; matches = Q.F + (1-Q).(1-F)
            d = 64 - (Q @ F.T + (1 - Q) @ (1 - F).T)
            dmin = d.min(1)
            res[k] = {"d_le_4": int((dmin <= 4).sum()), "d_le_6": int((dmin <= 6).sum()),
                      "d_le_10": int((dmin <= 10).sum()), "median_nn_distance": float(np.median(dmin))}
        ph[name] = res
    out["phash_v1_method"] = ph

    # ---- 3. Figshare provenance, label agreement, patient recovery ----------
    print("provenance and labels", flush=True)
    prov = {}
    rows = []
    for name, c in corpora.items():
        best_ncc, best_arg, which = per_corpus_fig[name]
        hit = best_ncc >= NCC_THRESHOLD
        by_class = {}
        for cls in sorted(set(c["labels"])):
            m = np.array(c["labels"]) == cls
            by_class[cls] = {"n": int(m.sum()), "from_figshare": int((hit & m).sum())}
        agree = Counter()
        for i in np.where(hit)[0]:
            fl = fig["labels"][best_arg[i]]
            kl = c["labels"][i]
            agree[f"{kl}<-{fl}"] += 1
            rel = os.path.relpath(c["paths"][i], KAGGLE_DIR).replace(os.sep, "/")
            rows.append([name, rel, c["splits"][i], kl, fig["ids"][best_arg[i]], fl,
                         fig["pids"][best_arg[i]], f"{best_ncc[i]:.4f}", list(DIHEDRAL)[which[i]]])
        # Distinct Figshare slices covered (several Kaggle copies may hit one slice).
        prov[name] = {"from_figshare": int(hit.sum()), "fraction": float(hit.mean()),
                      "distinct_figshare_slices": int(len(set(best_arg[hit]))),
                      "distinct_figshare_pids": int(len({fig["pids"][j] for j in best_arg[hit]})),
                      "distinct_figshare_patient_groups": int(len({patient_group(fig["pids"][j]) for j in best_arg[hit]})),
                      # Tumor images just below threshold: mostly true copies on visual
                      # inspection, but this band overlaps the null, so it is not counted.
                      "tumor_in_lower_band": int(((best_ncc >= LOWER_BAND) & ~hit &
                                                  (np.array(c["labels"]) != "notumor")).sum()),
                      "by_class": by_class, "label_pairs(kaggle<-figshare)": dict(agree)}
    out["figshare_provenance"] = prov
    with open(os.path.join(RESULTS_DIR, "overlap_audit_v2_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corpus", "path", "split", "kaggle_label", "figshare_id", "figshare_label", "figshare_pid", "ncc", "orientation"])
        w.writerows(rows)

    # ---- 4. Within-corpus train/test leakage --------------------------------
    print("train/test leakage", flush=True)
    leak = {}
    for name, c in corpora.items():
        sp = np.array(c["splits"])
        if not ((sp == "train").any() and (sp == "test").any()):
            continue
        tr, te = np.where(sp == "train")[0], np.where(sp == "test")[0]
        b, a = best_match(c["vecs"][te], c["vecs"][tr])
        md5_tr = {c["md5"][i] for i in tr}
        exact = sum(c["md5"][i] in md5_tr for i in te)
        hit = b >= NCC_THRESHOLD
        lab = np.array(c["labels"])
        # Patient-level: test images traceable to Figshare whose patient is also in train.
        best_ncc, best_arg, _ = per_corpus_fig[name]
        fhit = best_ncc >= NCC_THRESHOLD
        train_pids = {fig["pids"][best_arg[i]] for i in tr if fhit[i]}
        train_groups = {patient_group(p) for p in train_pids}
        te_traced = [i for i in te if fhit[i]]
        te_pid_leak = sum(fig["pids"][best_arg[i]] in train_pids for i in te_traced)
        te_group_leak = sum(patient_group(fig["pids"][best_arg[i]]) in train_groups for i in te_traced)
        leak[name] = {
            "test_images": int(len(te)),
            "test_exact_dup_in_train": int(exact),
            "test_near_dup_in_train": int(hit.sum()),
            "test_near_dup_fraction": float(hit.mean()),
            "near_dup_label_conflicts": int((lab[te][hit] != lab[tr][a[hit]]).sum()),
            "near_dup_by_test_class": dict(Counter(lab[te][hit])),
            "test_traced_to_figshare": len(te_traced),
            "test_traced_pid_in_train": int(te_pid_leak),
            "test_traced_patient_in_train": int(te_group_leak),
        }
    out["train_test_leakage"] = leak

    # ---- 5. Internal redundancy and cross-corpus containment ----------------
    print("cross-corpus containment", flush=True)
    names = list(corpora)
    contain = {}
    for a_name in names:
        contain[a_name] = {}
        for b_name in names:
            if a_name == b_name:
                b, _ = best_match(corpora[a_name]["vecs"], corpora[a_name]["vecs"], exclude_self=True)
                contain[a_name]["_internal_near_dup"] = int((b >= NCC_THRESHOLD).sum())
                continue
            b, _ = best_match(corpora[a_name]["vecs"], corpora[b_name]["vecs"])
            contain[a_name][b_name] = float((b >= NCC_THRESHOLD).mean())
        contain[a_name]["Figshare"] = prov[a_name]["fraction"]
    out["containment"] = contain  # contain[A][B] = fraction of A with a near-dup in B

    # ---- 6. Source decomposition of Nickparvar-v1 (its documented recipe) ---
    c = corpora["Nickparvar-v1"]
    srcs = {"Figshare": per_corpus_fig["Nickparvar-v1"][0] >= NCC_THRESHOLD}
    for s in ("SARTAJ", "Br35H"):
        srcs[s] = best_match(c["vecs"], corpora[s]["vecs"])[0] >= NCC_THRESHOLD
    decomp = {}
    for cls in sorted(set(c["labels"])):
        m = np.array(c["labels"]) == cls
        decomp[cls] = {"n": int(m.sum())}
        for s, h in srcs.items():
            decomp[cls][s] = int((h & m).sum())
        decomp[cls]["none_of_the_three"] = int((m & ~(srcs["Figshare"] | srcs["SARTAJ"] | srcs["Br35H"])).sum())
    out["nickparvar_v1_decomposition"] = decomp

    with open(os.path.join(RESULTS_DIR, "overlap_audit_v2.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.join(CACHE_DIR, "null.npz"), cross=cross, cross_raw=cross_raw, same=same, br35h_no=bn,
             nick_best=per_corpus_fig["Nickparvar-v1"][0], brisc_best=per_corpus_fig["BRISC"][0],
             sartaj_best=per_corpus_fig["SARTAJ"][0])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
