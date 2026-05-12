# Dataset

## What's in this folder

```
dataset/
├── README.md            <- you are here
└── sample/              <- 2 LIDC-IDRI scans bundled for smoke testing
    ├── ct/              <- CT volumes (NIfTI .nii.gz)
    └── masks/           <- ground-truth nodule masks
```

The two bundled scans (`LIDC-IDRI-0006_3000559` and
`LIDC-IDRI-0006_3000561`, ≈70 MB total) are provided **only so reviewers
can run the Streamlit deployment app end-to-end without downloading the
full dataset**. They are **not** sufficient to train the model.

## Smoke-test with the bundled sample

```powershell
# From the project root
py -m streamlit run app.py
```

In the upload widget, drag-and-drop
`dataset/sample/ct/LIDC-IDRI-0006_3000559.nii.gz`. The app will run
sliding-window inference and render the four-panel viewer. (Without a
trained checkpoint at `checkpoints/lung_cancer_attention_unet/best_model.pth`
the predictions are meaningless — see the main `README.md` to train.)

## Full LIDC-IDRI dataset

This project was trained and evaluated on **61 CT scans** from the
public **LIDC-IDRI** collection. The full dataset (~2.7 GB after the
preprocessing pipeline in this repo) is **not** stored on GitHub for
size reasons. Two options to obtain it:

### Option A — Download from TCIA (original source)
1. Register a (free) account at <https://www.cancerimagingarchive.net/>.
2. Download the *LIDC-IDRI* collection (DICOM, ~125 GB raw).
3. Use the preprocessing pipeline in `src/data/` of this repo (or any
   tool such as [pylidc](https://pylidc.github.io/)) to convert DICOM
   to the `.nii.gz` layout shown in `sample/`.

### Option B — Use the prepared NIfTI subset
The 61-scan preprocessed subset used in CS4610 (Fall 2025) and CS4620
(Spring 2026) lives locally at:

```
D:/Fall Senior Project/LIDC-exact/
├── ct/      (61 files, ≈ 2.7 GB)
└── masks/   (61 files, ≈ 9 MB)
```

For an external collaborator to obtain the same subset, contact the
author. A Zenodo mirror may be added later.

## Dataset layout the codebase expects

```
<DATASET_ROOT>/
├── ct/<patient_id>_<series_id>.nii.gz
└── masks/<patient_id>_<series_id>_segmask.nii.gz
```

CT and mask filenames are matched by removing the `_segmask` suffix.
Point `config/config_advanced.yaml`'s `data.root` field at your
`<DATASET_ROOT>`.

## License & ethics

LIDC-IDRI is released under the **TCIA Limited Access Data Use
Agreement**. The bundled sample is intended for **research and
educational evaluation only**. Predictions produced by this codebase
are **not** clinical decisions. See the disclaimer in the deployment
app.
