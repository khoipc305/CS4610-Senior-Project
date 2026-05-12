# PNG sample slices

13 axial PNG slices (`slice_01.png` ... `slice_13.png`) extracted from
`LIDC-IDRI-0006_3000563` and lung-windowed to Hounsfield range
`[-1000, +400]` rescaled to 0-255.

The slices span axial z = 64 to 76. The radiologist-annotated nodule
appears on slices **05 through 08** (z = 68-71), with the centroid on
**slice 07**.

## How to use these in the deployment app

```powershell
py -m streamlit run app.py
```

Then drag-and-drop *into the file uploader*:

- **A single PNG** -> the app tiles it along the depth axis to make a
  thin pseudo-3D volume. Useful for the fastest possible "does the
  pipeline run" check.
- **All 13 PNGs at once** -> the app stacks them in alphabetical
  filename order, producing a 13-slice 3D volume that contains the
  actual nodule. The Grad-CAM heatmap should localize on slice 07.

## Why PNGs and not just NIfTI?

Many reviewers do not have DICOM/NIfTI tooling, but every machine has
PNG support. The Streamlit app accepts both, so this folder makes the
demo runnable from a fresh laptop in under a minute.

## Provenance

- Source volume: `dataset/sample/ct/LIDC-IDRI-0006_3000563.nii.gz`
- Source mask:   `dataset/sample/masks/LIDC-IDRI-0006_3000563_segmask.nii.gz`
- Both originate from the public **LIDC-IDRI** collection
  (Armato III et al., 2011) distributed by The Cancer Imaging Archive.
- Lung-windowing applied for display; geometric content unchanged.
