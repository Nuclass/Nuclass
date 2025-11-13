# Figure Reproduction Utilities

This directory hosts self-contained scripts for reproducing the ablation figures and qualitative Grad-CAM visualisations from the paper. Every script accepts explicit command-line arguments—no hard-coded lab paths remain—so you can run them anywhere after pointing to your datasets, checkpoints, and desired output folder.

All commands below assume you run them from the repository root and that
`NUCLASS_DATA_DIRS` or `NUCLASS_DATA_ROOT` are set when a script needs access to the
standard eight-organ dataset structure.

## Film vs. Concat Ablation

```bash
python -m Nuclass.figure_reproduce.film_concat_ablation.Film_ablation \
  --ckpt_film /path/to/film_model.ckpt \
  --ckpt_concat /path/to/concat_model.ckpt \
  --target_root /abs/path/to/Ovarian_wsi \
  --subset_size 7000 \
  --results-dir Nuclass/results/figure_reproduce/film_concat_ablation
```

Optional flags:

- `--data-dirs` overrides the dataset list (defaults to `NUCLASS_DATA_*`).
- `--device/cuda` and `--gpu_index` choose compute resources.

## Grad-CAM Scripts

Each Grad-CAM script follows the same pattern: supply the H5/JSON inputs, checkpoint path, keys to visualise, and the output directory.

```bash
# Path-A (UNI + FiLM)
python -m Nuclass.figure_reproduce.grad_cam.gradcam_ovary \
  --anno-json /abs/path/to/annotations.json \
  --h5-path /abs/path/to/patches_224x224.h5 \
  --ckpt-path Nuclass/checkpoints/path_local_release/best.ckpt \
  --keys gidnfkoa-1 dicdigfp-1 mlabfdco-1 \
  --output-dir Nuclass/results/figure_reproduce/grad_cam/path_a

# LOKI (open_clip / CoCa)
python -m Nuclass.figure_reproduce.grad_cam.gradcam_loki \
  --h5-path /abs/path/to/patches_224x224.h5 \
  --ckpt-path /abs/path/to/loki.ckpt \
  --keys gidojinm-1 ignfhjao-1 \
  --output-dir Nuclass/results/figure_reproduce/grad_cam/loki

# PLIP (CLIP ViT-L/14)
python -m Nuclass.figure_reproduce.grad_cam.gradcam_plip \
  --h5-path /abs/path/to/patches_224x224.h5 \
  --ckpt-path /abs/path/to/plip.ckpt \
  --keys gidnfkoa-1 dicdigfp-1 \
  --output-dir Nuclass/results/figure_reproduce/grad_cam/plip

# MUSK (ViT/BEiT3 + organ embedding)
python -m Nuclass.figure_reproduce.grad_cam.gradcam_musk \
  --anno-json /abs/path/to/annotations.json \
  --h5-path /abs/path/to/patches_224x224.h5 \
  --ckpt-path /abs/path/to/musk.ckpt \
  --keys gidojinm-1 ijjnlofi-1 \
  --output-dir Nuclass/results/figure_reproduce/grad_cam/musk
```

Common options:

- `--device {auto,cuda,cpu}` and `--gpu-index` control execution targets.
- `--pretrained` (LOKI) lets you point to a local CoCa initialization.
- `--tissue` / `--tissue-id` (Path-A/MUSK) switch which organ-specific embedding to use.

## Local vs. Global Complementarity (A/B)

```bash
python -m Nuclass.figure_reproduce.local_global_complementary.complementary \
  --ckptA Nuclass/checkpoints/path_local_release/best.ckpt \
  --ckptB Nuclass/checkpoints/path_global_release/best.ckpt \
  --results-dir Nuclass/results/figure_reproduce/local_global_complementary \
  --data-dirs-A /abs/path/to/Ovarian_wsi /abs/path/to/Pancreas_wsi \
  --data-dirs-B /abs/path/to/Ovarian_wsi /abs/path/to/Pancreas_wsi
```

Arguments `--data-dirs-A/B` are optional; by default the script calls
`resolve_data_dirs(DEFAULT_DATASET_FOLDERS)` so exporting `NUCLASS_DATA_ROOT`
is often enough. The resulting CSV (`ab_complementarity_no_fusion.csv`)
summarises A-only, B-only, both, and neither counts plus accuracy/F1 metrics.

---

Each script writes outputs under `Nuclass/results/figure_reproduce/...` by default,
keeping the repository self-contained. Adjust the CLI flags to match your local ckpt
names, dataset layouts, and hardware. Refer to the docstrings in each script for
fine-grained parameter descriptions.
