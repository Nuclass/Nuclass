# Nuclass Training Release

This directory contains the training scripts for the Path-Local (A), Path-Global (B), and fusion Gate models that appear in the double-blind manuscript. The code paths are now environment-agnostic so collaborators can clone, configure, and utilize the same experiments without editing source files.

## Utilize

1. Create the runtime environment (CUDA 12.x) with `conda env create -f environment.yaml` and `conda activate nuclass`.
2. Point the code to your data once per session:  
   `export NUCLASS_DATA_ROOT=/abs/path/to/WSI_patches`  
   or list explicit folders with `export NUCLASS_DATA_DIRS="/data/A:/scratch/B"`.
3. Place the released checkpoints under `Nuclass/checkpoints/...` (default) or set `NUCLASS_CHECKPOINT_ROOT`.
4. Optional: direct inference dumps elsewhere via `NUCLASS_RESULTS_ROOT`.

After the variables are in place you can launch any experiment from the repo root. Each trainer reads a YAML config (see `Nuclass/configs/`) and lets you override any field via `-o key=value` on the command line:

```bash
# run with the published defaults
python -m Nuclass.train.path_local  --config Nuclass/configs/path_local.yaml
python -m Nuclass.train.path_global --config Nuclass/configs/path_global.yaml
python -m Nuclass.train.path_gate   --config Nuclass/configs/path_gate.yaml

# override any hyper-parameter on demand (dot notation supported)
python -m Nuclass.train.path_local \
  --config Nuclass/configs/path_local.yaml \
  -o max_epochs=20 \
  -o gpus=[0,1] \
  -o lr_head=1e-4
```

Each script looks up datasets, checkpoints, and output folders through `train/config_utils.py`, so no relative or machine-specific paths remain in the code.
Use `--print-config` to inspect the fully merged configuration (YAML values and overrides) before launching a run.

## Key Environment Knobs

| Variable | Description |
| --- | --- |
| `NUCLASS_DATA_ROOT` | Absolute path whose subfolders contain the eight organ datasets (used when `NUCLASS_DATA_DIRS` is unset). |
| `NUCLASS_DATA_DIRS` | `os.pathsep` separated list of dataset folders if they live in multiple roots. |
| `NUCLASS_CHECKPOINT_ROOT` | Directory that stores and loads checkpoints (defaults to `Nuclass/checkpoints`). |
| `NUCLASS_RESULTS_ROOT` | Directory for inference dumps, plots, and evaluation CSVs (defaults to `Nuclass/results`). |

If you rely on `NUCLASS_DATA_ROOT`, place the eight organ folders listed in `Nuclass/train/dataset_defaults.py` directly under that directory. Otherwise export `NUCLASS_DATA_DIRS` with explicit absolute paths.

Weights & Biases logging is optional; set `WANDB_API_KEY`, `WANDB_ENTITY`, or override the values inside each config if you need a different workspace.

## Configuration Files

- `Nuclass/configs/path_local.yaml` – Path-local (UNI + FiLM head) training recipe.
- `Nuclass/configs/path_global.yaml` – Path-global (UNI + DINO context) training recipe.
- `Nuclass/configs/path_gate.yaml` – Fusion gate training recipe that consumes the Path-A/B checkpoints.

Every key in these YAML files can be overridden ad hoc via `-o key=value`. The dot notation supports nested updates (e.g. `-o unfreeze.B_head=false`). This mirrors the workflow common in open-source releases: clone, edit/duplicate a config, or inject overrides per experiment without touching the Python sources.

## Figure Reproduction

Every paper-figure helper lives under `Nuclass/figure_reproduce/` with English-only CLIs:

- `film_concat_ablation/` replays the FiLM-vs-concat ablation and emits both the PDF summary and the per-metric JSON (bootstrap deltas, clustering scores, etc.).
- `grad_cam/` hosts the PLIP, LOKI, MUSK, and Path-A Grad‑CAM scripts. Each one takes `--h5-path`, `--ckpt-path`, and `--keys`, so you can aim them at your own patches without touching source paths.
- `local_global_complementary/` contains the A/B complementarity utility that measures A-only/B-only/Both/Neither counts and writes the CSV used in the manuscript.

See `Nuclass/figure_reproduce/README.md` for invocation snippets covering every script above.

## Table Reproduction

`Nuclass/table_reproduce/` contains the automation for the release tables:

- `table_reproduce/best_ckpts/` lists the checkpoint IDs pulled into the paper.
- `table_reproduce/evals/` captures the emitted `probs/`, `logs/`, `f1_align_report/`, and other artifacts so reviewers can diff the raw scores against the PDF.
- The `README.md` in that directory documents how to rerun a table end-to-end (Path-A, Path-B, fusion gate, plus the GateMix safety policy evaluation).

Point the scripts at `Nuclass/checkpoints/` (or override `NUCLASS_CHECKPOINT_ROOT`) to reuse the public weights, or drop in your own checkpoints before re-running.

## Inference Scripts

Path-A, Path-B, and GateMix inference utilities live under `Nuclass/test/`. Configure your holdout datasets via `Nuclass/test/configs/holdouts_template.json` (or your own JSON file) and follow the CLI examples in `Nuclass/test/README.md` to reproduce the evaluation tables.

## Checkpoints and Outputs

- Released checkpoints for Path-A, Path-B, and the fusion gate live under `Nuclass/checkpoints`, matching the relative paths already wired into the configs.
- New training runs will emit Lightning checkpoints into `<checkpoint_root>/<experiment_name>` and inference artifacts into `<results_root>/<experiment_name>`.
- Safety policies (`safety_policy.json`) and confusion matrices are stored alongside the checkpoints so reviewers can audit fusion behaviour.

With this layout the repository stays self-contained, while every environment-dependent location can be swapped by editing a single environment variable rather than the training code.

## Preparing Your Own Dataset

All inference/training entry points expect each holdout folder to expose the same trio of artifacts that ship with the demo pancreas set:

```
<dataset_root>/
├── annotations.json          # per-patch metadata
├── patches_224x224.h5        # RGB patches used by Path-Local / fusion A-branch
└── patches_1024x1024.h5      # Context tiles used by Path-Global / fusion B-branch
```

**H5 layout.** Each H5 stores either:

- a merged dataset (`images` dataset with shape `[N, H, W, 3]` plus a `keys` dataset containing UTF-8 patch IDs), or  
- individual datasets per key (e.g., `/abcd1234-1`, `/efgh5678-1`, …).

Whichever layout you pick, the ASCII keys must match across the 224×224 and 1024×1024 files as well as in `annotations.json`. The inference scripts call `H5Reader.get(key)` which probes the merged store first and falls back to per-dataset lookup, so no further code changes are required.

> **Path requirements:**  
> • Path-Local only uses `patches_224x224.h5`, so you can store just the 224 px crops when you only need A-branch inference.  
> • Path-Global and GateMix consume both resolutions (224 for UNI, 1024 for the DINO context), so keep both files side-by-side when you plan to run B-branch or fusion experiments.

**annotations.json.** Store one object per key with at least a label column (`class_id`, `label`, `cell_type`, `type`, or `annotation`). The preprocessing tutorial emits this schema automatically; otherwise create a JSON that looks like:

```json
{
  "abcd1234-1": {"label": "Acinar", "tissue": "pancreas"},
  "efgh5678-1": {"label": "T_Cell", "tissue": "pancreas"}
}
```

**Recommended placement.** Place each dataset under `Nuclass/demo/<dataset_name>` or another absolute directory and point the holdout config (see `Nuclass/test/configs/holdouts_template.json`) to that root:

```json
{
  "my_pancreas_holdout": {
    "root": "/abs/path/to/Nuclass/demo/my_pancreas",
    "mode": "xenium_exact",
    "tissue": "pancreas"
  }
}
```

`root` is the folder that contains the `.h5` and `annotations.json` files. `mode` selects the label-normalization recipe (`xenium_exact` for one-to-one mapping, `lizard_grouped` for 3-class grouping), and `tissue` feeds the FiLM embedding.

Once this structure is in place you can re-run the notebook demo, the CLI inference scripts under `Nuclass/test/`, or the training scripts without touching source files—just update the holdout JSON or the `NUCLASS_DATA_DIRS` environment variable to include the new dataset root.

### Quick sanity-check on the demo notebook

If you only want to verify the release, launch `Nuclass/demo/cell_type_prediction.ipynb` and follow the instructions: it consumes a small pancreas Xenium subset (`Nuclass/demo/pancreas/...`) and runs Path-Local, Path-Global, and GateMix end-to-end. The notebook is wired up to the sanitized inference modules, so it doubles as an example of how to call the CLI utilities programmatically.

### Preprocessing your own Xenium export

We bundled an annotated walkthrough under `Nuclass/preprocess_tutorial/`. The colon notebook shows how to:

1. Point the tutorial at a Xenium `outs/` directory (raw `cell_feature_matrix.h5`, `cells.parquet`, etc.).
2. Generate the normalized AnnData objects, QC figures, and JSON outputs.
3. Produce the H&E-aligned tiles and the pair of `patches_224x224.h5` / `patches_1024x1024.h5` files expected by the inference scripts.

Follow the same steps for your own samples and drop the resulting `annotations.json` + H5 bundles into a folder referenced by your holdout configuration.
