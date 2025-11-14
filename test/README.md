# Inference Utilities

The scripts in `Nuclass/test/` reproduce the evaluation figures for Path-A, Path-B, and the fusion gate. They no longer contain lab-specific paths; everything is driven by command-line arguments and a holdout configuration file.

## Holdout Configuration

All three scripts expect a JSON file describing the evaluation datasets:

```json
{
  "ovary": {
    "root": "/abs/path/to/ovary_holdout",
    "mode": "xenium_exact",
    "tissue": "ovary"
  },
  "lung": {
    "root": "/abs/path/to/lung_holdout",
    "mode": "xenium_exact",
    "tissue": "lung"
  }
}
```

Use `Nuclass/test/configs/holdouts_template.json` as a starting point and replace each `root` with the actual location of your holdout H5/JSON datasets. The `mode` field must be `xenium_exact`, and `tissue` determines the tissue index used for the run.

## Path-A Inference

```bash
python -m Nuclass.test.A_infrence \
  --ckpt Nuclass/checkpoints/path_local_release/best.ckpt \
  --holdouts Nuclass/test/configs/holdouts_template.json \
  --output-dir Nuclass/results/test/path_a_inference \
  --select ovary lung
```

## Path-B Inference

```bash
python -m Nuclass.test.B_inference \
  --ckpt Nuclass/checkpoints/path_global_release/best.ckpt \
  --holdouts-config Nuclass/test/configs/holdouts_template.json \
  --output-dir Nuclass/results/test/path_b_inference \
  --select ovary lung
```

Optional flags:

- `--ctx_size` (default 512) controls the context resize.
- `--mb` sets the DINO micro-batch size.
- `--batch` controls the inference batch size.

## Fusion Gate (A+B) Inference

```bash
python -m Nuclass.test.AB_inference \
  --ckpt Nuclass/checkpoints/path_gate_release/best.ckpt \
  --holdouts-config Nuclass/test/configs/holdouts_template.json \
  --output-dir Nuclass/results/test/gate_inference
```

Add `--select` to restrict to a subset of holdouts, and use `--ctx_size / --mb / --batch` to match your hardware limits.

Each script emits NumPy arrays (`embeddings.npy`, `probs.npy`, etc.), per-holdout metrics, and confusion matrices under the selected output directory. Use the same holdout config across the three scripts to ensure class alignment.
