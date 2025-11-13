![Nuclass Figure](figures/Figure_1.pdf)

# Nuclass: Adaptive Multi-Scale Integration Unlocks Robust Cell Annotation in Histopathology Images

## Abstract

Distinguishing cell/nuclei types and subtypes from routine histopathology images is key to advancing computational understanding of human disease. Current tile-based models can capture detailed nuclear appearances but often overlook the broader tissue context that influences a cell's function and identity. Moreover, existing human annotations are coarse-grained, making it difficult to achieve fine-grained, cell subtype–level labeling.

We introduce **NuClass**, a pathologist-inspired framework for *cell-wise* multi-scale inference that integrates both morphology and microenvironmental context. Specifically, **Path local** focuses on nuclear morphology from \(224^{2}\) pixel image crops, while **Path global** models the surrounding \(1024^{2}\) pixel neighborhood. A learnable gating module fuses their per-cell probability outputs, adaptively balancing local detail and contextual cues. To promote complementary learning, we incorporate an uncertainty-guided objective that drives the global path to focus on regions where the local path is uncertain, and we ensure model interpretability through calibrated confidence and saliency visualization.

To address the lack of fine supervision, we construct a large-scale, marker-guided dataset from Xenium-like spatial transcriptomics assays, providing pixel-level cell-type labels for over two million cells across eight organs and sixteen classes. Evaluated on four fully held-out cohorts (~300k cells), NuClass achieves up to **96% F1 (macro or micro? TBD)**, outperforming strong baselines while maintaining high calibration and interpretability.

Our results demonstrate that multi-scale, uncertainty-aware fusion can bridge the gap between slide-level pathological foundation vision models and reliable, cell-level phenotyping.

---

End-to-end training, inference, and figure reproduction for the double-blind NuClass manuscript. Paths are fully configurable so collaborators can recreate every experiment after downloading the required public assets.

---

## 1. Installation

```bash
git clone https://github.com/Nuclass/Nuclass.git
cd Nuclass
conda env create -f environment.yaml
conda activate nuclass
pip install -e .
```

> The lightweight `environment.yaml` installs CUDA 12.1-compatible PyTorch, timm, transformers, open-clip, Lightning, and the plotting/data deps that appear throughout the repo.

---

## 2. Required Downloads

### 2.1 Xenium datasets

We support any 10x Genomics Xenium FFPE dataset (Human, 1.6.0 or later). Browse and download from:  
[10X_Genomics_Xenium](https://www.10xgenomics.com/datasets?configure%5BhitsPerPage%5D=50&configure%5BmaxValuesPerFacet%5D=1000&query=Xenium&refinementList%5Bspecies%5D%5B0%5D=Human)

For the release tutorial we reference the Xenium Colon add-on FFPE sample. Download all archives into `preprocess_tutorial/Xenium_colon_wsi/`:

```bash
mkdir -p preprocess_tutorial/Xenium_colon_wsi
cd preprocess_tutorial/Xenium_colon_wsi

# Output files
wget https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.6.0/Xenium_V1_hColon_Cancer_Add_on_FFPE/Xenium_V1_hColon_Cancer_Add_on_FFPE_outs.zip

# Supplemental files
wget https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hColon_Cancer_Add_on_FFPE/Xenium_V1_hColon_Cancer_Add_on_FFPE_he_image.ome.tif
wget https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hColon_Cancer_Add_on_FFPE/Xenium_V1_hColon_Cancer_Add_on_FFPE_he_imagealignment.csv
```

Unzip the zip file in-place—`preprocess_tutorial/Xenium_colon_wsi/outs/` is already ignored by git.

> We no longer ship `demo/pancreas/` to keep the repo lightweight. Running the preprocessing notebook on any Xenium dataset recreates the demo assets locally.

### 2.2 Backbones (DINOv3)

The fusion/global branches expect the DINOv3 ViT-L/16 release under `backbones/facebook/dinov3-vitl16-pretrain-lvd1689m/`. Download from Hugging Face (requires `huggingface_hub>=0.23`):

```bash
huggingface-cli download facebook/dinov3-vitl16-pretrain-lvd1689m \
  --include "*.json" "*.safetensors" "*.md" \
  --local-dir backbones/facebook/dinov3-vitl16-pretrain-lvd1689m \
  --local-dir-use-symlinks False
```

Other backbone folders follow the same layout; place any additional checkpoints under `backbones/`.

### 2.3 Released checkpoints & table artifacts

Public checkpoints (Path-A, Path-B, Gate) and the evaluation dumps referenced in `table_reproduce/` (probabilities, logits, alignment reports) are hosted on Zenodo:  
`TODO: https://zenodo.org/record/<placeholder>`

After download, unpack into:

```
Nuclass/checkpoints/          # lightning .ckpt files
Nuclass/table_reproduce/evals # per-run metrics/probs/logs
```

These directories are git-ignored, so feel free to symlink if your storage lives elsewhere.

---

## 3. Preprocess & Demo

1. Launch `preprocess_tutorial/Xenium_data_preprocess_colon.ipynb` (or duplicate for your dataset).  
   - Input: the raw Xenium archives placed above.  
   - Output: `annotations.json`, `patches_224x224.h5`, `patches_1024x1024.h5`, and QC plots.
2. Point the Nuclass scripts to the generated dataset by exporting either:
   ```bash
   export NUCLASS_DATA_ROOT=/abs/path/to/processed_datasets
   # or
   export NUCLASS_DATA_DIRS="/abs/path/to/datasetA:/abs/path/to/datasetB"
   ```
3. Run `demo/cell_type_prediction.ipynb` to sanity-check Path-A, Path-B, and GateMix end-to-end with the processed data. The notebook mirrors the CLI flows.

### Using your own pipeline

Already have a preprocessing script/notebook? Produce the same trio of files per dataset (`annotations.json`, `patches_224x224.h5`, `patches_1024x1024.h5`) and store them anywhere on disk, e.g.

```
/data/xenium/my_sample/
├── annotations.json
├── patches_224x224.h5
└── patches_1024x1024.h5
```

Then either:

- add the folder to `NUCLASS_DATA_DIRS` (colon-separated), or  
- reference it directly inside your holdout JSON (`Nuclass/test/configs/...`), or  
- drop the folder under `Nuclass/demo/<name>` if you want to reuse the demo paths.

As long as the three files exist per dataset, the training/eval scripts pick them up without further code changes.

---

## 4. Training

Every trainer accepts a YAML config plus optional overrides:

```bash
# Path-A / Path-B / Fusion gate
python -m Nuclass.train.path_local  --config Nuclass/configs/path_local.yaml
python -m Nuclass.train.path_global --config Nuclass/configs/path_global.yaml
python -m Nuclass.train.path_gate   --config Nuclass/configs/path_gate.yaml

# Override hyper-parameters without editing YAML
python -m Nuclass.train.path_local \
  --config Nuclass/configs/path_local.yaml \
  -o max_epochs=20 \
  -o trainer.devices=[0,1] \
  -o optimizer.lr_head=1e-4
```

All data/checkpoint/result paths resolve through `Nuclass/train/config_utils.py` using the environment variables above.

---

## 5. Evaluation / Test Scripts

Holdout evaluation lives under `Nuclass/test/` and is fully CLI-driven.

```bash
# Configure holdouts (root/mode/tissue) in Nuclass/test/configs/holdouts_template.json

# Path-A (local)
python Nuclass/test/local_infrence.py \
  --ckpt Nuclass/checkpoints/path_local_release/best.ckpt \
  --holdouts Nuclass/test/configs/holdouts_template.json \
  --results-dir results/test/path_a

# Path-B (global)
python Nuclass/test/global_inference.py \
  --ckpt Nuclass/checkpoints/path_global_release/best.ckpt \
  --holdouts Nuclass/test/configs/holdouts_template.json \
  --results-dir results/test/path_b

# Gate fusion
python Nuclass/test/gate_inference.py \
  --ckpt Nuclass/checkpoints/path_gate_release/best.ckpt \
  --holdouts Nuclass/test/configs/holdouts_template.json \
  --results-dir results/test/gate
```

Each script emits embeddings, logits/probs, metrics JSON, and confusion matrices under `results/test/...`.

---

## 6. Figure & Table Reproduction

- **Figure scripts (`Nuclass/figure_reproduce/`)**
  - `grad_cam/`: PLIP, LOKI, MUSK, and Path-A Grad-CAM overlays.
  - `film_concat_ablation/`: FiLM vs concat ablations (PDF + JSON metrics).
  - `local_global_complementary/`: A/B complementarity counts and CSV export.
- **Table scripts (`Nuclass/table_reproduce/`)**
  - `best_ckpts/`: source checkpoint index.  
  - `evals/`: references to the Zenodo-provided `probs/`, `logs/`, `f1_align_report/`, etc.  
  - `README.md`: per-table replay instructions.

Run any script with `--help` to view the required arguments; all of them share the path conventions described above.

---

## 7. Directory Notes

| Folder | Notes |
| --- | --- |
| `preprocess_tutorial/Xenium_colon_wsi/` | Holds raw Xenium downloads (ignored by git). |
| `demo/` | Not populated by default; run the tutorial notebook to regenerate demo assets. |
| `backbones/` | Store downloaded DINO (and other) backbone weights here. |
| `checkpoints/` | Place released or custom Lightning checkpoints (Zenodo bundle). |
| `table_reproduce/` | Contains evaluation metadata; copy the Zenodo `evals/` tree here. |

All large artifacts remain local to keep the GitHub repo code-only. The README sections above list every required download so reviewers can recreate the full workflow.
