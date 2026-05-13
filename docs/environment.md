# Environment

The reported DINOv2/WILD_DATA runs were executed on the following software stack.

| Component | Version / setting |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.6.0 |
| TorchVision | 0.21.0 |
| CUDA wheel index | `cu126` |
| CUDA runtime target | 12.6 |
| NumPy | 2.2.4 |
| pandas | 2.2.3 |
| SciPy | 1.15.2 |
| scikit-learn | 1.6.1 |
| Pillow | 11.1.0 |
| tqdm | 4.67.1 |
| PyYAML | 6.0.2 |
| timm | 1.0.15 |

The cleaned MAF implementation does not require `timm` for the DINOv2 extractor, but it is pinned because earlier comparison runs and some backbone extensions used `timm` models.

## Install With Conda

```bash
conda env create -f environment.yml
conda activate maf01
```

## Install With pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e .
```

## DINOv2 Pin

The DINOv2 model is loaded from Meta's official repository:

```text
repository: facebookresearch/dinov2
model: dinov2_vitb14
git ref: 7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c
loader: torch.hub.load("facebookresearch/dinov2:<ref>", "dinov2_vitb14")
```

This pin prevents silent changes in the PyTorch Hub source. To override it intentionally:

```bash
python scripts/extract_wild_dinov2.py \
  --data-root /path/to/WILD_DATA/splits \
  --output /path/to/analysis_v3.npz \
  --dinov2-ref main
```
