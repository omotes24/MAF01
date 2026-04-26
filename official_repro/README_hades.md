# Hades Paths

このファイルに書く `OODD_REPO` と `OODD_CHECKPOINT` は、hades 上で「見つかった既存値」ではありません。
ここでは、`/home/omote` 配下に新しく固定する運用値として定義します。

## Fixed Paths

- `DATA_SRC=/home/omote/WILD_DATA/splits`
- `OODD_REPO=/home/omote/OODD`
- `OODD_CHECKPOINT=/home/omote/OODD/checkpoints/resnet50-0676ba61.pth`
- `PROJECT_REPO=/home/omote/MAF-OOD-v51`
- `SAVE_ROOT=/home/omote/maf_ood_v51`

## Why This Checkpoint

`OODD` の ImageNet `resnet50.yml` では、PyTorch の pretrained ResNet-50 checkpoint を使う設定になっています。
そのデフォルト名が `resnet50-0676ba61.pth` です。

したがって、hades 側では次の URL から取得して

- `https://download.pytorch.org/models/resnet50-0676ba61.pth`

このパスに置く運用にします。

- `/home/omote/OODD/checkpoints/resnet50-0676ba61.pth`

## Setup

まず OODD を `/home/omote` 配下に作ります。

```bash
cd /home/omote/MAF-OOD-v51
bash official_repro/setup_oodd_on_hades.sh
```

このスクリプトは次を行います。

1. `https://github.com/zxk1212/OODD.git` を `/home/omote/OODD` に clone
2. commit `edbb1a32e5fe81e443942156f0a9cafb0297d95b` に checkout
3. `resnet50-0676ba61.pth` を `/home/omote/OODD/checkpoints/` に保存

## Track I Run

PyTorch は hades のドライバに合わせて CUDA 12.6 wheel に固定します。

```bash
cd /home/omote/MAF-OOD-v51

CUDA_VISIBLE_DEVICES=1 \
DATA_SRC=/home/omote/WILD_DATA/splits \
SAVE_ROOT=/home/omote/maf_ood_v51 \
OODD_REPO=/home/omote/OODD \
OODD_CHECKPOINT=/home/omote/OODD/checkpoints/resnet50-0676ba61.pth \
TORCH_VERSION=2.6.0 \
TORCHVISION_VERSION=0.21.0 \
TORCH_WHL_INDEX=https://download.pytorch.org/whl/cu126 \
bash run_official_track_i_on_hades.sh
```

## Notes

- `ViM / GEN / kNN` は `timm` の `vit_base_patch16_224` を直接ロードします。
- `OODD` だけは公式 OpenOOD パイプライン条件です。
- `OODD` は README 上でも OpenOOD の setup を前提にしているため、`cv2` や `imgaug` など OpenOOD 系依存が追加で必要です。hades の pip 運用では少なくとも `opencv-python-headless` と `imgaug` を入れ、`imgaug` 互換のため `numpy<2` に固定します。
- 既に別の場所へ clone 済みなら、その実パスを環境変数で上書きしてください。
