# Official Track I Notes

このディレクトリは、`Track I` を notebook のローカル近似で埋めず、外部の原典実装結果を `CSV` として取り込むための補助です。

## 方針

- notebook / `run_seed42_dualtrack.py` が直接計算するのは `Track II` と `Proposal(MAF)` だけです。
- `Track I` は、各手法の公式 repo / 論文準拠の実行結果を `official_track_i_csv` として import します。
- `bioclip` については、現時点で各公式 repo がそのまま対応していない手法があるため、厳密な `Track I` は空欄になり得ます。
- `ViM / GEN / kNN` の `Track I` は `test/id` を ID 評価 split として使います。
- `OODD` の `Track I` は公式 OpenOOD パイプライン条件です。したがって local の `imagenet_vit / bioclip` と別条件になります。

## Pin した source version

- `ViM`: `dabf9e5b242dbd31c15e09ff12af3d11f009f79c`
- `GEN`: `1e792b56aebf75ec1106952e9093584b8ed70313`
- `kNN`: `2afb2bbed60a8d69384dc9b28e5637711345222b`
- `OODD`: `edbb1a32e5fe81e443942156f0a9cafb0297d95b`

## 先に作るもの

WILD split から公式 repo 向けの txt manifest を作ります。

```bash
python official_repro/prepare_wild_lists.py \
  --data-src /home/omote/WILD_DATA/splits \
  --out-root /home/omote/maf_ood_v51/official_inputs
```

生成物:

- `/home/omote/maf_ood_v51/official_inputs/vim_gen/train_id.txt`
- `/home/omote/maf_ood_v51/official_inputs/vim_gen/val_id.txt`
- `/home/omote/maf_ood_v51/official_inputs/vim_gen/test_id.txt`
- `/home/omote/maf_ood_v51/official_inputs/vim_gen/test_ood.txt`
- `/home/omote/maf_ood_v51/official_inputs/oodd/train_id.txt`
- `/home/omote/maf_ood_v51/official_inputs/oodd/val_id.txt`
- `/home/omote/maf_ood_v51/official_inputs/oodd/test_id.txt`
- `/home/omote/maf_ood_v51/official_inputs/oodd/test_ood.txt`

## 公式側の確認点

- `ViM`: repo README の ViT 手順では `extract_feature_vit.py` と `benchmark.py` を使います。
- `GEN`: repo README では `benchmark.py` を使い、`gamma=0.1`, `M=100` です。
- `kNN`: `run_imagenet_vit.py` は `ViT('B_16_imagenet1k')`, `Resize(384,384)`, `Normalize(0.5,0.5)`, `FAISS IndexFlatL2`, `K=1000` です。
- `OODD`: `bash.sh` の ImageNet 設定は `K1=100`, `K2=5`, `ALPHA=0.5`, `queue_size=2048` で、OpenOOD の `ImglistDataset` と `data_aux` multicrop を使います。
- `RMD`: 論文式で raw train 統計を使います。

## WILD での注意

- `ViM / GEN` は `img_list` を受けるので、上の manifest をそのまま流用しやすいです。
- `OODD` は OpenOOD の custom yaml を作れば WILD に流せます。`ImglistDataset` は `-1` ラベルの OOD 行を受けられます。
- `kNN` の公式 ImageNet ViT スクリプトは benchmark データセット名をかなり固定しているので、WILD で完全に同じファイル I/O を使うのは難しいです。`same images` を守るなら、前処理・特徴抽出・FAISS 部分を変えずに最小限の入出力だけ合わせる adapter が必要です。

## 実行コード

`hades` 上の固定パス運用は [README_hades.md](/Users/k.omote/MAF-OOD-v51/official_repro/README_hades.md) に分けてあります。

### ViM / GEN

依存:

- `timm`
- `torchvision`

実行:

```bash
python official_repro/run_vim_gen_track_i.py \
  --data-src /home/omote/WILD_DATA/splits \
  --list-root /home/omote/maf_ood_v51/official_inputs/vim_gen \
  --save-root /home/omote/maf_ood_v51/official_runs/vim_gen \
  --methods ViM GEN \
  --model vit_base_patch16_224
```

出力:

- `per_ood.csv`
- `summary_import.csv`

### kNN

依存:

- `timm`
- `faiss-cpu` または `faiss-gpu`

実行:

```bash
python official_repro/run_knn_track_i_wild.py \
  --data-src /home/omote/WILD_DATA/splits \
  --save-root /home/omote/maf_ood_v51/official_runs/knn \
  --model vit_base_patch16_224
```

出力:

- `per_ood.csv`
- `summary_import.csv`

### OODD

依存:

- clone 済みの `OODD` repo
- 公式 checkpoint

実行:

```bash
OODD_REPO=/home/omote/OODD \
DATA_SRC=/home/omote/WILD_DATA/splits \
LIST_ROOT=/home/omote/maf_ood_v51/official_inputs/oodd \
OUT_ROOT=/home/omote/maf_ood_v51/official_runs/oodd \
OODD_CHECKPOINT=/path/to/resnet50_imagenet1k.pth \
bash official_repro/run_oodd_track_i.sh
```

出力:

- `summary_import.csv`

## import 用 CSV 形式

notebook / CLI に渡す CSV は少なくとも次の列を持たせてください。

```csv
backbone,method,AUROC,AUPR-IN,AUPR-OUT,FPR95,AUTC,note
imagenet_vit,GEN,0.9000,0.9100,0.8900,0.1200,0.2200,"official repo on WILD"
```

- `track` 列は不要です。import 時に自動で `I` になります。
- `display_name` 列も不要です。自動で `METHOD [I]` になります。
- `OODD` は `backbone=official_oodd_resnet50` のような別条件名で構いません。CLI の combined summary には残ります。

## まとめて結合

```bash
python official_repro/combine_official_track_i.py \
  --inputs \
    /home/omote/maf_ood_v51/official_runs/vim_gen/summary_import.csv \
    /home/omote/maf_ood_v51/official_runs/knn/summary_import.csv \
    /home/omote/maf_ood_v51/official_runs/oodd/summary_import.csv \
  --output /home/omote/maf_ood_v51/official_track_i_results.csv
```

## notebook / CLI への取り込み

CLI:

```bash
python run_seed42_dualtrack.py \
  --data-src /home/omote/WILD_DATA/splits \
  --save-root /home/omote/maf_ood_v51 \
  --backbones imagenet_vit bioclip \
  --official-track-i-csv /home/omote/maf_ood_v51/official_track_i_results.csv
```

notebook:

- `OFFICIAL_TRACK_I_CSV = "/home/omote/maf_ood_v51/official_track_i_results.csv"` を設定して実行します。
