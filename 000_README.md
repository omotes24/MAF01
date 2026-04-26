# MAF-OOD v51 Reproducible Code Guide

この README は、第三者がこのディレクトリから MAF-OOD v51 の主要実験を再現実装・再実行できるように、必要なファイル、環境、データ配置、実行順、出力確認をまとめたものです。

このリポジトリでは、再現実行の入口を番号付きファイルで管理します。一方、既存の実装本体は `maf_ood_dual_pipeline.py` などの説明的なファイル名のまま残しています。Python は数字で始まるファイルを通常の module として import しづらいため、番号付きファイルは「実行入口」、説明的な既存ファイルは「実装本体」として分けます。

## 0. Scope

この README で再現対象にするものは次です。

- WILD split 上の MAF-OOD evaluation
- seed=42 の小規模 dual-track run
- 複数 seed / 複数 backbone の adaptive MAF study
- TERM1 foundation study
- temperature-scaling ablation
- rival method reproduction comparison
- optional な official Track I adapter run
- 論文・報告用の CSV / figure / table 生成に必要な source manifest

この repository 自体には、WILD 画像データ、事前学習 backbone のダウンロード済み重み、実験済み checkpoint は含めません。再現者は `~/WILD_DATA/splits` 相当のデータを用意し、初回実行では checkpoint と feature cache を生成します。

## 1. Numbered Entry Points

| File | Role |
| --- | --- |
| `000_README.md` | このガイド。最初に読むファイル。 |
| `001_requirements.txt` | PyTorch 以外の Python package list。CUDA 環境ごとの PyTorch wheel pin は別途指定する。 |
| `002_check_environment.py` | package import、CUDA、data layout、core module import を確認する。 |
| `010_run_seed42_dualtrack.py` | seed=42 dual-track experiment の再現入口。 |
| `011_run_official_track_i_multibackbone.py` | official Track I adapter の multi-backbone 実行入口。 |
| `012_run_term1_foundation.py` | TERM1 foundation study の再現入口。 |
| `013_run_adaptive_multiseed_study.py` | main adaptive MAF multi-seed study の再現入口。 |
| `014_run_temperature_scaling_ablation.py` | temperature-scaling ablation の再現入口。 |
| `015_run_rival_repro_comparison.py` | rival method reproduction comparison の再現入口。 |
| `020_run_seed42_dualtrack_on_hades.sh` | Hades/tmux 用 seed=42 wrapper。 |
| `021_run_official_track_i_on_hades.sh` | Hades/tmux 用 official Track I wrapper。 |
| `022_run_term1_foundation_on_hades.sh` | Hades/tmux 用 TERM1 wrapper。 |
| `023_run_adaptive_multiseed_on_hades.sh` | Hades/tmux 用 adaptive study wrapper。 |
| `024_run_temperature_ablation_on_hades.sh` | Hades/tmux 用 temperature ablation wrapper。 |
| `025_run_rival_repro_on_hades.sh` | Hades 用 rival reproduction wrapper。 |
| `090_write_repro_manifest.py` | source file の size、line count、SHA-256 を JSON に保存する。 |

## 2. Core Source Files

| File or directory | Contents |
| --- | --- |
| `maf_ood_dual_pipeline.py` | dataset loading、backbone loading、training、feature extraction、basic OOD score。 |
| `maf_ood_notebook_utils.py` | MAF scoring、adaptive alpha、summary table、plot helper、experiment helper。 |
| `corrected_vim_oodd.py` | corrected ViM / OODD-like scoring utility。 |
| `dual_track_eval.py` | dual-track evaluation utility。 |
| `run_seed42_dualtrack.py` | seed=42 dual-track の元 CLI implementation。 |
| `01_run_term1_foundation.py` | TERM1 foundation study の元 CLI implementation。 |
| `run_multiseed_adaptive_study.py` | main adaptive multi-seed study の元 CLI implementation。 |
| `run_temperature_scaling_ablation.py` | temperature ablation の元 CLI implementation。 |
| `run_rival_repro_comparison.py` | rival reproduction comparison の元 CLI implementation。 |
| `official_repro/` | official Track I adapter、manifest 作成、official source notes。 |
| `reports/` | figure / table / appendix generation scripts と生成済み report artifacts。 |

生成物は source と分けて扱います。`figs/`、`reports/*.pdf`、`reports/*.aux`、`reports/*.log`、`reports/*.out`、`rival_repro_comparison*/`、`*.csv`、`*.svg`、`__pycache__/` は output または intermediate artifact です。

## 3. Environment

GPU 実験では CUDA 対応 PyTorch が必要です。Hades で使っていた pin は次です。

```bash
torch==2.6.0
torchvision==0.21.0
--index-url https://download.pytorch.org/whl/cu126
```

新規 venv を作る場合:

```bash
cd /home/omote/MAF-OOD-v51
python3 -m venv /home/omote/.venv-maf-ood-v51
source /home/omote/.venv-maf-ood-v51/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --force-reinstall --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r 001_requirements.txt
```

CPU-only 環境では long experiment は現実的ではありません。CPU-only で行うのは source inspection、`--help`、manifest 生成、small import check 程度にしてください。その場合は環境に合う CPU PyTorch を入れた上で `pip install -r 001_requirements.txt` を使います。

## 4. Data Layout

default の data root は `~/WILD_DATA/splits` です。`--data-src` で変更できます。

期待する layout:

```text
WILD_DATA/splits/
  train/
    id/
      buffalo/
      cheetah/
      elephant/
      giraffe/
      hippo/
  val/
    id/
      buffalo/
      cheetah/
      elephant/
      giraffe/
      hippo/
  test/
    id/
      buffalo/
      cheetah/
      elephant/
      giraffe/
      hippo/
    ood/
      ...
```

ID split は class directory 直下の `*.jpg`、`*.jpeg`、`*.png` を読みます。OOD split は `test/ood/` 配下を再帰的に探索します。

事前確認:

```bash
python 002_check_environment.py --data-src ~/WILD_DATA/splits
```

FAISS を使わない run だけ確認したい場合:

```bash
python 002_check_environment.py --data-src ~/WILD_DATA/splits --skip-faiss
```

この check が通らない場合、長い GPU job は開始しないでください。

## 5. Important Run Modes

初回実行と再実行で flag の意味が違います。

| Flag | Meaning |
| --- | --- |
| no `--eval-only` | checkpoint が無ければ training し、feature cache も作る。初回再現はこちら。 |
| `--eval-only` | 既存の `best.pt` と cache を使う。checkpoint が無い場合は失敗する。 |
| `--force-reextract` | 既存の feature cache を無視して再抽出する。backbone や transform を変えた時だけ使う。 |
| `--official-track-i-csv` | official Track I result を CSV から import する。 |
| `--include-approx-track-i` | local approximation の Track I も含める。fairness が変わるので明示した時だけ使う。 |

再現論文・報告用の run では、各 output directory の `run_meta*.json` を必ず CSV と一緒に保存してください。そこに data path、save path、backbone、seed、flag が記録されます。

## 6. Minimal Reproduction

まず seed=42、backbone 2 種の小さな run で pipeline を確認します。

初回実行:

```bash
CUDA_VISIBLE_DEVICES=0 python 010_run_seed42_dualtrack.py \
  --data-src ~/WILD_DATA/splits \
  --save-root ~/maf_ood_v51 \
  --backbones imagenet_vit bioclip
```

既に checkpoint / cache がある再実行:

```bash
CUDA_VISIBLE_DEVICES=0 python 010_run_seed42_dualtrack.py \
  --data-src ~/WILD_DATA/splits \
  --save-root ~/maf_ood_v51 \
  --backbones imagenet_vit bioclip \
  --eval-only
```

主な出力:

```text
~/maf_ood_v51/
  imagenet_vit/seed42/best.pt
  imagenet_vit/seed42/analysis_v3.npz
  bioclip/seed42/best.pt
  bioclip/seed42/analysis_v3.npz
  seed42_dualtrack_summary/
    combined_results_seed42.csv
    combined_alpha_sweep_seed42.csv
    run_meta_seed42.json
```

成功確認:

```bash
ls -lh ~/maf_ood_v51/seed42_dualtrack_summary
python - <<'PY'
from pathlib import Path
import pandas as pd
p = Path("~/maf_ood_v51/seed42_dualtrack_summary/combined_results_seed42.csv").expanduser()
df = pd.read_csv(p)
print(df[["backbone", "track", "method", "AUROC", "FPR95"]].head(20).to_string(index=False))
PY
```

## 7. Main Adaptive Multi-Seed Study

MAF adaptive rule の主実験です。標準は `imagenet_vit` と `bioclip`、seed は `42 123 456` です。

初回実行:

```bash
CUDA_VISIBLE_DEVICES=0 python 013_run_adaptive_multiseed_study.py \
  --data-src ~/WILD_DATA/splits \
  --save-root ~/maf_ood_v51 \
  --backbones imagenet_vit bioclip \
  --seeds 42 123 456
```

既存 checkpoint / cache を使う再実行:

```bash
CUDA_VISIBLE_DEVICES=0 python 013_run_adaptive_multiseed_study.py \
  --data-src ~/WILD_DATA/splits \
  --save-root ~/maf_ood_v51 \
  --backbones imagenet_vit bioclip \
  --seeds 42 123 456 \
  --eval-only
```

主な出力 directory:

```text
~/maf_ood_v51/adaptive_multiseed_study_summary/
```

重要な CSV:

| File | Meaning |
| --- | --- |
| `local_results_all_seeds.csv` | seed ごとの全 method metrics。 |
| `local_method_summary_mean_std.csv` | method ごとの mean/std summary。 |
| `adaptive_rule_all_seeds.csv` | seed ごとの adaptive alpha rule parameter。 |
| `adaptive_rule_summary_mean_std.csv` | adaptive rule parameter の summary。 |
| `adaptive_ablation_all_seeds.csv` | conf/cons/margin/product などの ablation。 |
| `alpha_alignment_summary_mean_std.csv` | adaptive alpha と oracle fixed alpha の対応。 |
| `subgroup_margin_metrics_summary_mean_std.csv` | margin bin ごとの subgroup metrics。 |
| `claim_ablation_comparisons_summary_mean_std.csv` | claim-oriented ablation summary。 |
| `integrated_summary_with_track_i.csv` | local summary と optional Track I import の統合。 |
| `run_meta.json` | 実行条件。必ず保存する。 |

成功確認:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("~/maf_ood_v51/adaptive_multiseed_study_summary").expanduser()
df = pd.read_csv(root / "local_method_summary_mean_std.csv")
cols = ["backbone", "track", "method", "AUROC_mean_std", "FPR95_mean_std", "AUPR-OUT_mean_std"]
print(df[cols].groupby("backbone", group_keys=False).head(8).to_string(index=False))
PY
```

## 8. TERM1 Foundation Study

TERM1 foundation study は、default で 7 backbones と 5 seeds を対象にします。

default backbones:

```text
imagenet_vit openai_clip_b16 openai_clip_b32 bioclip dinov2_vitb14 resnet50 swin_base
```

default seeds:

```text
42 123 456 789 1011
```

初回実行:

```bash
CUDA_VISIBLE_DEVICES=0 python 012_run_term1_foundation.py \
  --data-src ~/WILD_DATA/splits \
  --save-root ~/260421_term1 \
  --artifact-root ~/maf_ood_v51
```

既存 artifact を使う再実行:

```bash
CUDA_VISIBLE_DEVICES=0 python 012_run_term1_foundation.py \
  --data-src ~/WILD_DATA/splits \
  --save-root ~/260421_term1 \
  --artifact-root ~/maf_ood_v51 \
  --eval-only
```

主な出力:

```text
~/260421_term1/term1_foundation_summary/
  seed_raw_method_metrics.csv
  local_method_summary_mean_std.csv
  proposal_vs_best_baseline_all_seeds.csv
  proposal_vs_best_baseline_summary_mean_std.csv
  alpha_sweep_all_seeds.csv
  integrated_summary_with_track_i.csv
  run_meta.json
```

注意: `--artifact-root` は checkpoint / feature cache の保存・参照先です。既に `~/maf_ood_v51/{backbone}/seed{seed}/best.pt` がある場合、`--eval-only` で再利用できます。

## 9. Temperature-Scaling Ablation

MAF consistency term の temperature grid を比較します。

初回または cache 再利用実行:

```bash
CUDA_VISIBLE_DEVICES=0 python 014_run_temperature_scaling_ablation.py \
  --data-src ~/WILD_DATA/splits \
  --save-root ~/260422_temperature_ablation \
  --artifact-root ~/maf_ood_v51 \
  --backbones imagenet_vit openai_clip_b16 bioclip \
  --seeds 42 123 456 \
  --skip-missing
```

既存 checkpoint / cache が揃っている場合:

```bash
CUDA_VISIBLE_DEVICES=0 python 014_run_temperature_scaling_ablation.py \
  --data-src ~/WILD_DATA/splits \
  --save-root ~/260422_temperature_ablation \
  --artifact-root ~/maf_ood_v51 \
  --backbones imagenet_vit openai_clip_b16 bioclip \
  --seeds 42 123 456 \
  --eval-only \
  --skip-missing
```

主な出力:

```text
~/260422_temperature_ablation/temperature_scaling_ablation/
  temperature_ablation_all_seeds.csv
  temperature_ablation_summary_mean_std.csv
  temperature_entropy_diagnostics_all_seeds.csv
  temperature_entropy_diagnostics_summary_mean_std.csv
  best_temperature_adaptive.csv
  best_temperature_best_fixed_alpha.csv
  run_meta.json
```

## 10. Rival Reproduction Comparison

既存 artifact root の `best.pt` と `analysis_v3.npz` を使い、rival methods と MAF を同じ cached split 上で比較します。

```bash
CUDA_VISIBLE_DEVICES=0 python 015_run_rival_repro_comparison.py \
  --artifact-root ~/maf_ood_v51 \
  --output-root ~/maf_ood_v51/rival_repro_comparison \
  --backbones dinov2_vitb14 imagenet_vit openai_clip_b16 bioclip \
  --seeds 42 123 456
```

途中結果を再利用する場合:

```bash
CUDA_VISIBLE_DEVICES=0 python 015_run_rival_repro_comparison.py \
  --artifact-root ~/maf_ood_v51 \
  --output-root ~/maf_ood_v51/rival_repro_comparison \
  --backbones dinov2_vitb14 imagenet_vit openai_clip_b16 bioclip \
  --seeds 42 123 456 \
  --skip-existing
```

主な出力:

```text
~/maf_ood_v51/rival_repro_comparison/
  per_seed/
  rival_results_all_seeds.csv
  rival_summary_mean_std.csv
  rival_summary_with_oracle_mean_std.csv
  rival_best_by_backbone.csv
  rival_summary_mean_std.tex
  rival_summary_with_oracle_mean_std.tex
  run_meta.json
```

## 11. Official Track I Import

official Track I は local approximation ではなく、official-style adapter の出力 CSV を import します。`official_repro/README.md` と `official_repro/README_hades.md` も参照してください。

### 11.1 ViM / GEN / KNN adapters

`011_run_official_track_i_multibackbone.py` は `--pairs backbone=timm_model_name` が必須です。例:

```bash
CUDA_VISIBLE_DEVICES=0 python 011_run_official_track_i_multibackbone.py \
  --data-src ~/WILD_DATA/splits \
  --save-root ~/260421_term1 \
  --pairs \
    imagenet_vit=vit_base_patch16_224 \
    resnet50=resnet50 \
    swin_base=swin_base_patch4_window7_224 \
  --output ~/260421_term1/official_track_i_results_multi_backbone.csv
```

主な出力:

```text
~/260421_term1/
  official_inputs/
  official_track_i_multi/
  official_track_i_results_multi_backbone.csv
```

### 11.2 Import into local summaries

Track I CSV を local summary に統合する場合:

```bash
CUDA_VISIBLE_DEVICES=0 python 012_run_term1_foundation.py \
  --data-src ~/WILD_DATA/splits \
  --save-root ~/260421_term1 \
  --artifact-root ~/maf_ood_v51 \
  --official-track-i-csv ~/260421_term1/official_track_i_results_multi_backbone.csv \
  --eval-only
```

CSV は少なくとも次の列を持つ必要があります。

```csv
backbone,method,AUROC,AUPR-IN,AUPR-OUT,FPR95,AUTC,note
imagenet_vit,GEN,0.9000,0.9100,0.8900,0.1200,0.2200,official-style adapter on WILD
```

`track` と `display_name` は import 時に自動で付与されます。

## 12. Hades Run Shortcuts

Hades では番号付き shell wrapper を使えます。元 script の環境変数 control はそのまま残しています。

環境確認:

```bash
ssh omote@glacus.jn.sfc.keio.ac.jp
cd /home/omote/MAF-OOD-v51
source /home/omote/.venv-maf-ood-v51/bin/activate
python 002_check_environment.py --data-src /home/omote/WILD_DATA/splits
```

adaptive study を tmux で開始:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_SRC=/home/omote/WILD_DATA/splits \
SAVE_ROOT=/home/omote/maf_ood_v51 \
BACKBONES="imagenet_vit bioclip" \
SEEDS="42 123 456" \
bash 023_run_adaptive_multiseed_on_hades.sh
```

temperature ablation:

```bash
CUDA_VISIBLE_DEVICES=2 \
DATA_SRC=/home/omote/WILD_DATA/splits \
SAVE_ROOT=/home/omote/260422_temperature_ablation \
ARTIFACT_ROOT=/home/omote/maf_ood_v51 \
bash 024_run_temperature_ablation_on_hades.sh
```

script が `Started tmux session: ...` を表示したら、次で attach します。

```bash
tmux attach -t SESSION_NAME
```

GPU 利用状況は次で確認します。

```bash
nvidia-smi
```

## 13. Output Integrity and Archiving

実験完了後、source manifest を作成します。

```bash
python 090_write_repro_manifest.py --output repro_manifest.json
```

archive すべき最小セット:

```text
000_README.md
001_requirements.txt
002_check_environment.py
010_*.py to 015_*.py
020_*.sh to 025_*.sh
090_write_repro_manifest.py
maf_ood_dual_pipeline.py
maf_ood_notebook_utils.py
corrected_vim_oodd.py
dual_track_eval.py
run_*.py
01_run_term1_foundation.py
official_repro/
reports/*.py
各 output directory の *.csv と run_meta*.json
repro_manifest.json
```

実験 output の中で、少なくとも次を確認してください。

```bash
test -f ~/maf_ood_v51/adaptive_multiseed_study_summary/run_meta.json
test -f ~/maf_ood_v51/adaptive_multiseed_study_summary/local_method_summary_mean_std.csv
test -f ~/maf_ood_v51/adaptive_multiseed_study_summary/adaptive_rule_summary_mean_std.csv
```

## 14. Troubleshooting

`ModuleNotFoundError: pandas` などが出る場合:

```bash
source /home/omote/.venv-maf-ood-v51/bin/activate
pip install -r 001_requirements.txt
```

`Missing checkpoint: .../best.pt` が出る場合:

- 初回実行なのに `--eval-only` を付けている可能性があります。初回は `--eval-only` を外してください。
- artifact root が違う可能性があります。`--artifact-root` と `--save-root` を確認してください。

CUDA が見えない場合:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

GPU を固定したい場合:

```bash
CUDA_VISIBLE_DEVICES=1 python 013_run_adaptive_multiseed_study.py ...
```

feature cache を作り直したい場合:

```bash
CUDA_VISIBLE_DEVICES=0 python 013_run_adaptive_multiseed_study.py \
  --data-src ~/WILD_DATA/splits \
  --save-root ~/maf_ood_v51 \
  --backbones imagenet_vit bioclip \
  --seeds 42 123 456 \
  --force-reextract
```

`faiss` が入らない場合:

- `faiss` は official KNN adapter と一部 rival comparison で必要です。
- MAF main study だけを確認する場合は、まず `002_check_environment.py --skip-faiss` で環境確認できます。

## 15. Protocol Notes

- `MAF Mah(tied) adaptive` の alpha は ID validation 情報から推定します。test OOD label では選びません。
- `MAF Mah(tied) oracle alpha` は analysis-only row です。fair model-selection result として扱わないでください。
- `official Track I` は CSV import です。local approximation と混ぜる場合は `--include-approx-track-i` を明示し、表の note に残してください。
- `run_meta*.json` と `repro_manifest.json` が無い結果は、後から完全な実行条件を復元しづらくなります。
