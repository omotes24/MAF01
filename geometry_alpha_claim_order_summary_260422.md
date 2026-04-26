# Geometry Alpha Claim-Order Summary (260422)

## Core Claim

本研究の主張は次の 1 文に固定する。

> OOD 用に `alpha` を探索するのではなく、ID reliability から決まる backbone-aware な `alpha_0` と、Mahalanobis 局所 margin から決まる `alpha(x)` によって融合比を幾何学的に定める。

以下の結果整理は、この 1 文のどこを支えるかで並べる。

## 0. Reproducibility Base

まず seed 生値から、`±0.0000` が集計バグではないことを確認できる。

- `bioclip` proposal (`MAF Mah(tied) adaptive`) の 3 seed 振れ幅:
  - `AUROC` range = `0.000046`
  - `FPR95` range = `0.000300`
  - `AUPR-OUT` range = `0.000067`
- `imagenet_vit` proposal の 3 seed 振れ幅:
  - `AUROC` range = `0.000009`
  - `FPR95` range = `0.000500`
  - `AUPR-OUT` range = `0.000026`

したがって、proposal 行の `±0.0000` は丸め込みであり、実際には極小だが非ゼロの seed 差がある。

参照:
- `/Users/k.omote/Downloads/maf_geometry_alpha_260422/geometry_alpha_study_summary/seed_raw_method_metrics.csv`

## 1. Global Evidence: Backbone-Aware `alpha_0`

ID validation から推定した `alpha_0` は backbone ごとに安定しており、`r_conf > r_cons` の方向を保っている。

- `bioclip`
  - `alpha_0 = 0.5155±0.0007`
  - `r_conf = 0.8629±0.0023`
  - `r_cons = 0.8111±0.0043`
- `imagenet_vit`
  - `alpha_0 = 0.5188±0.0008`
  - `r_conf = 0.8948±0.0065`
  - `r_cons = 0.8300±0.0084`

OOD test を見て選んだ固定 `alpha` の最良値はかなり大きい。

- `bioclip`: fixed best `alpha = 0.7200`
- `imagenet_vit`: fixed best `alpha = 0.9300`

ただし、`alpha_0` と fixed best `alpha` は**方向の整合**を保っている。

- `preference_match = 1.000` for both backbones
- つまり、両 backbone で「`conf` 側へ寄せる」傾向自体は一致している
- 一方で magnitude gap は大きい
  - `bioclip`: `alpha_0 - alpha*_fixed = -0.2045±0.0007`
  - `imagenet_vit`: `alpha_0 - alpha*_fixed = -0.4112±0.0008`

この結果から本文で言うべきことは次の通り。

> ID reliability から決めた `alpha_0` は、OOD tuning された最良固定 `alpha` を数値的に再現するものではない。しかし backbone ごとの融合の向きは正しく捉えており、固定共通 `alpha` より自然な prior を与える。

参照:
- `/Users/k.omote/Downloads/maf_geometry_alpha_260422/geometry_alpha_study_summary/adaptive_rule_summary_mean_std.csv`
- `/Users/k.omote/Downloads/maf_geometry_alpha_260422/geometry_alpha_study_summary/alpha_alignment_summary_mean_std.csv`

## 2. Local Evidence: Margin-Adaptive `alpha(x)`

`margin` と `alpha(x)` の関係は非常に強く、局所幾何に応じて gate が動いていることは確認できる。

- `bioclip`
  - `test_id`: `margin_mean = 0.024803`, `alpha_mean = 0.517445`, `corr(m, alpha) = 0.954828`
  - `test_ood`: `margin_mean = 0.008783`, `alpha_mean = 0.299564`, `corr(m, alpha) = 0.992910`
- `imagenet_vit`
  - `test_id`: `margin_mean = 0.031939`, `alpha_mean = 0.540942`, `corr(m, alpha) = 0.893037`
  - `test_ood`: `margin_mean = 0.006744`, `alpha_mean = 0.276565`, `corr(m, alpha) = 0.988424`

ここで言えることは明確。

- ID 側では `margin` が大きく、`alpha(x)` も大きい
- OOD 側では `margin` が小さく、`alpha(x)` も小さい
- `alpha(x)` は learned black-box gate ではなく、少なくとも実装どおりに局所幾何量に単調応答している

本文の 1 文はこう置ける。

> `alpha(x)` は局所 margin に対して単調に変化し、ID 側では高く、OOD 側では低くなるため、融合比がサンプルごとの局所幾何に従って決まっていることが確認できる。

参照:
- `/Users/k.omote/Downloads/maf_geometry_alpha_260422/geometry_alpha_study_summary/sample_level_geometry_all_seeds.csv`

## 3. Subgroup Evidence: Margin Bin での振る舞い

margin 5 分位 (`Q1` low margin, `Q5` high margin) の結果は、**backbone と bin によって有利な成分が変わる**ことを示している。ただし、現行の adaptive rule が常にその切替を最良に吸収できているわけではない。

### bioclip

- `Q1` では `conf_only` が明確に不利
  - `adaptive_full`: `AUROC 0.6681`, `FPR95 0.8634`, `AUPR-OUT 0.8165`
  - `conf_only`: `AUROC 0.6526`, `FPR95 0.8858`, `AUPR-OUT 0.8074`
  - `cons_only`: `AUROC 0.6690`, `FPR95 0.8651`, `AUPR-OUT 0.8172`
- しかし `Q3-Q5` では `conf_only` または `cons_only` が adaptive を上回る
  - `Q5` では `cons_only` が最良
  - `adaptive_full` は切替を吸収し切れていない

### imagenet_vit

- `Q1` low margin では `adaptive_full` と `cons_only` がほぼ同等で、`conf_only` は劣る
- `Q2-Q5` では `conf_only` あるいは `cons_only` が adaptive を上回る bin が多い
- 特に `Q3-Q5` では、現行の margin gate だけでは bin 内最良法を常に再現できていない

この section で無理に言ってはいけないこと:

- 「adaptive が margin bin ごとに最強だった」
- 「小 margin では必ず cons, 大 margin では必ず conf」

今回の結果から言える、より正確な文は次。

> margin-stratified subgroup では有利な融合成分が backbone と subgroup に依存して変化し、固定単一則では説明しにくい切替構造が存在した。一方で、現行の adaptive rule はその切替を部分的に捉えるが、全 subgroup で最良になるわけではなかった。

参照:
- `/Users/k.omote/Downloads/maf_geometry_alpha_260422/geometry_alpha_study_summary/subgroup_margin_metrics_summary_mean_std.csv`

## 4. Claim-Oriented Ablation

この表は「部品比較」ではなく、主張比較として読む。

### imagenet_vit

- `Q1 global prior needed`
  - `adaptive - margin_only_centered`
  - `delta_AUROC = +0.0001±0.0000`
  - `delta_FPR95 = -0.0012±0.0003`
- `Q2 local adaptation needed`
  - `adaptive - backbone_only_alpha0`
  - `delta_AUROC = +0.0007±0.0000`
  - `delta_FPR95 = -0.0027±0.0012`
- `Q3 plain product suffices`
  - `adaptive - product`
  - `delta_AUROC = +0.0009±0.0000`
  - `delta_FPR95 = -0.0023±0.0015`

ここでは `adaptive_full` が小さいが一貫した改善を出している。特に FPR95 改善が読みやすい。

### bioclip

- `Q1 global prior needed`
  - `adaptive - margin_only_centered`
  - `delta_AUROC = -0.0002±0.0000`
  - `delta_FPR95 = +0.0002±0.0002`
- `Q2 local adaptation needed`
  - `adaptive - backbone_only_alpha0`
  - `delta_AUROC = -0.0025±0.0000`
  - `delta_FPR95 = +0.0081±0.0005`
- `Q3 plain product suffices`
  - `adaptive - product`
  - `delta_AUROC = -0.0025±0.0000`
  - `delta_FPR95 = +0.0085±0.0002`

`bioclip` では local adaptation が利益に転じていない。本文ではここを隠さず、

> local adaptation の利益は backbone 依存であり、表現空間によっては backbone-aware な global prior がほぼ十分である。

と書くのが自然。

参照:
- `/Users/k.omote/Downloads/maf_geometry_alpha_260422/geometry_alpha_study_summary/claim_ablation_comparisons_summary_mean_std.csv`

## 5. Overall Performance and Track I Integration

### Local mean±std

- `bioclip`
  - Proposal `MAF Mah(tied) adaptive`: `AUROC 0.8000±0.0000`, `FPR95 0.7365±0.0001`, `AUPR-OUT 0.7696±0.0000`
  - Best local baselines:
    - `Energy`: `AUROC 0.7980±0.0096`, `FPR95 0.7256±0.0065`, `AUPR-OUT 0.7761±0.0069`
    - `MaxLogit`: `AUROC 0.7980±0.0096`, `FPR95 0.7268±0.0073`, `AUPR-OUT 0.7778±0.0071`
- `imagenet_vit`
  - Proposal `MAF Mah(tied) adaptive`: `AUROC 0.8435±0.0000`, `FPR95 0.6753±0.0002`, `AUPR-OUT 0.8130±0.0000`
  - Strongest local baseline:
    - `KNN`: `AUROC 0.8272±0.0104`, `FPR95 0.7200±0.0210`, `AUPR-OUT 0.7967±0.0133`

### Track I integration

`integrated_summary_with_track_i.csv` には

- `local_mean_std`: `22` rows
- `official_track_i`: `3` rows

が入っている。現時点の official import は `imagenet_vit` の `KNN [I]`, `ViM [I]`, `GEN [I]` の 3 行であり、`bioclip` の official Track I 行はまだ入っていない。

したがって、本文では次のように書くのが正確。

> Track I 統合は実装済みだが、現段階で official import が揃っているのは `imagenet_vit` の 3 ベースラインのみであり、backbone 横断の Track I 比較はまだ不完全である。

参照:
- `/Users/k.omote/Downloads/maf_geometry_alpha_260422/geometry_alpha_study_summary/local_method_summary_mean_std.csv`
- `/Users/k.omote/Downloads/maf_geometry_alpha_260422/geometry_alpha_study_summary/integrated_summary_with_track_i.csv`

## 6. Failure Analysis: Why bioclip Loses

`bioclip` では `adaptive_full` が `backbone_only_alpha0` と `product` に負ける。その失敗例を集計すると、主に OOD false accept が増えている。

- vs `backbone_only_alpha0`
  - `false_accept_ood = 170.6667±2.6247`
  - `margin_mean = 0.0084±0.0000`
  - `alpha_mean = 0.2859±0.0003`
  - `score_gap = -0.0265±0.0001`
- vs `product`
  - `false_accept_ood = 176.0000±1.6330`
  - `margin_mean = 0.0083±0.0000`
  - `alpha_mean = 0.2858±0.0006`
  - `score_gap = +0.0203±0.0000`

ここから読めること:

- 負け方の中心は low-margin OOD
- その領域では `alpha(x)` が低くなり、局所適応が `cons` 側へ寄る
- しかし `bioclip` では、その low-margin OOD に対して単純 global baseline の方が結果として強い

本文では、

> `bioclip` では low-margin OOD 群において local adaptation が過剰に働き、backbone-aware global prior を上回る改善に繋がらなかった。これは提案原理の破綻ではなく、有効条件が backbone 依存であることを示す。

と書ける。

参照:
- `/Users/k.omote/Downloads/maf_geometry_alpha_260422/geometry_alpha_study_summary/bioclip_failure_summary_mean_std.csv`
- `/Users/k.omote/Downloads/maf_geometry_alpha_260422/geometry_alpha_study_summary/bioclip_failure_cases.csv`

## Paper Narrative to Use

論文の結果 section は次の順で置く。

1. Reproducibility: seed 生値と mean±std
2. Global principle: `alpha_0` is backbone-aware
3. Local principle: `alpha(x)` tracks margin
4. Subgroup evidence: margin-stratified behavior changes by subgroup/backbone
5. Claim-oriented ablation: global prior / local adaptation / plain product
6. Overall benchmark result: local + partial Track I integration
7. Limitation: `bioclip` failure analysis

この順であれば、

- `imagenet_vit` の強さは実用上の positive evidence
- `bioclip` の弱さは principle の適用条件

として同じ物語に収まる。

## What Not to Claim

今回の結果からは、次はまだ言い過ぎになる。

- adaptive が全 subgroup で最良
- margin だけで conf/cons 切替が完全に説明できた
- backbone-aware + local margin rule が universal optimum

今言える最も強い結論は次。

> `alpha` を OOD tuning で選ぶのではなく、ID reliability と局所 margin から決めるという幾何学的決定原理は、global / local の両レベルで観測できる。ただし、その adaptive rule が常に最強になるわけではなく、特に `bioclip` では global prior が十分に強い条件が残る。
