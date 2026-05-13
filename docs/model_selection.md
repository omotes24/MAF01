# Model Selection And Exploratory Decisions

This project contains both exploratory analysis and a fixed final scoring configuration. For paper submission, these must be separated clearly.

## Final Fixed Configuration

The final MAF configuration implemented by default is:

| Item | Value |
| --- | --- |
| Backbone | `dinov2_vitb14` |
| Feature | raw CLS feature |
| Feature normalization | none |
| Distance | Mahalanobis |
| Covariance | tied covariance |
| Covariance estimator | Ledoit-Wolf |
| Distance-to-distribution transform | softmax over negative distances |
| Temperature | \(\tau=1.0\) |
| Fusion | geometric fusion |
| Fusion weight | \(\alpha=0.50\) |
| OOD validation for alpha | not used in the default script |

The default command in `scripts/run_cached_features.py` evaluates this fixed configuration without selecting \(\alpha\) on `test/ood`.

## Exploratory Decisions

The following choices were made after exploratory experiments on the project data and should be disclosed as such:

1. **Using Mahalanobis feature-space geometry instead of only logit confidence.**  
   MSP, Entropy, Energy, and MaxLogit were used as logit-score baselines.

2. **Using the two MAF components \(C(x)\) and \(G(x)\).**  
   We compared `s_conf only`, `s_cons only`, product, arithmetic fusion, and geometric fusion.

3. **Using \(\alpha=0.50\) as the final fixed fusion weight.**  
   Earlier analysis included other alpha values and oracle-style sweeps. The submitted main setting should be described as a fixed post-exploration setting, not as an independently validated hyperparameter.

4. **Using tied covariance instead of class-wise covariance.**  
   Class-wise covariance was evaluated as an ablation and was less stable on this dataset.

5. **Using Ledoit-Wolf covariance estimation.**  
   Empirical covariance was evaluated as an ablation.

6. **Using raw DINOv2 features rather than L2-normalized features.**  
   L2 variants were explored and should be reported as ablations if included.

7. **Using Mahalanobis distance rather than Euclidean distance.**  
   Euclidean fusion was evaluated to test whether covariance structure matters.

8. **Excluding complex exploratory variants from the final method.**  
   Rank residual, oracle alpha, and combined/oracle variants are analysis-only unless explicitly labeled.

## How To State This In The Paper

Recommended wording:

> We first conducted exploratory ablations on WILD_DATA to identify a stable MAF configuration. Based on these ablations, we fixed the final method to raw DINOv2 features, tied Ledoit-Wolf covariance, \(\tau=1.0\), and \(\alpha=0.50\). The final table reports this fixed configuration across three seeds without using OOD validation samples for alpha selection. Because the design was selected through exploratory analysis on WILD_DATA, we treat the results as evidence for this benchmark rather than as a claim of dataset-independent optimality.

If only WILD_DATA is used, avoid wording such as:

```text
MAF is universally superior to prior methods.
```

Use a narrower claim:

```text
Under the WILD_DATA Near-OOD wildlife benchmark, the fixed MAF configuration improves AUROC and FPR95 over the evaluated baselines.
```

## Oracle Rows

Any row that selects \(\alpha\), score variant, feature normalization, or covariance type using `test/ood` labels must be labeled as:

```text
oracle / analysis-only / not deployable
```

Such rows should not be mixed into the main fair-comparison ranking.

## Remaining Limitation

The current repository supports exact reproduction of the WILD_DATA experiments. It does not by itself prove that the selected configuration is optimal for other wildlife datasets. To answer the criticism "Was this only favorable for the current dataset?", the strongest additional evidence would be:

- a second independent wildlife OOD dataset,
- a validation-only selection protocol,
- or a leave-one-OOD-species-out protocol where the held-out species is not used during design selection.
