# Baselines And Implementation Notes

This document defines exactly how each baseline score is computed in this repository. All rows use the same `test/id` and `test/ood` samples as MAF. Higher scores are always treated as more ID-like.

The goal is not to claim that these scripts reproduce every training detail of each original paper. The goal is to reproduce the **post-hoc OOD scoring rule** under the same WILD_DATA split and the same DINOv2 feature/logit cache. Therefore, each baseline below states its source paper, equation used here, input representation, default hyperparameters, and known differences from the original code path.

## Shared Notation

- \(x\): input image.
- \(h=f(x)\in\mathbb{R}^{d}\): frozen-backbone feature.
- \(z=g(h)\in\mathbb{R}^{K}\): logits from the local frozen-feature linear head.
- \(p=\operatorname{softmax}(z)\).
- \(K\): number of ID classes.
- Higher \(s(x)\) means more ID-like.

In the DINOv2 setting, \(h\) is the raw CLS feature from `dinov2_vitb14`. Logit-based baselines use a one-layer `Linear(d->K)` head trained on `train/id` frozen features and selected with `val/id`. For WILD_DATA with DINOv2-ViT-B/14, this is `Linear(768->5)`. No OOD images are used for training or hyperparameter selection in the default setting.

## Logit-Based Baselines

### MSP

Source: Hendrycks and Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks", ICLR 2017.

Score:

\[
s_{\mathrm{MSP}}(x)=\max_{c}p_c.
\]

Implementation:

```python
softmax(logits, axis=1).max(axis=1)
```

Input: `id_logits`, `ood_logits`.

Difference from source paper: the scoring formula is identical. The classifier is not the original paper's classifier; it is the local DINOv2 frozen-feature `Linear(d->K)` head.

### Entropy

Source: standard softmax uncertainty baseline commonly reported with MSP-style OOD detectors.

Score:

\[
s_{\mathrm{Entropy}}(x)=-H(p)=\sum_{c=1}^{K}p_c\log(p_c+\epsilon).
\]

Implementation:

```python
np.sum(p * np.log(p + EPS), axis=1)
```

Input: `id_logits`, `ood_logits`.

Difference from entropy-as-uncertainty convention: entropy itself is larger for uncertain samples, so this repository uses **negative entropy** to preserve the "higher is ID-like" convention.

### Energy

Source: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.

Score used here:

\[
s_{\mathrm{Energy}}(x;T)=T\log\sum_{c=1}^{K}\exp(z_c/T).
\]

Default:

\[
T=1.
\]

Implementation:

```python
temperature * logsumexp(logits / temperature, axis=1)
```

Input: `id_logits`, `ood_logits`.

Difference from source paper: this repository uses the post-hoc energy score only. It does not perform energy-bounded fine-tuning or outlier exposure.

### MaxLogit

Source: Hendrycks et al., "Scaling Out-of-Distribution Detection for Real-World Settings", ICML 2022.

Score:

\[
s_{\mathrm{MaxLogit}}(x)=\max_{c}z_c.
\]

Implementation:

```python
logits.max(axis=1)
```

Input: `id_logits`, `ood_logits`.

Difference from source paper: the score formula is identical, but logits come from the local frozen-feature `Linear(d->K)` head rather than the original large-scale trained models.

## Feature-Based Baselines

### KNN

Source: Sun et al., "Out-of-Distribution Detection with Deep Nearest Neighbors", ICML 2022.

Feature normalization:

\[
\bar{h}=\frac{h}{\lVert h\rVert_2+\epsilon}.
\]

Let \(\mathcal{B}\) be the feature bank. The score is:

\[
s_{\mathrm{KNN}}(x)=-d_k(\bar{h}(x),\mathcal{B}),
\]

where \(d_k\) is the Euclidean distance to the \(k\)-th nearest neighbor.

Default:

\[
k=50.
\]

Feature bank:

- `train/id` if `tr_features` and `tr_labels` exist in `analysis_v3.npz`.
- otherwise `val/id`.

Implementation:

```python
NearestNeighbors(n_neighbors=k, metric="euclidean")
```

Difference from common official implementations: this repository uses scikit-learn exact nearest-neighbor search, not FAISS. For the same normalized feature bank and exact Euclidean search, the score definition is equivalent; runtime may differ.

### Mah-MinDist

Source: Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks", NeurIPS 2018.

Class statistics are estimated from `val/id` in this repository:

\[
\mu_c=\frac{1}{N_c}\sum_{i:y_i=c}h_i.
\]

With tied covariance \(\Sigma\), the quadratic Mahalanobis distance is:

\[
q_c(x)=(h(x)-\mu_c)^\top\Sigma^{-1}(h(x)-\mu_c).
\]

The repository uses the square-rooted distance for consistency with MAF:

\[
d_c(x)=\sqrt{\max(q_c(x),0)}.
\]

Score:

\[
s_{\mathrm{MahMin}}(x)=-\min_c d_c(x).
\]

Difference from formulations using \(-\min_c q_c(x)\): square root is monotonic, so ranking-based metrics such as AUROC are unchanged relative to the quadratic score, but raw score scale differs.

### Mahalanobis++

Source: Maximilian Mueller and Matthias Hein, "Mahalanobis++: Improving OOD Detection via Feature Normalization", ICML 2025.

This repository implements Mahalanobis++ as a separate baseline, not as MAF with L2 normalization. It follows the public reference implementation:

\[
\bar{h}=\frac{h}{\lVert h\rVert_2+\epsilon}.
\]

Class means are estimated from `train/id` normalized features:

\[
\mu_c=\frac{1}{N_c}\sum_{i:y_i=c}\bar{h}_i.
\]

The shared covariance is estimated with `sklearn.covariance.EmpiricalCovariance(assume_centered=True)` on class-centered train features:

\[
\Sigma
=
\operatorname{Cov}\left(
\left\{\bar{h}_i-\mu_{y_i}\right\}_{i=1}^{N}
\right).
\]

The score is the negative nearest-class quadratic Mahalanobis distance:

\[
s_{\mathrm{Mahalanobis++}}(x)
=
-\min_c
(\bar{h}(x)-\mu_c)^\top
\Sigma^{-1}
(\bar{h}(x)-\mu_c).
\]

Implementation:

```python
feature_train = l2_normalize(feature_train)
feature_query = l2_normalize(feature_query)
means = class_means(feature_train, train_labels)
centered = feature_train - means[train_labels]
precision = EmpiricalCovariance(assume_centered=True).fit(centered).precision_
score = -min_quadratic_mahalanobis(feature_query, means, precision)
```

Input requirement: exact Mahalanobis++ requires `tr_features` and `tr_labels` in `analysis_v3.npz`. If the cached artifact has no train split, this row is not emitted by `scripts/run_cached_features.py`.

Difference from the source paper: the scoring formula and covariance estimator match the reference code path, but the backbone/features are the WILD_DATA cached features used in this repository, not the ImageNet-scale model zoo evaluated in the original paper.

### RMD

Source: Ren et al., "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection", UDL 2021 / arXiv 2021.

Class Mahalanobis distance:

\[
q_c(x)=(h(x)-\mu_c)^\top\Sigma^{-1}(h(x)-\mu_c).
\]

Global Mahalanobis distance:

\[
q_0(x)=(h(x)-\mu_0)^\top\Sigma_0^{-1}(h(x)-\mu_0).
\]

Score:

\[
s_{\mathrm{RMD}}(x)=-\min_c q_c(x)+q_0(x).
\]

Statistics:

- \(\mu_c,\Sigma\): class means and tied class covariance from the feature bank.
- \(\mu_0,\Sigma_0\): global mean and global covariance from the same feature bank.
- Covariances use Ledoit-Wolf shrinkage plus a small ridge for numerical stability.

Feature bank:

- `train/id` if present.
- otherwise `val/id`.

Difference from the source paper: the RMD formula is the same, but the feature extractor is DINOv2 and the data split is WILD_DATA. The original paper did not use this wildlife split.

## Reporting Language

Use this wording in the paper or supplement:

> Baseline scores were recomputed under the same WILD_DATA split and the same frozen DINOv2 feature/logit cache. For one-line post-hoc scores such as MSP, Energy, and MaxLogit, we use the standard equations from the cited papers. For feature-space baselines such as KNN and RMD, we match the published score definitions while using the same feature bank as MAF for a controlled comparison.

Avoid the vague phrase "local reimplementation" by itself. If a table row is not produced by an official repository, state the exact score equation and feature/logit source.
