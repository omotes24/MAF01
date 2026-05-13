# Method Notes

## MAF Score

MAF is a fusion of two quantities derived from a class-wise distance distribution.

- \(C(x)\): nearest-class Mahalanobis confidence, defined as the maximum probability after applying softmax to negative class distances.
- \(G(x)\): distribution sharpness, defined as the complement of normalized entropy.

\[
d_c(x)=\sqrt{\max((f(x)-\mu_c)^\top\Sigma^{-1}(f(x)-\mu_c),0)}
\]

\[
p_c(x;\tau)=\frac{\exp(-d_c(x)/\tau)}{\sum_j\exp(-d_j(x)/\tau)}
\]

\[
C(x)=\max_c p_c(x;\tau)
\]

\[
G(x)=1-\frac{-\sum_c p_c(x;\tau)\log(p_c(x;\tau)+\epsilon)}{\log K}
\]

\[
S_{\mathrm{MAF}}(x)=
\operatorname{clip}(C(x),\epsilon,1)^{\alpha}
\times
\operatorname{clip}(G(x),\epsilon,1)^{1-\alpha}
\]

The main paper setting uses \(\alpha=0.50\), \(\tau=1.0\), raw DINOv2 features, tied covariance, and Ledoit-Wolf covariance estimation.

## Covariance Variants

- `mah_t`: tied covariance. One covariance matrix is shared across classes. In this implementation it is the mean of class covariance estimates.
- `mah_c`: class-wise covariance. Each class has its own covariance matrix.
- `euc`: Euclidean distance. No covariance inverse is used.

## Feature Variants

- `raw`: the frozen backbone feature is used as extracted.
- `l2`: features are row-wise L2-normalized before estimating class means/covariance and before scoring.

## Oracle Setting

The main scripts do not tune \(\alpha\) on test OOD. Oracle alpha sweeps can be implemented for analysis, but should be labeled explicitly as `oracle_ood` in paper tables.
