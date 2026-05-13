# Dataset

The WILD_DATA split used for the paper experiments is distributed outside this repository.

```text
https://drive.google.com/drive/folders/1LCdZnAN6gWIv_Ds0bTWLipD6jjptLAgj?usp=sharing
```

After downloading, arrange the directory as:

```text
WILD_DATA/splits/
  train/id/
    buffalo/
    cheetah/
    elephant/
    giraffe/
    hippo/
  val/id/
    buffalo/
    cheetah/
    elephant/
    giraffe/
    hippo/
  test/id/
    buffalo/
    cheetah/
    elephant/
    giraffe/
    hippo/
  test/ood/
    impala/
    leopard/
    lion/
    rhino/
    wildebeest/
```

## Split Use

| Split | Use |
| --- | --- |
| `train/id` | Optional logit-head training and feature baselines such as KNN/RMD when available. |
| `val/id` | MAF class means and covariance estimation. |
| `test/id` | ID evaluation samples. |
| `test/ood` | OOD evaluation samples. |
| `val/ood` | Not used in the main MAF setting. |

The main MAF score does not use OOD validation samples and does not tune \(\alpha\) on test OOD.

## Classes

ID classes:

```text
buffalo, cheetah, elephant, giraffe, hippo
```

OOD classes:

```text
impala, leopard, lion, rhino, wildebeest
```

## Notes For Submission

- Do not commit image data or cached feature files to this repository.
- If the paper requires fully public reproducibility, include the Google Drive link and the split directory description in the supplementary material.
- License and redistribution terms of the images should be checked and stated in the paper or supplement. This repository only records the experimental directory contract.
