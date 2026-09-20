# CUDA-Optimised-LR

A small benchmarking exercise comparing three logistic regression implementations: a
Numba/CUDA GPU version, a plain NumPy CPU version, and scikit-learn's
`LogisticRegression`. The two implementations being compared are **not original to this
repository** — they come from [NechbaMohammed/SwiftLogisticReg](https://github.com/NechbaMohammed/SwiftLogisticReg)
(MIT). The original work here is `comparison.ipynb`, the notebook that times them
against each other on two datasets.

## What's here

| File | Description |
|------|-------------|
| `logistic_gpu.py` | `LogisticRegressionGPU` — batch gradient descent with five `@cuda.jit` kernels (`vector_matrix_mul`, `matrix_col_sum`, `sigmoid`, `substract`, `norm2`). Copied from SwiftLogisticReg. |
| `logistic_cpu.py` | `LogisticRegression` — the same algorithm in NumPy. Copied from SwiftLogisticReg. |
| `__init__.py` | Re-exports both classes. Copied from SwiftLogisticReg. |
| `comparison.ipynb` | The benchmark notebook. Original to this repo. |

## What the numbers actually showed

The notebook's saved outputs are the only measurements this repo contains. They do not
show a GPU speedup:

- **GPU implementation**, HiggsML `training.csv` (250k rows, 30 features): **343.7 s**, F1 0.523.
- **scikit-learn** `LogisticRegression`, credit-card sample: **1.19 s**, F1 0.685 (with a
  `ConvergenceWarning` — lbfgs hit its iteration limit).
- **CPU implementation**: both cells raise `RuntimeWarning: overflow encountered in exp`
  and no completed timing was recorded. `logistic_cpu.fit` loops on `while True` with no
  iteration cap, so on unscaled data it does not reliably terminate.

The slowness is explainable from the code rather than from the GPU: `fit` copies the full
feature matrix from host to device twice per iteration (`cuda.to_device(X.copy())`),
because `vector_matrix_mul` writes its result in place over its input. For 1000 iterations
that transfer cost dominates everything else. Neither implementation scales its inputs,
which is what produces the `exp` overflow.

Treat this as a negative result: a hand-written kernel-per-operation design, with no
fusion and no attention to transfers, loses badly to a tuned BLAS-backed CPU library.

## Running it

```bash
pip install numpy numba scikit-learn pandas
```

The notebook needs an NVIDIA GPU with a working CUDA toolkit for the GPU cells, and it
imports from the installed `SwiftLogisticReg` package rather than the local files:

```bash
pip install SwiftLogisticReg
```

**The notebook will not run as committed.** It reads `creditcardsample.csv` and
`training.csv`, neither of which is in the repository. The first is a sample of the
[Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud);
the second is the training set from the
[HiggsML challenge](https://www.kaggle.com/c/higgs-boson). You would need to supply both.

## Attribution

`logistic_cpu.py`, `logistic_gpu.py` and `__init__.py` are taken from
**[NechbaMohammed/SwiftLogisticReg](https://github.com/NechbaMohammed/SwiftLogisticReg)**,
by Mohammed Nechba, Mohamed Mouhajir and Yassine Sedjari, used under the MIT License.
The approach is described in their paper, *High Performance Computing Applied to Logistic
Regression: A CPU and GPU Implementation Comparison* ([arXiv:2308.10037](https://arxiv.org/abs/2308.10037)).
