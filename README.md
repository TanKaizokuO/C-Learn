# CML — C Machine Learning Library

A minimal, educational machine learning library written in **pure C99**. Built from scratch across 5 iterations, covering linear algebra, activations, loss functions, optimization, and a first working ML model.

---

## Project Structure

```
cml/
├── include/
│   ├── matrix.h               # Core matrix struct + all ops
│   ├── activations.h          # relu, sigmoid
│   ├── loss.h                 # MSE, Binary Cross-Entropy
│   ├── optimizer.h            # Gradient descent
│   └── linear_regression.h   # Linear Regression model
│
├── src/
│   ├── matrix.c
│   ├── activations.c
│   ├── loss.c
│   ├── optimizer.c
│   └── linear_regression.c
│
├── examples/
│   ├── demo.c                       # Iterations 1–4 demo
│   └── train_linear_regression.c   # Iteration 5 demo
│
└── Makefile
```

---

## Build

```bash
cd cml
make          # builds both: demo and train_lr
make demo     # Iterations 1–4 only
make train_lr # Iteration 5 only
make clean
```

**Requirements:** `gcc` with C99 support, `libm`

---

## Iterations

### Iteration 1 — Linear Algebra Core
Row-major `Matrix` struct with full operations:
`create_matrix`, `free_matrix`, `add`, `subtract`, `scalar_multiply`, `matmul`, `transpose`, `dot_product`, `print_matrix`

### Iteration 2 — Tensor Utilities
Initialization and element-wise ops:
`zeros`, `ones`, `random_matrix`, `elementwise_add`, `elementwise_multiply`, `apply_function`

### Iteration 3 — Loss Functions
- **MSE** — Mean Squared Error
- **BCE** — Binary Cross-Entropy (log-clamped for numerical stability)

### Iteration 4 — Gradient Descent Optimizer
In-place weight update: `W = W - η · ∇W`

### Iteration 5 — Linear Regression
First ML model — trains `y = XW + b` using gradient descent.

```
Epoch  100 | Loss: 1.086636
Epoch  200 | Loss: 0.259073
Epoch  300 | Loss: 0.070156
Epoch  400 | Loss: 0.024221
Epoch  500 | Loss: 0.012999

Learned Weight :  2.9158   (true: 3.0)
Learned Bias   :  1.9937   (true: 2.0)
```

---

## Quick Example

```c
// Create model and data
LinearRegression model = create_linear_regression(1);
Matrix X = create_matrix(100, 1);
Matrix y = create_matrix(100, 1);
// ... fill X and y ...

// Train
train_linear_regression(&model, X, y, 500, 0.01f);

// Predict
Matrix y_pred = predict(&model, X);

// Cleanup
free_matrix(&X); free_matrix(&y); free_matrix(&y_pred);
free_linear_regression(&model);
```

---

## Design Principles

- Pure **C99** — no external dependencies
- **Row-major** matrix storage
- All functions **validate dimensions** before operating
- No memory leaks — every `create_*` has a matching `free_*`
- Zero compiler warnings under `-Wall -Wextra`
