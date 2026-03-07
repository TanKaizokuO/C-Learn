# CML — C Machine Learning Library

A minimal, educational machine learning library written in **pure C99**. Built from scratch, covering linear algebra, activations, loss functions, optimization, and full ML models up to neural networks.

---

## Project Structure

```
C-Learn/
├── include/
│   ├── matrix.h                 # Core matrix struct + all ops
│   ├── activations.h            # relu, sigmoid
│   ├── loss.h                   # MSE, Binary Cross-Entropy
│   ├── optimizer.h              # Gradient descent
│   ├── linear_regression.h      # Iteration 5
│   ├── logistic_regression.h    # Iteration 6
│   ├── dense_layer.h            # Iteration 7
│   └── neural_network.h         # Iteration 7
│
├── src/
│   ├── matrix.c
│   ├── activations.c
│   ├── loss.c
│   ├── optimizer.c
│   ├── linear_regression.c
│   ├── logistic_regression.c
│   ├── dense_layer.c
│   └── neural_network.c
│
├── examples/
│   ├── demo.c                        # Iterations 1–4
│   ├── train_linear_regression.c     # Iteration 5
│   ├── train_logistic.c              # Iteration 6
│   └── neural_network_demo.c         # Iteration 7
│
└── Makefile
```

---

## Build

```bash
make                # build all binaries
make demo           # Iterations 1–4
make train_lr       # Iteration 5 — Linear Regression
make train_logistic # Iteration 6 — Logistic Regression
make nn_demo        # Iteration 7 — Neural Network
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
Trains `y = XW + b` via gradient descent (MSE loss).

```
Epoch  500 | Loss: 0.012999
Learned Weight :  2.9158   (true: 3.0)
Learned Bias   :  1.9937   (true: 2.0)
```

### Iteration 6 — Logistic Regression

Binary classification model:

```
y = sigmoid(X·W + b)
```

- Sigmoid activation squashes output to a probability in `(0, 1)`
- Binary Cross-Entropy loss:  `L = -(1/n) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]`
- Gradient: `dW = (1/n) · Xᵀ · (ŷ - y)`

```c
LogisticRegression model = create_logistic_regression(2);
train_logistic_regression(&model, X, y, 1000, 0.1f);
Matrix probs = predict_logistic(&model, X);
// Accuracy: threshold at 0.5
```

Sample output:
```
Epoch  100 | Loss: 0.521803
Epoch  500 | Loss: 0.153291
Epoch 1000 | Loss: 0.103842

Accuracy : 196 / 200 = 98.0%
```

### Iteration 7 — Neural Network Core

Minimal two-layer feedforward network built from a `DenseLayer` abstraction:

```
Input → Dense → ReLU → Dense → Sigmoid → Output
```

**Dense Layer:**
```c
typedef struct {
    Matrix weights;   // (input_size × output_size), random init
    Matrix bias;      // (1 × output_size), zero init — broadcast
} DenseLayer;
```

**Network:**
```c
NeuralNetwork net = create_network(10, 16, 1);
Matrix output = forward_network(&net, X);
free_network(&net);
```

Forward pass internals:
```c
Z1 = forward_dense(&net.layer1, X);   // linear
A1 = apply_activation(Z1, relu);       // non-linearity
Z2 = forward_dense(&net.layer2, A1);  // linear
A2 = apply_activation(Z2, sigmoid);   // output probability
```

> **Note:** Backpropagation training is planned for a future iteration.

---

## Design Principles

- Pure **C99** — no external dependencies
- **Row-major** matrix storage
- All functions **validate dimensions** before operating
- No memory leaks — every `create_*` has a matching `free_*`
- Zero compiler warnings under `-Wall -Wextra`
