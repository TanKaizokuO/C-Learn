# CML — A Machine Learning Library in Pure C

> A lightweight, educational machine learning framework built from scratch in **pure C99** — no external ML dependencies, no black boxes.

CML implements the full stack of machine learning fundamentals — from raw matrix arithmetic up to feedforward neural networks — entirely in C. Every gradient, every weight update, and every forward pass is written out explicitly, making it a hands-on reference for understanding how ML systems work at the systems level.

---

## Overview

Most ML frameworks abstract away the internals behind high-level APIs. CML does the opposite — it exposes every detail by design.

- **Educational by intent** — built to be read alongside theory, not just executed
- **Pure C99** — no Python, no BLAS, no external ML libraries
- **Zero dependencies** — only the C standard library and `libm`
- **Iterative architecture** — each layer of the stack builds directly on the previous one
- **Memory-transparent** — explicit allocation and deallocation throughout; no hidden heap usage

The library grows the ML stack from the ground up across 7 self-contained iterations, starting from a matrix struct and ending at a two-layer neural network.

---

## Features

| Component | Description |
|---|---|
| **Matrix Core** | Row-major matrix struct with full linear algebra ops |
| **Tensor Utilities** | `zeros`, `ones`, `random_matrix`, element-wise ops, `apply_function` |
| **Activation Functions** | ReLU, Sigmoid — composable with any matrix operation |
| **Loss Functions** | Mean Squared Error, Binary Cross-Entropy (numerically stable) |
| **Optimizer** | Gradient Descent — in-place parameter update |
| **Linear Regression** | `y = XW + b` — trained with MSE + gradient descent |
| **Logistic Regression** | `y = sigmoid(XW + b)` — binary classification with BCE loss |
| **Dense Layer** | Fully-connected layer with weight + bias, forward pass |
| **Neural Network** | Two-layer feedforward net: `Input → Dense → ReLU → Dense → Sigmoid` |

---

## Architecture

All models are built on top of the core `Matrix` system. The general pipeline for any model is:

```
Input (Matrix X)
    ↓
Linear Transformation  →  Z = XW + b
    ↓
Activation Function    →  A = relu(Z)  or  sigmoid(Z)
    ↓
Loss Computation       →  L = MSE(y, ŷ)  or  BCE(y, ŷ)
    ↓
Gradient Computation   →  dW = ∂L/∂W,  db = ∂L/∂b
    ↓
Parameter Update       →  W = W - η · dW
```

For the neural network, this pipeline is stacked:

```
Input → Dense → ReLU → Dense → Sigmoid → Output
```

---

## Core Data Structure

The entire library is built around a single flat matrix type:

```c
typedef struct {
    int    rows;
    int    cols;
    float *data;   /* heap-allocated, row-major: element (i,j) at data[i*cols+j] */
} Matrix;
```

**Row-major storage** means element `(i, j)` lives at `data[i * cols + j]`. All operations respect this layout. Every matrix created with `create_matrix()` must be released with `free_matrix()`.

```c
Matrix m = create_matrix(3, 4);   /* allocate a 3×4 matrix */
/* ... use m ... */
free_matrix(&m);                  /* release; sets m.data = NULL */
```

---

## Implemented Models

### Linear Regression

Models a continuous output using a linear relationship:

```
y = XW + b
```

- **Loss:** Mean Squared Error → `MSE = (1/n) Σ (ŷ - y)²`
- **Gradients:** `dW = (2/n) Xᵀ(ŷ - y)` · `db = (2/n) Σ(ŷ - y)`
- **Update:** Standard gradient descent

```c
LinearRegression model = create_linear_regression(n_features);
train_linear_regression(&model, X, y, epochs, learning_rate);
Matrix y_pred = predict(&model, X);
free_matrix(&y_pred);
free_linear_regression(&model);
```

Sample result on `y = 3x + 2`:

```
Epoch  100 | Loss: 1.086636
Epoch  500 | Loss: 0.012999

Learned Weight :  2.9158   (true: 3.0)
Learned Bias   :  1.9937   (true: 2.0)
```

---

### Logistic Regression

Binary classification — maps inputs to a probability in `(0, 1)`:

```
y = sigmoid(XW + b)
```

- **Activation:** Sigmoid  →  `σ(x) = 1 / (1 + e⁻ˣ)`
- **Loss:** Binary Cross-Entropy  →  `L = -(1/n) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]`
- **Stability:** predictions clamped to `[ε, 1-ε]` to prevent `log(0)`
- **Gradients:** `dW = (1/n) Xᵀ(ŷ - y)` — exact gradient of BCE with sigmoid output

```c
LogisticRegression model = create_logistic_regression(n_features);
train_logistic_regression(&model, X, y, epochs, learning_rate);
Matrix probs = predict_logistic(&model, X);
/* threshold at 0.5 for hard classification */
free_matrix(&probs);
free_logistic_regression(&model);
```

Sample result on a linearly separable 2D dataset:

```
Epoch  100 | Loss: 0.372483
Epoch 1000 | Loss: 0.180042

Accuracy : 199 / 200 = 99.5%
```

---

### Neural Network Core

A minimal two-layer feedforward network built from a `DenseLayer` abstraction:

```
Input → [ Dense + ReLU ] → [ Dense + Sigmoid ] → Output
```

**Dense Layer:**

```c
typedef struct {
    Matrix weights;   /* (input_size × output_size) — random init */
    Matrix bias;      /* (1 × output_size)          — zero init, broadcast */
} DenseLayer;
```

Forward pass of a single dense layer:

```
Z = X · W + b
```

Bias is broadcast across all sample rows — `Z[i][j] += bias[j]` for every sample `i`.

**Network:**

```c
NeuralNetwork net = create_network(input_size, hidden_size, output_size);
Matrix output = forward_network(&net, X);   /* shape: (n_samples × output_size) */
free_matrix(&output);
free_network(&net);
```

Forward pass internals:

```c
Z1 = forward_dense(&net.layer1, X);    /* linear:  Input → Hidden */
A1 = apply_activation(Z1, relu);        /* non-linearity           */
Z2 = forward_dense(&net.layer2, A1);   /* linear:  Hidden → Output */
A2 = apply_activation(Z2, sigmoid);    /* output probability       */
```

> **Roadmap:** Backpropagation and training loop for the neural network are planned as a future iteration.

---

## Project Structure

```
C-Learn/
│
├── include/
│   ├── matrix.h                 ← Core matrix struct and operations
│   ├── activations.h            ← ReLU, Sigmoid
│   ├── loss.h                   ← MSE, Binary Cross-Entropy
│   ├── optimizer.h              ← Gradient Descent
│   ├── linear_regression.h      ← Iteration 5
│   ├── logistic_regression.h    ← Iteration 6
│   ├── dense_layer.h            ← Iteration 7
│   └── neural_network.h         ← Iteration 7
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
│   ├── demo.c                        ← Iterations 1–4 showcase
│   ├── train_linear_regression.c     ← Linear Regression demo
│   ├── train_logistic.c              ← Logistic Regression demo
│   └── neural_network_demo.c         ← Neural Network forward pass demo
│
├── Makefile
└── README.md
```

---

## Build Instructions

The project uses a standard `Makefile`. A C99-compatible compiler and `libm` are the only requirements.

```bash
# Build everything
make

# Build individual targets
make demo           # Iterations 1–4 showcase
make train_lr       # Linear Regression
make train_logistic # Logistic Regression
make nn_demo        # Neural Network

# Remove build artifacts
make clean
```

**Manual compilation** (without Make):

```bash
gcc -std=c99 -Wall -Iinclude src/*.c examples/train_linear_regression.c -o train_lr -lm
```

**Requirements:**
- `gcc` or any C99-compliant compiler
- Standard C library + `libm` (`-lm` flag)
- No other dependencies

---

## Example Usage

### Training a Linear Regression Model

```c
#include "matrix.h"
#include "linear_regression.h"

/* Create a model for a single input feature */
LinearRegression model = create_linear_regression(1);

/* Train on X (n×1) and y (n×1) for 500 epochs */
train_linear_regression(&model, X, y, 500, 0.01f);

/* Run inference */
Matrix y_pred = predict(&model, X);

printf("Weight: %.4f\n", model.weights.data[0]);
printf("Bias  : %.4f\n", model.bias);

/* Cleanup */
free_matrix(&y_pred);
free_linear_regression(&model);
```

### Training a Logistic Classifier

```c
#include "matrix.h"
#include "logistic_regression.h"

LogisticRegression model = create_logistic_regression(n_features);
train_logistic_regression(&model, X, y, 1000, 0.1f);

Matrix probs = predict_logistic(&model, X);
/* probs.data[i] > 0.5 → class 1, else class 0 */

free_matrix(&probs);
free_logistic_regression(&model);
```

### Using the Neural Network

```c
#include "matrix.h"
#include "neural_network.h"

/* 10 inputs → 16 hidden neurons → 1 output */
NeuralNetwork net = create_network(10, 16, 1);

Matrix output = forward_network(&net, X);
/* output.data[i] ∈ (0, 1) — sigmoid probability per sample */

free_matrix(&output);
free_network(&net);
```

---

## Design Principles

| Principle | Detail |
|---|---|
| **Pure C99** | No C++, no external ML libraries |
| **Row-major storage** | `data[i * cols + j]`  — compatible with standard layout |
| **Explicit memory** | Every `create_*` has a matching `free_*`; no hidden allocations |
| **Dimension validation** | All operations check shapes and abort with a descriptive error |
| **Zero warnings** | Compiles clean under `-Wall -Wextra` |
| **No global state** | All state lives in model structs passed by pointer |

---

## Goals of the Project

CML exists to answer a simple question: *what does it actually take to implement machine learning from scratch?*

- **Understand ML internals** — see exactly how backprop, loss, and gradient updates work
- **Learn systems programming** — practice C memory management in a real-world project
- **Build a reference implementation** — something concrete to compare against textbook theory
- **Grow iteratively** — each iteration is a complete, working piece of the puzzle

---

## Contributing

Contributions are welcome. The project is intentionally structured to be easy to extend — each new algorithm is a self-contained header + source pair.

Ways to contribute:

- **Add new models** — SVM, k-NN, softmax regression, multi-class classification
- **Implement backpropagation** — the natural next step for the neural network
- **Improve numerical stability** — gradient clipping, weight initialization strategies
- **Optimize matrix operations** — cache-friendly algorithms, SIMD hints
- **Add tests** — unit tests for individual operations and end-to-end training checks
- **Improve documentation** — inline comments, worked examples, architecture diagrams

To contribute, fork the repository, make your changes in a feature branch, and open a pull request with a clear description of what was added or fixed.

---

## License

This project is licensed under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for details.
