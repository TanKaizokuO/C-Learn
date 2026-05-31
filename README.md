# C-Learn — A Machine Learning Library in Pure C

> A lightweight, educational machine learning framework built from scratch in **pure C99** — no external ML dependencies, no black boxes.

CML implements the full stack of machine learning fundamentals — from raw matrix arithmetic up to feedforward neural networks — entirely in C. Every gradient, every weight update, and every forward pass is written out explicitly, making it a hands-on reference for understanding how ML systems work at the systems level.

---

## 🚀 Overview

Most ML frameworks abstract away the internals behind high-level APIs. CML does the opposite — it exposes every detail by design.

*   **Educational by intent** — built to be read alongside theory, not just executed.
*   **Pure C99** — no Python dependencies for training, no BLAS, no external ML libraries.
*   **Zero dependencies** — only the C standard library and `libm`.
*   **Iterative architecture** — each layer of the stack builds directly on the previous one.
*   **Memory-transparent** — explicit allocation and deallocation throughout; no hidden heap usage.

The library grows the ML stack from the ground up across 7 self-contained iterations, starting from a matrix struct and ending at a two-layer neural network.

---

## 🛠️ Features

| Component | Description |
| :--- | :--- |
| **Matrix Core** | Row-major matrix struct with full linear algebra ops |
| **Tensor Utilities** | `zeros`, `ones`, `random_matrix`, element-wise ops, `apply_function` |
| **Activation Functions** | ReLU, Sigmoid — composable with any matrix operation |
| **Loss Functions** | Mean Squared Error, Binary Cross-Entropy (numerically stable) |
| **Optimizer** | Gradient Descent — in-place parameter update |
| **Linear Regression** | $y = XW + b$ — trained with MSE + gradient descent |
| **Logistic Regression** | $y = \sigma(XW + b)$ — binary classification with BCE loss |
| **Dense Layer** | Fully-connected layer with weight + bias, forward pass |
| **Neural Network** | Two-layer feedforward net: $\text{Input} \to \text{Dense} \to \text{ReLU} \to \text{Dense} \to \text{Sigmoid}$ |

---

## 📐 Architecture

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

## 💾 Core Data Structure

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
// ... use m ...
free_matrix(&m);                  /* release; sets m.data = NULL */
```

---

## 🤖 Implemented Models & Examples

### Linear Regression

Models a continuous output using a linear relationship:

$$y = XW + b$$

*   **Loss:** Mean Squared Error → $\text{MSE} = \frac{1}{n} \sum (\hat{y} - y)^2$
*   **Gradients:** $dW = \frac{2}{n} X^T(\hat{y} - y)$, $db = \frac{2}{n} \sum(\hat{y} - y)$
*   **Update:** Standard gradient descent

```c
LinearRegression model = create_linear_regression(n_features);
train_linear_regression(&model, X, y, epochs, learning_rate);
Matrix y_pred = predict(&model, X);
free_matrix(&y_pred);
free_linear_regression(&model);
```

Sample result on $y = 3x + 2$:

```
Epoch  100 | Loss: 1.086636
Epoch  500 | Loss: 0.012999

Learned Weight :  2.9158   (true: 3.0)
Learned Bias   :  1.9937   (true: 2.0)
```

---

### Logistic Regression

Binary classification — maps inputs to a probability in $(0, 1)$:

$$y = \sigma(XW + b)$$

*   **Activation:** Sigmoid → $\sigma(x) = \frac{1}{1 + e^{-x}}$
*   **Loss:** Binary Cross-Entropy → $\text{L} = -\frac{1}{n} \sum [y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})]$
*   **Stability:** Predictions clamped to $[\epsilon, 1-\epsilon]$ to prevent $\log(0)$
*   **Gradients:** $dW = \frac{1}{n} X^T(\hat{y} - y)$ — exact gradient of BCE with sigmoid output

```c
LogisticRegression model = create_logistic_regression(n_features);
train_logistic_regression(&model, X, y, epochs, learning_rate);
Matrix probs = predict_logistic(&model, X);
// threshold at 0.5 for hard classification
free_matrix(&probs);
free_logistic_regression(&model);
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

$$Z = X \cdot W + b$$

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

> 📌 **Roadmap:** Backpropagation and training loop for the neural network are planned as a future iteration.

---

## 🚢 Case Study: Titanic Survival Prediction

A complete real-world classification pipeline is provided under `examples/titanic_logistic.c`. It loads and trains on preprocessed passenger data (`preprocessedDB.csv` derived from the Kaggle Titanic Dataset) to predict passenger survival.

### Pipeline Details
1. **Data Loading:** Parses `preprocessedDB.csv` into training and test matrices.
2. **Train/Test Split:** Splits the 891 samples (712 for training, 179 for testing).
3. **Training:** Trains the logistic regression classifier for 1,000 epochs with a learning rate of `0.1`.
4. **Evaluation:** Computes the predictions on the test set, builds a confusion matrix, and calculates overall accuracy.

### Run the Titanic Demo
Compile and execute the C binary:
```bash
make test_titanic_c
./test_titanic_c
```

Output Sample:
```
Loading data from preprocessedDB.csv...
Loaded 891 rows.
Training Logistic Regression model on 712 samples for 1000 epochs (LR=0.10)...
Training Logistic Regression...
Epoch  100 | Loss: 0.497180
Epoch  500 | Loss: 0.438229
Epoch 1000 | Loss: 0.428084

Evaluating precision on 179 test samples...

Accuracy: 0.8492

Confusion Matrix:
[[104 11]
 [16 48]]

Execution time: 0.07906 seconds
```

---

## 📁 Project Structure

```
C-Learn/
│
├── include/
│   ├── matrix.h                 ← Core matrix struct and operations
│   ├── activations.h            ← ReLU, Sigmoid activations
│   ├── loss.h                   ← MSE, Binary Cross-Entropy
│   ├── optimizer.h              ← Gradient Descent parameter updates
│   ├── linear_regression.h      ← Linear Regression definition & training
│   ├── logistic_regression.h    ← Logistic Regression definition & training
│   ├── dense_layer.h            ← Dense / Fully-Connected layer
│   └── neural_network.h         ← Two-layer Neural Network
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
│   ├── demo.c                        ← Iterations 1–4 showcase (matrix & activations)
│   ├── train_linear_regression.c     ← Linear Regression sample training
│   ├── train_logistic.c              ← Synthetic Logistic Regression training
│   ├── titanic_logistic.c            ← Real-world Titanic survival training
│   └── neural_network_demo.c         ← Neural Network forward pass demo
│
├── Makefile
├── Titanic-Dataset.csv               ← Raw Kaggle Titanic dataset
├── preprocessedDB.csv                ← Cleaned/preprocessed dataset for C ingestion
├── titanic.ipynb                     ← Jupyter notebook for data preprocessing & EDA
└── README.md
```

---

## ⚙️ Build Instructions

The project uses a standard `Makefile`. A C99-compatible compiler and `libm` are the only requirements.

```bash
# Build all standard targets
make

# Build individual targets
make demo           # Iterations 1–4 showcase
make train_lr       # Linear Regression demo
make train_logistic # Synthetic Logistic Regression demo
make test_titanic_c # Titanic Survival Classifier (real data)
make nn_demo        # Neural Network forward pass demo

# Remove build artifacts
make clean
```

### Manual Compilation
If you prefer to compile manually without `make`:
```bash
gcc -std=c99 -Wall -Wextra -Iinclude src/*.c examples/titanic_logistic.c -o test_titanic_c -lm
```

---

## 🧠 Design Principles

*   **Pure C99:** No C++ features, no external libraries.
*   **Row-major layout:** `data[i * cols + j]` facilitates memory layout alignment.
*   **Explicit memory:** Every `create_*` function has a corresponding `free_*` function; zero hidden heap operations.
*   **Dimension validation:** Matrix sizes are validated before execution to prevent out-of-bound errors.
*   **Warning-free:** Compiles cleanly with `-Wall -Wextra`.
*   **Stateless design:** All model parameters live in explicitly managed structs passed by reference.

---

## 🤝 Contributing

Contributions are welcome! If you want to expand CML, here are some great starting points:
*   **Backpropagation:** Implement backprop and training for the `NeuralNetwork` struct.
*   **New Models:** Add K-Nearest Neighbors (k-NN), Support Vector Machines (SVM), or Softmax Multi-class Regression.
*   **Numerical Optimizations:** Cache-friendly multiplication, SIMD auto-vectorization friendly loops, or Adam optimizer.
*   **More Data Pipelines:** Add parsing/training scripts for other classic datasets (e.g., MNIST).

To contribute, fork the repository, make your changes in a feature branch, and open a pull request.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
