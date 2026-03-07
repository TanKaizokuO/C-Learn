/*
 * logistic_regression.c — Logistic Regression Implementation
 *
 * Iteration 6: Binary classification using sigmoid activation and
 * Binary Cross-Entropy loss.
 *
 * Key difference from linear regression:
 *   - Predictions are passed through sigmoid → probabilities in (0,1)
 *   - Loss is BCE, not MSE
 *   - Weight gradient uses factor (1/n) instead of (2/n)
 */

#include "logistic_regression.h"
#include "activations.h"
#include "loss.h"
#include "optimizer.h"

#include <stdio.h>

/* ─────────────────────────────────────────────
 * Lifecycle
 * ───────────────────────────────────────────── */

/*
 * create_logistic_regression — Allocate and initialise a model.
 * Weights are random in [-1, 1], bias starts at 0.
 */
LogisticRegression create_logistic_regression(int input_features) {
  LogisticRegression model;
  model.weights = random_matrix(input_features, 1);
  model.bias = 0.0f;
  return model;
}

/* free_logistic_regression — Release the weight buffer. */
void free_logistic_regression(LogisticRegression *model) {
  free_matrix(&model->weights);
}

/* ─────────────────────────────────────────────
 * Forward Pass
 * ───────────────────────────────────────────── */

/*
 * predict_logistic — y_pred = sigmoid(X·W + b)
 *
 *   1. Linear: Z = X · W + b
 *   2. Activation: apply sigmoid element-wise to Z
 *
 * Returns (n_samples × 1).  Caller must free.
 */
Matrix predict_logistic(LogisticRegression *model, Matrix X) {
  /* Step 1: linear combination */
  Matrix Z = matmul(X, model->weights);
  int n = Z.rows;
  for (int i = 0; i < n; i++)
    Z.data[i] += model->bias;

  /* Step 2: sigmoid squash → probabilities */
  apply_function(&Z, sigmoid);
  return Z;
}

/* ─────────────────────────────────────────────
 * Gradient Computation
 * ───────────────────────────────────────────── */

/*
 * compute_logistic_weight_gradient
 *
 *   dL/dW = (1/n) · Xᵀ · (y_pred - y_true)
 *
 * This is the exact gradient of BCE loss w.r.t. W when the output
 * activation is sigmoid (derivative of BCE and sigmoid chain-rule cancel).
 *
 * Returns (n_features × 1). Caller must free.
 */
Matrix compute_logistic_weight_gradient(Matrix X, Matrix y_true,
                                        Matrix y_pred) {
  int n = y_true.rows * y_true.cols;

  /* error = y_pred - y_true  (n_samples × 1) */
  Matrix error = subtract(y_pred, y_true);

  /* Xᵀ · error → (n_features × 1) */
  Matrix Xt = transpose(X);
  Matrix dW = matmul(Xt, error);

  /* Scale by 1/n */
  float scale = 1.0f / (float)n;
  int sz = dW.rows * dW.cols;
  for (int i = 0; i < sz; i++)
    dW.data[i] *= scale;

  free_matrix(&error);
  free_matrix(&Xt);
  return dW;
}

/*
 * compute_logistic_bias_gradient
 *
 *   dL/db = (1/n) · Σ(y_pred - y_true)
 */
float compute_logistic_bias_gradient(Matrix y_true, Matrix y_pred) {
  int n = y_true.rows * y_true.cols;
  float sum = 0.0f;
  for (int i = 0; i < n; i++)
    sum += y_pred.data[i] - y_true.data[i];
  return sum / (float)n;
}

/* ─────────────────────────────────────────────
 * Training Loop
 * ───────────────────────────────────────────── */

/*
 * train_logistic_regression — Gradient descent with BCE loss.
 *
 * Per epoch:
 *   1. Forward: y_pred = sigmoid(XW + b)
 *   2. Loss:    BCE(y_true, y_pred)
 *   3. Grads:   dW, db
 *   4. Update:  W -= lr * dW  |  b -= lr * db
 *   5. Free temporaries
 *
 * Progress printed every 100 epochs.
 */
void train_logistic_regression(LogisticRegression *model, Matrix X, Matrix y,
                               int epochs, float learning_rate) {
  printf("Training Logistic Regression...\n\n");

  for (int epoch = 1; epoch <= epochs; epoch++) {

    /* Forward */
    Matrix y_pred = predict_logistic(model, X);

    /* Loss */
    float loss = binary_cross_entropy(y, y_pred);

    /* Gradients */
    Matrix dW = compute_logistic_weight_gradient(X, y, y_pred);
    float db = compute_logistic_bias_gradient(y, y_pred);

    /* Weight update (in place via optimizer) */
    gradient_descent(&model->weights, dW, learning_rate);

    /* Bias update */
    model->bias -= learning_rate * db;

    /* Logging */
    if (epoch % 100 == 0)
      printf("Epoch %4d | Loss: %.6f\n", epoch, loss);

    /* Cleanup */
    free_matrix(&y_pred);
    free_matrix(&dW);
  }

  printf("\n");
}
