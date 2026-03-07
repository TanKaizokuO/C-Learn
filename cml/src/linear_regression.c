/*
 * linear_regression.c — Linear Regression Model Implementation
 *
 * Iteration 5: Ties together all four previous library layers:
 *
 *   Matrix ops  →  Forward pass  →  MSE loss
 *   →  Gradient computation  →  Gradient descent update
 *
 * Training rule:
 *   W ← W - lr · dW      where  dW = (2/n) · Xᵀ · (ŷ - y)
 *   b ← b - lr · db      where  db = (2/n) · Σ(ŷ - y)
 */

#include "linear_regression.h"
#include "loss.h"
#include "optimizer.h"

#include <stdio.h>
#include <stdlib.h>

/* ─────────────────────────────────────────────
 * Lifecycle
 * ───────────────────────────────────────────── */

/*
 * create_linear_regression — Allocate a model.
 *
 * Weights are drawn from [-1, 1] via random_matrix().
 * Bias is set to 0.
 */
LinearRegression create_linear_regression(int input_features) {
  LinearRegression model;
  model.weights = random_matrix(input_features, 1);
  model.bias = 0.0f;
  return model;
}

/*
 * free_linear_regression — Release the weight matrix.
 */
void free_linear_regression(LinearRegression *model) {
  free_matrix(&model->weights);
}

/* ─────────────────────────────────────────────
 * Forward Pass
 * ───────────────────────────────────────────── */

/*
 * predict — y_pred = X·W + b
 *
 *   X       : (n_samples × n_features)
 *   W       : (n_features × 1)
 *   result  : (n_samples × 1)  — caller must free
 *
 * Bias is broadcast: added to every row of the matmul result.
 */
Matrix predict(LinearRegression *model, Matrix X) {
  /* Step 1: linear combination  Z = X · W  →  (n_samples × 1) */
  Matrix Z = matmul(X, model->weights);

  /* Step 2: add bias to every sample output */
  int n = Z.rows; /* n_samples */
  for (int i = 0; i < n; i++)
    Z.data[i] += model->bias;

  return Z; /* shape: (n_samples × 1) */
}

/* ─────────────────────────────────────────────
 * Loss
 * ───────────────────────────────────────────── */

/*
 * compute_loss — Thin wrapper over mse().
 * Kept here so the training loop is self-documenting.
 */
float compute_loss(Matrix y_true, Matrix y_pred) { return mse(y_true, y_pred); }

/* ─────────────────────────────────────────────
 * Gradient Computation
 * ───────────────────────────────────────────── */

/*
 * compute_weight_gradient — dL/dW = (2/n) · Xᵀ · (ŷ - y)
 *
 *   error   : (n_samples × 1)  = y_pred - y_true
 *   Xᵀ      : (n_features × n_samples)
 *   dW      : (n_features × 1) — caller must free
 */
Matrix compute_weight_gradient(Matrix X, Matrix y_true, Matrix y_pred) {
  int n = y_true.rows * y_true.cols; /* total elements */

  /* error = y_pred - y_true  (n_samples × 1) */
  Matrix error = subtract(y_pred, y_true);

  /* Xᵀ · error  →  (n_features × 1) */
  Matrix Xt = transpose(X);
  Matrix dW = matmul(Xt, error);

  /* Scale by 2/n */
  float scale = 2.0f / (float)n;
  for (int i = 0; i < dW.rows * dW.cols; i++)
    dW.data[i] *= scale;

  free_matrix(&error);
  free_matrix(&Xt);

  return dW; /* shape: (n_features × 1) — caller owns this */
}

/*
 * compute_bias_gradient — dL/db = (2/n) · Σ(ŷ - y)
 *
 * Returns a scalar: the mean of (y_pred - y_true) times 2.
 */
float compute_bias_gradient(Matrix y_true, Matrix y_pred) {
  int n = y_true.rows * y_true.cols;
  float sum = 0.0f;
  for (int i = 0; i < n; i++)
    sum += y_pred.data[i] - y_true.data[i];
  return (2.0f / (float)n) * sum;
}

/* ─────────────────────────────────────────────
 * Training Loop
 * ───────────────────────────────────────────── */

/*
 * train_linear_regression — Full gradient descent training.
 *
 * Per epoch:
 *   1. Forward pass → y_pred
 *   2. Compute MSE loss
 *   3. Compute dW, db
 *   4. Update weights in place (gradient_descent)
 *   5. Update bias:  b ← b - lr * db
 *   6. Free temporaries
 *
 * Progress is printed every 100 epochs.
 */
void train_linear_regression(LinearRegression *model, Matrix X, Matrix y,
                             int epochs, float learning_rate) {
  printf("Training Linear Regression...\n\n");

  for (int epoch = 1; epoch <= epochs; epoch++) {

    /* ── Forward pass ─────────────────────── */
    Matrix y_pred = predict(model, X);

    /* ── Loss ─────────────────────────────── */
    float loss = compute_loss(y, y_pred);

    /* ── Gradients ────────────────────────── */
    Matrix dW = compute_weight_gradient(X, y, y_pred);
    float db = compute_bias_gradient(y, y_pred);

    /* ── Weight update (in place) ─────────── */
    gradient_descent(&model->weights, dW, learning_rate);

    /* ── Bias update (scalar, in place) ───── */
    model->bias -= learning_rate * db;

    /* ── Logging ──────────────────────────── */
    if (epoch % 100 == 0)
      printf("Epoch %4d | Loss: %.6f\n", epoch, loss);

    /* ── Cleanup temporaries ──────────────── */
    free_matrix(&y_pred);
    free_matrix(&dW);
  }

  printf("\n");
}
