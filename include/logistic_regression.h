/*
 * logistic_regression.h — Logistic Regression Model
 *
 * Iteration 6: Binary classification via sigmoid activation.
 *
 * Model:
 *   y = sigmoid(X·W + b)
 *
 *   X  — input   (n_samples × n_features)
 *   W  — weights (n_features × 1)
 *   b  — bias scalar
 *   y  — output probability in (0, 1)  →  (n_samples × 1)
 */

#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "matrix.h"

/* ─────────────────────────────────────────────
 * Model Structure
 * ───────────────────────────────────────────── */

typedef struct {
  Matrix weights; /* shape: (n_features × 1), randomly initialized */
  float bias;     /* scalar bias, initialized to 0                 */
} LogisticRegression;

/* ─────────────────────────────────────────────
 * Lifecycle
 * ───────────────────────────────────────────── */

/* Allocate and randomly initialize a logistic model. */
LogisticRegression create_logistic_regression(int input_features);

/* Free the weight matrix held by the model. */
void free_logistic_regression(LogisticRegression *model);

/* ─────────────────────────────────────────────
 * Forward Pass
 * ───────────────────────────────────────────── */

/*
 * predict_logistic — Compute y_pred = sigmoid(X·W + b).
 *
 * Returns a newly allocated (n_samples × 1) probability matrix.
 * Each value is in (0, 1).  Caller must free.
 */
Matrix predict_logistic(LogisticRegression *model, Matrix X);

/* ─────────────────────────────────────────────
 * Gradient Computation
 * ───────────────────────────────────────────── */

/*
 * compute_logistic_weight_gradient
 *   dL/dW = (1/n) · Xᵀ · (y_pred - y_true)
 *
 * Returns (n_features × 1). Caller must free.
 */
Matrix compute_logistic_weight_gradient(Matrix X, Matrix y_true, Matrix y_pred);

/*
 * compute_logistic_bias_gradient
 *   dL/db = (1/n) · Σ(y_pred - y_true)
 */
float compute_logistic_bias_gradient(Matrix y_true, Matrix y_pred);

/* ─────────────────────────────────────────────
 * Training
 * ───────────────────────────────────────────── */

/*
 * train_logistic_regression — Run gradient descent for `epochs` steps.
 *
 * Uses Binary Cross-Entropy loss.
 * Prints loss every 100 epochs.
 */
void train_logistic_regression(LogisticRegression *model, Matrix X, Matrix y,
                               int epochs, float learning_rate);

#endif /* LOGISTIC_REGRESSION_H */
