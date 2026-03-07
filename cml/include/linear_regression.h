/*
 * linear_regression.h — Linear Regression Model
 *
 * Iteration 5: First ML model built on top of the CML core.
 *
 * Model:  y = X·W + b
 *
 *   X  — input  (n_samples × n_features)
 *   W  — weights (n_features × 1)
 *   b  — bias scalar
 *   y  — output  (n_samples × 1)
 */

#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "matrix.h"

/* ─────────────────────────────────────────────
 * Model Structure
 * ───────────────────────────────────────────── */

typedef struct {
  Matrix weights; /* shape: (n_features × 1), randomly initialized */
  float bias;     /* scalar bias term, initialized to 0             */
} LinearRegression;

/* ─────────────────────────────────────────────
 * Lifecycle
 * ───────────────────────────────────────────── */

/* Allocate and randomly initialize a model for `input_features` inputs. */
LinearRegression create_linear_regression(int input_features);

/* Free the weight matrix held by the model. */
void free_linear_regression(LinearRegression *model);

/* ─────────────────────────────────────────────
 * Forward Pass
 * ───────────────────────────────────────────── */

/*
 * predict — Compute y_pred = X·W + b.
 *
 * Returns a newly allocated (n_samples × 1) matrix.
 * Caller is responsible for freeing the result.
 */
Matrix predict(LinearRegression *model, Matrix X);

/* ─────────────────────────────────────────────
 * Loss
 * ───────────────────────────────────────────── */

/*
 * compute_loss — Thin wrapper around mse().
 * Keeps the training loop self-contained.
 */
float compute_loss(Matrix y_true, Matrix y_pred);

/* ─────────────────────────────────────────────
 * Gradient Computation
 * ───────────────────────────────────────────── */

/*
 * compute_weight_gradient — dL/dW = (2/n) · Xᵀ · (y_pred - y_true)
 *
 * Returns a (n_features × 1) gradient matrix.
 * Caller is responsible for freeing the result.
 */
Matrix compute_weight_gradient(Matrix X, Matrix y_true, Matrix y_pred);

/*
 * compute_bias_gradient — dL/db = (2/n) · Σ(y_pred - y_true)
 *
 * Returns a scalar.
 */
float compute_bias_gradient(Matrix y_true, Matrix y_pred);

/* ─────────────────────────────────────────────
 * Training
 * ───────────────────────────────────────────── */

/*
 * train_linear_regression — Run gradient descent for `epochs` steps.
 *
 * Prints loss every 100 epochs.
 * Updates model->weights and model->bias in place.
 */
void train_linear_regression(LinearRegression *model, Matrix X, Matrix y,
                             int epochs, float learning_rate);

#endif /* LINEAR_REGRESSION_H */
