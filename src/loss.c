/*
 * loss.c — Loss Functions Implementation
 *
 * Iteration 3: Scalar metrics for supervised learning.
 *
 * Both functions:
 *   - validate that y_true and y_pred share the same dimensions
 *   - average the loss across all elements (not just rows)
 */

#include "loss.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Numerical epsilon for log clamping in BCE. */
#define EPSILON 1e-7f

/*
 * mse — Mean Squared Error.
 *
 *   MSE = (1/N) * Σ (y_pred_i - y_true_i)²
 *
 * N = total number of elements (rows × cols).
 */
float mse(Matrix y_true, Matrix y_pred) {
  if (y_true.rows != y_pred.rows || y_true.cols != y_pred.cols) {
    fprintf(stderr,
            "[cml] dimension mismatch in 'mse': "
            "(%d×%d) vs (%d×%d)\n",
            y_true.rows, y_true.cols, y_pred.rows, y_pred.cols);
    exit(EXIT_FAILURE);
  }

  int n = y_true.rows * y_true.cols;
  float sum = 0.0f;

  for (int i = 0; i < n; i++) {
    float diff = y_pred.data[i] - y_true.data[i];
    sum += diff * diff;
  }

  return sum / (float)n;
}

/*
 * binary_cross_entropy — Binary Cross-Entropy loss.
 *
 *   BCE = -(1/N) * Σ [ y*log(p) + (1-y)*log(1-p) ]
 *
 * Predictions `p` are clamped to [EPSILON, 1-EPSILON] to prevent
 * log(0) → -inf, which would produce NaN in gradient computation.
 *
 * y_true values should be in {0, 1}.
 * y_pred values should be in (0, 1) — i.e. sigmoid outputs.
 */
float binary_cross_entropy(Matrix y_true, Matrix y_pred) {
  if (y_true.rows != y_pred.rows || y_true.cols != y_pred.cols) {
    fprintf(stderr,
            "[cml] dimension mismatch in 'binary_cross_entropy': "
            "(%d×%d) vs (%d×%d)\n",
            y_true.rows, y_true.cols, y_pred.rows, y_pred.cols);
    exit(EXIT_FAILURE);
  }

  int n = y_true.rows * y_true.cols;
  float sum = 0.0f;

  for (int i = 0; i < n; i++) {
    float y = y_true.data[i];

    /* Clamp prediction to avoid log(0) */
    float p = y_pred.data[i];
    if (p < EPSILON)
      p = EPSILON;
    if (p > 1.0f - EPSILON)
      p = 1.0f - EPSILON;

    sum += -(y * logf(p) + (1.0f - y) * logf(1.0f - p));
  }

  return sum / (float)n;
}
