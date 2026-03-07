/*
 * optimizer.c — Gradient Descent Optimizer Implementation
 *
 * Iteration 4: In-place weight update rule:
 *
 *   W = W - learning_rate * ∇W
 *
 * All updates are applied element-wise directly to the weights buffer.
 */

#include "optimizer.h"

#include <stdio.h>
#include <stdlib.h>

/*
 * gradient_descent — Apply one step of gradient descent in place.
 *
 *   weights      : the parameter matrix to update (modified directly)
 *   gradients    : ∇W, must match dimensions of weights exactly
 *   learning_rate: step size η (e.g. 0.01, 0.001)
 *
 * No new memory is allocated — updates are applied directly to
 * weights->data for efficiency.
 */
void gradient_descent(Matrix *weights, Matrix gradients, float learning_rate) {
  /* Validate dimension compatibility */
  if (weights->rows != gradients.rows || weights->cols != gradients.cols) {
    fprintf(stderr,
            "[cml] dimension mismatch in 'gradient_descent': "
            "weights (%d×%d) vs gradients (%d×%d)\n",
            weights->rows, weights->cols, gradients.rows, gradients.cols);
    exit(EXIT_FAILURE);
  }

  int n = weights->rows * weights->cols;

  /* W = W - η * ∇W  (element-wise, in place) */
  for (int i = 0; i < n; i++)
    weights->data[i] -= learning_rate * gradients.data[i];
}
