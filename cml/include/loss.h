/*
 * loss.h — Loss Functions
 *
 * Iteration 3: Scalar loss metrics for supervised learning.
 */

#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"

/* Mean Squared Error: mean of (y_pred - y_true)^2 over all elements. */
float mse(Matrix y_true, Matrix y_pred);

/*
 * Binary Cross-Entropy:
 *   mean of -[y*log(p) + (1-y)*log(1-p)]
 * Predictions are clamped to [1e-7, 1-1e-7] to avoid log(0).
 */
float binary_cross_entropy(Matrix y_true, Matrix y_pred);

#endif /* LOSS_H */
