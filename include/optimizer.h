/*
 * optimizer.h — Gradient Descent Optimizer
 *
 * Iteration 4: In-place weight update using the rule:
 *   W = W - learning_rate * gradients
 */

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.h"

/*
 * gradient_descent — In-place update of weights.
 *
 *   weights      : matrix to update (modified in place)
 *   gradients    : ∇W, must have same dimensions as weights
 *   learning_rate: step size η  (e.g. 0.01)
 */
void gradient_descent(Matrix *weights,
                      Matrix  gradients,
                      float   learning_rate);

#endif /* OPTIMIZER_H */
