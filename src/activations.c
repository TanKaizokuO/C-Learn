/*
 * activations.c — Activation Functions Implementation
 *
 * Iteration 2: relu and sigmoid are standard float→float functions
 * and can be passed directly to apply_function().
 *
 * Example:
 *   Matrix m = random_matrix(3, 3);
 *   apply_function(&m, sigmoid);
 */

#include "activations.h"
#include <math.h>

/*
 * relu — Rectified Linear Unit.
 * Returns x if x > 0, otherwise 0.
 * Introduces non-linearity while being cheap to compute.
 */
float relu(float x) { return x > 0.0f ? x : 0.0f; }

/*
 * sigmoid — Logistic sigmoid function.
 * Maps any real number to the open interval (0, 1).
 * Commonly used in binary classification output layers.
 *
 *   σ(x) = 1 / (1 + e^(-x))
 */
float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/*
 * relu_derivative — Derivative of relu, evaluated at the pre-activation z.
 * relu is non-differentiable at z = 0; by convention we return 0 there.
 */
float relu_derivative(float z) { return z > 0.0f ? 1.0f : 0.0f; }

/*
 * sigmoid_derivative — Derivative of sigmoid, evaluated at the
 * pre-activation z.
 *
 *   sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
 */
float sigmoid_derivative(float z) {
  float s = sigmoid(z);
  return s * (1.0f - s);
}
