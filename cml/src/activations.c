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
