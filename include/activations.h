/*
 * activations.h — Activation Functions
 *
 * Iteration 2: Functions that can be passed to apply_function().
 */

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

/* Rectified Linear Unit: max(0, x) */
float relu(float x);

/* Logistic sigmoid: 1 / (1 + exp(-x)) */
float sigmoid(float x);

/*
 * relu_derivative — d/dz relu(z), evaluated at the pre-activation z.
 * Returns 1 if z > 0, otherwise 0.
 */
float relu_derivative(float z);

/*
 * sigmoid_derivative — d/dz sigmoid(z), evaluated at the pre-activation z.
 *   sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
 */
float sigmoid_derivative(float z);

#endif /* ACTIVATIONS_H */
