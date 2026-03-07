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

#endif /* ACTIVATIONS_H */
