/*
 * dense_layer.h — Dense (Fully-Connected) Layer
 *
 * Iteration 7: Building block of feedforward neural networks.
 *
 * Forward pass:
 *   Z = X · W + b
 *
 *   X  — input  (n_samples × input_size)
 *   W  — weights (input_size × output_size)
 *   b  — bias    (1 × output_size), broadcast across all samples
 *   Z  — output  (n_samples × output_size)
 */

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "matrix.h"

/* ─────────────────────────────────────────────
 * Layer Structure
 * ───────────────────────────────────────────── */

typedef struct {
  Matrix weights; /* shape: (input_size × output_size), random init */
  Matrix bias;    /* shape: (1 × output_size), zero init             */
} DenseLayer;

/* ─────────────────────────────────────────────
 * Lifecycle
 * ───────────────────────────────────────────── */

/* Allocate a dense layer with random weights and zero bias. */
DenseLayer create_dense_layer(int input_size, int output_size);

/* Free weights and bias matrices. */
void free_dense_layer(DenseLayer *layer);

/* ─────────────────────────────────────────────
 * Forward Pass
 * ───────────────────────────────────────────── */

/*
 * forward_dense — Linear transformation: Z = X·W + b
 *
 * Bias is broadcast: bias[j] added to every row of Z.
 * Returns a newly allocated (n_samples × output_size) matrix.
 * Caller must free.
 */
Matrix forward_dense(DenseLayer *layer, Matrix X);

/* ─────────────────────────────────────────────
 * Activation Helper
 * ───────────────────────────────────────────── */

/*
 * apply_activation — Apply a unary function element-wise.
 *
 * Unlike apply_function() (which edits in place), this returns a
 * newly allocated result so intermediate values can be preserved.
 * Caller must free the returned matrix.
 *
 * Compatible with relu, sigmoid, or any float→float function.
 */
Matrix apply_activation(Matrix m, float (*func)(float));

#endif /* DENSE_LAYER_H */
