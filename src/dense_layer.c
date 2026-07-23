/*
 * dense_layer.c — Dense (Fully-Connected) Layer Implementation
 *
 * Iteration 7: Core building block of neural networks.
 *
 * Bias broadcast:
 *   Z[i][j] = Σk X[i][k] * W[k][j] + bias[j]
 *   Applied to every sample row i.
 */

#include "dense_layer.h"
#include "activations.h"

/* ─────────────────────────────────────────────
 * Lifecycle
 * ───────────────────────────────────────────── */

/*
 * create_dense_layer — Allocate a fully-connected layer.
 *
 *   weights : random_matrix(input_size, output_size) — values in [-1, 1]
 *   bias    : zeros(1, output_size) — broadcast to each sample row
 */
DenseLayer create_dense_layer(int input_size, int output_size) {
  DenseLayer layer;
  layer.weights = random_matrix(input_size, output_size);
  layer.bias = zeros(1, output_size);
  return layer;
}

/*
 * free_dense_layer — Release weights and bias matrices.
 */
void free_dense_layer(DenseLayer *layer) {
  free_matrix(&layer->weights);
  free_matrix(&layer->bias);
}

/* ─────────────────────────────────────────────
 * Forward Pass
 * ───────────────────────────────────────────── */

/*
 * forward_dense — Z = X · W + b
 *
 *   X       : (n_samples × input_size)
 *   W       : (input_size × output_size)
 *   b       : (1 × output_size), added to every row of XW
 *   result  : (n_samples × output_size) — caller must free
 *
 * Bias broadcast: for each sample i and output neuron j,
 *   Z[i][j] += bias[0][j]
 */
Matrix forward_dense(DenseLayer *layer, Matrix X) {
  /* Linear transformation: Z = X · W */
  Matrix Z = matmul(X, layer->weights);

  /* Broadcast bias across all sample rows */
  int output_size = Z.cols;
  for (int i = 0; i < Z.rows; i++)
    for (int j = 0; j < output_size; j++)
      Z.data[i * output_size + j] += layer->bias.data[j];

  return Z;
}

/* ─────────────────────────────────────────────
 * Backward Pass
 * ───────────────────────────────────────────── */

/*
 * sum_rows — Sum a matrix's rows into a single (1 × cols) row.
 * Used for the bias gradient: bias is broadcast to every sample row
 * on the forward pass, so its gradient is the sum across rows.
 */
static Matrix sum_rows(Matrix m) {
  Matrix result = zeros(1, m.cols);
  for (int i = 0; i < m.rows; i++)
    for (int j = 0; j < m.cols; j++)
      result.data[j] += m.data[i * m.cols + j];
  return result;
}

/*
 * backward_dense — Gradients of Z = X·W + b w.r.t. W, b, and X.
 *
 *   dW = Xᵀ · dZ    (input_size × output_size)
 *   db = Σ_rows dZ  (1 × output_size)
 *   dX = dZ · Wᵀ    (n_samples × input_size)
 */
DenseGradients backward_dense(DenseLayer *layer, Matrix X, Matrix dZ) {
  Matrix Xt = transpose(X);
  Matrix dW = matmul(Xt, dZ);
  free_matrix(&Xt);

  Matrix db = sum_rows(dZ);

  Matrix Wt = transpose(layer->weights);
  Matrix dX = matmul(dZ, Wt);
  free_matrix(&Wt);

  DenseGradients grads;
  grads.dW = dW;
  grads.db = db;
  grads.dX = dX;
  return grads;
}

/* free_dense_gradients — Release every matrix held by a DenseGradients. */
void free_dense_gradients(DenseGradients *grads) {
  free_matrix(&grads->dW);
  free_matrix(&grads->db);
  free_matrix(&grads->dX);
}

/* ─────────────────────────────────────────────
 * Activation Helper
 * ───────────────────────────────────────────── */

/*
 * apply_activation — Non-mutating activation: returns a new matrix.
 *
 * Applies func(x) to every element and returns the result.
 * The input matrix `m` is NOT modified.
 * Caller must free the returned matrix.
 *
 * Use this instead of apply_function() when you need to keep the
 * pre-activation values (e.g., for future backprop).
 */
Matrix apply_activation(Matrix m, float (*func)(float)) {
  Matrix result = create_matrix(m.rows, m.cols);
  int n = m.rows * m.cols;
  for (int i = 0; i < n; i++)
    result.data[i] = func(m.data[i]);
  return result;
}
