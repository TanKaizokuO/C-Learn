/*
 * neural_network.c — Feedforward Neural Network Implementation
 *
 * Iteration 7: Two-layer network using DenseLayer + activations.
 *
 * Forward pass:
 *   Z1 = layer1(X)    →  (n_samples × hidden_size)
 *   A1 = relu(Z1)     →  (n_samples × hidden_size)
 *   Z2 = layer2(A1)   →  (n_samples × output_size)
 *   A2 = sigmoid(Z2)  →  (n_samples × output_size)   ← returned
 *
 * Intermediate matrices Z1, Z2, and A1 are freed before returning,
 * so the caller only owns A2.
 */

#include "neural_network.h"
#include "activations.h"

/* ─────────────────────────────────────────────
 * Lifecycle
 * ───────────────────────────────────────────── */

/*
 * create_network — Initialise a two-layer feedforward network.
 *
 *   layer1 : input_size  → hidden_size  (weights random, bias zero)
 *   layer2 : hidden_size → output_size  (weights random, bias zero)
 */
NeuralNetwork create_network(int input_size, int hidden_size, int output_size) {
  NeuralNetwork net;
  net.layer1 = create_dense_layer(input_size, hidden_size);
  net.layer2 = create_dense_layer(hidden_size, output_size);
  return net;
}

/*
 * free_network — Release all layer weights and biases.
 */
void free_network(NeuralNetwork *net) {
  free_dense_layer(&net->layer1);
  free_dense_layer(&net->layer2);
}

/* ─────────────────────────────────────────────
 * Forward Pass
 * ───────────────────────────────────────────── */

/*
 * forward_network — Full two-layer forward pass.
 *
 * Step 1  Hidden layer:
 *   Z1 = layer1(X)          linear transformation
 *   A1 = relu(Z1)           introduces non-linearity
 *
 * Step 2  Output layer:
 *   Z2 = layer2(A1)         projects to output space
 *   A2 = sigmoid(Z2)        squash to (0, 1) probability
 *
 * All intermediates (Z1, A1, Z2) are freed internally.
 * Returns A2  — caller must free.
 */
Matrix forward_network(NeuralNetwork *net, Matrix X) {
  /* ── Hidden layer ────────────────────────── */
  Matrix Z1 = forward_dense(&net->layer1, X);
  Matrix A1 = apply_activation(Z1, relu);
  free_matrix(&Z1);

  /* ── Output layer ────────────────────────── */
  Matrix Z2 = forward_dense(&net->layer2, A1);
  Matrix A2 = apply_activation(Z2, sigmoid);
  free_matrix(&A1);
  free_matrix(&Z2);

  return A2; /* (n_samples × output_size) */
}
