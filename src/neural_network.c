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
#include "loss.h"
#include "optimizer.h"

#include <stdio.h>

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

/*
 * forward_network_cached — Same computation as forward_network, but
 * keeps every intermediate alive for backward_network.
 */
ForwardCache forward_network_cached(NeuralNetwork *net, Matrix X) {
  ForwardCache cache;

  cache.Z1 = forward_dense(&net->layer1, X);
  cache.A1 = apply_activation(cache.Z1, relu);

  cache.Z2 = forward_dense(&net->layer2, cache.A1);
  cache.A2 = apply_activation(cache.Z2, sigmoid);

  return cache;
}

/* free_forward_cache — Release every matrix held by a ForwardCache. */
void free_forward_cache(ForwardCache *cache) {
  free_matrix(&cache->Z1);
  free_matrix(&cache->A1);
  free_matrix(&cache->Z2);
  free_matrix(&cache->A2);
}

/* ─────────────────────────────────────────────
 * Backward Pass
 * ───────────────────────────────────────────── */

/*
 * backward_network — Gradients of BCE loss w.r.t. every parameter.
 *
 * Step 1  Output layer:
 *   dZ2 = (A2 - y) / N     fused BCE+sigmoid shortcut, N = total
 *                          elements in y (matches loss.c's averaging)
 *   dW2, db2, dA1 = backward_dense(layer2, A1, dZ2)
 *
 * Step 2  Hidden layer:
 *   dZ1 = dA1 ⊙ relu_derivative(Z1)   chained through the real
 *                                     activation derivative — ReLU has
 *                                     no fused shortcut
 *   dW1, db1, _ = backward_dense(layer1, X, dZ1)
 *
 * Pure: net, cache, X, and y are all read-only.
 */
Gradients backward_network(NeuralNetwork *net, ForwardCache cache, Matrix X,
                            Matrix y) {
  float scale = 1.0f / (float)(y.rows * y.cols);

  Matrix error = subtract(cache.A2, y);
  Matrix dZ2 = scalar_multiply(error, scale);
  free_matrix(&error);

  DenseGradients out_grads = backward_dense(&net->layer2, cache.A1, dZ2);
  free_matrix(&dZ2);

  Matrix relu_deriv_Z1 = apply_activation(cache.Z1, relu_derivative);
  Matrix dZ1 = elementwise_multiply(out_grads.dX, relu_deriv_Z1);
  free_matrix(&relu_deriv_Z1);
  free_matrix(&out_grads.dX);

  DenseGradients hidden_grads = backward_dense(&net->layer1, X, dZ1);
  free_matrix(&dZ1);
  free_matrix(&hidden_grads.dX); /* no earlier layer to propagate to */

  Gradients grads;
  grads.dW1 = hidden_grads.dW;
  grads.db1 = hidden_grads.db;
  grads.dW2 = out_grads.dW;
  grads.db2 = out_grads.db;
  return grads;
}

/* free_gradients — Release every matrix held by a Gradients struct. */
void free_gradients(Gradients *grads) {
  free_matrix(&grads->dW1);
  free_matrix(&grads->db1);
  free_matrix(&grads->dW2);
  free_matrix(&grads->db2);
}

/* ─────────────────────────────────────────────
 * Training
 * ───────────────────────────────────────────── */

/*
 * train_network — Gradient descent with BCE loss.
 *
 * Per epoch:
 *   1. Forward:  cache = forward_network_cached(net, X)
 *   2. Loss:     BCE(y, cache.A2)
 *   3. Grads:    backward_network(net, cache, X, y)
 *   4. Update:   gradient_descent per parameter (W1, b1, W2, b2)
 *   5. Free temporaries
 *
 * Progress printed every 100 epochs (plus first/last), matching the
 * logging style of train_logistic_regression.
 */
void train_network(NeuralNetwork *net, Matrix X, Matrix y, int epochs,
                    float learning_rate) {
  printf("Training Neural Network...\n\n");

  for (int epoch = 1; epoch <= epochs; epoch++) {
    ForwardCache cache = forward_network_cached(net, X);
    float loss = binary_cross_entropy(y, cache.A2);

    Gradients grads = backward_network(net, cache, X, y);

    gradient_descent(&net->layer1.weights, grads.dW1, learning_rate);
    gradient_descent(&net->layer1.bias, grads.db1, learning_rate);
    gradient_descent(&net->layer2.weights, grads.dW2, learning_rate);
    gradient_descent(&net->layer2.bias, grads.db2, learning_rate);

    if (epoch == 1 || epoch % 100 == 0 || epoch == epochs)
      printf("Epoch %4d | Loss: %.6f\n", epoch, loss);

    free_gradients(&grads);
    free_forward_cache(&cache);
  }

  printf("\n");
}
