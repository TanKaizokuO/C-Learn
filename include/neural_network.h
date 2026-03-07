/*
 * neural_network.h — Minimal Feedforward Neural Network
 *
 * Iteration 7: Two-layer network built on DenseLayer.
 *
 * Architecture:
 *   Input → Dense(hidden) → ReLU → Dense(output) → Sigmoid
 *
 * For a single forward pass:
 *   Z1 = layer1(X)       : (n_samples × hidden_size)
 *   A1 = relu(Z1)        : (n_samples × hidden_size)
 *   Z2 = layer2(A1)      : (n_samples × output_size)
 *   A2 = sigmoid(Z2)     : (n_samples × output_size)  ← final prediction
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "dense_layer.h"
#include "matrix.h"

/* ─────────────────────────────────────────────
 * Network Structure
 * ───────────────────────────────────────────── */

typedef struct {
  DenseLayer layer1; /* input  → hidden : (input_size  × hidden_size)  */
  DenseLayer layer2; /* hidden → output : (hidden_size × output_size)  */
} NeuralNetwork;

/* ─────────────────────────────────────────────
 * Lifecycle
 * ───────────────────────────────────────────── */

/*
 * create_network — Build a two-layer network with random weights.
 *
 *   input_size  : number of input features
 *   hidden_size : neurons in the hidden layer
 *   output_size : neurons in the output layer (1 for binary)
 */
NeuralNetwork create_network(int input_size, int hidden_size, int output_size);

/* Free both layers. */
void free_network(NeuralNetwork *net);

/* ─────────────────────────────────────────────
 * Forward Pass
 * ───────────────────────────────────────────── */

/*
 * forward_network — Full forward pass through the two-layer network.
 *
 * Computes: A2 = sigmoid(layer2(relu(layer1(X))))
 *
 * Returns a newly allocated (n_samples × output_size) matrix.
 * Caller must free.
 */
Matrix forward_network(NeuralNetwork *net, Matrix X);

#endif /* NEURAL_NETWORK_H */
