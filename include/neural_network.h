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

/*
 * ForwardCache — Every intermediate from a forward pass, kept alive for
 * the backward pass. Unlike forward_network (which frees its
 * intermediates and returns only A2), forward_network_cached keeps
 * everything backward_network needs.
 */
typedef struct {
  Matrix Z1; /* layer1 pre-activation  : (n_samples × hidden_size) */
  Matrix A1; /* relu(Z1)                : (n_samples × hidden_size) */
  Matrix Z2; /* layer2 pre-activation  : (n_samples × output_size) */
  Matrix A2; /* sigmoid(Z2)             : (n_samples × output_size) */
} ForwardCache;

/*
 * forward_network_cached — Training-only forward pass.
 *
 * Same computation as forward_network, but returns every intermediate
 * instead of freeing them. Caller owns and must free the returned
 * cache (see free_forward_cache). forward_network is unchanged and
 * stays the lean path for inference.
 */
ForwardCache forward_network_cached(NeuralNetwork *net, Matrix X);

/* Free every matrix held by a ForwardCache. */
void free_forward_cache(ForwardCache *cache);

/* ─────────────────────────────────────────────
 * Backward Pass
 * ───────────────────────────────────────────── */

/*
 * Gradients — dL/d(parameter) for every weight and bias in the network,
 * same shapes as the corresponding layer1/layer2 weights and biases.
 */
typedef struct {
  Matrix dW1;
  Matrix db1;
  Matrix dW2;
  Matrix db2;
} Gradients;

/*
 * backward_network — Gradients of the loss w.r.t. every parameter.
 *
 * Pure function: computes gradients only, does not mutate net or apply
 * any update. cache must come from forward_network_cached(net, X) for
 * this same X.
 *
 * The output layer uses the fused BCE+sigmoid shortcut dZ2 = A2 - y
 * (see docs/adr/0001-fused-sigmoid-bce-gradient.md) rather than
 * chaining sigmoid_derivative through a separately computed BCE
 * derivative. The hidden layer has no such shortcut and chains through
 * the real relu_derivative(Z1).
 *
 * Caller must free the returned Gradients (see free_gradients).
 */
Gradients backward_network(NeuralNetwork *net, ForwardCache cache, Matrix X,
                            Matrix y);

/* Free every matrix held by a Gradients struct. */
void free_gradients(Gradients *grads);

/* ─────────────────────────────────────────────
 * Training
 * ───────────────────────────────────────────── */

/*
 * train_network — Gradient descent with Binary Cross-Entropy loss.
 *
 * Per epoch: forward_network_cached -> backward_network ->
 * gradient_descent, applied once per parameter (W1, b1, W2, b2).
 * Prints loss periodically.
 */
void train_network(NeuralNetwork *net, Matrix X, Matrix y, int epochs,
                    float learning_rate);

#endif /* NEURAL_NETWORK_H */
