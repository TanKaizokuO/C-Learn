/*
 * neural_network_demo.c — Neural Network Forward Pass Demo
 *
 * Iteration 7: Demonstrates the two-layer feedforward network.
 *
 * Architecture:  10 features → 16 hidden (ReLU) → 1 output (Sigmoid)
 *
 * This demo covers:
 *   1. Network creation and architecture overview
 *   2. Inspecting layer dimensions
 *   3. Forward pass on a batch of random inputs
 *   4. Forward pass on a single hand-crafted input
 *   5. Memory cleanup
 *
 * Note: Full training (backpropagation) is a planned future iteration.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "activations.h"
#include "dense_layer.h"
#include "matrix.h"
#include "neural_network.h"

/* ─── Configuration ─────────────────────────── */
#define INPUT_SIZE 10
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 1
#define BATCH_SIZE 5

int main(void) {
  srand((unsigned int)time(NULL));

  printf("══════════════════════════════════════════\n");
  printf("  Iteration 7 — Neural Network Core in C \n");
  printf("══════════════════════════════════════════\n\n");

  /* ────────────────────────────────────────────
   * 1. Create network
   * ──────────────────────────────────────────── */
  NeuralNetwork net = create_network(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

  printf("Architecture:\n");
  printf("  Input   : %d features\n", INPUT_SIZE);
  printf("  Hidden  : %d neurons  (activation: ReLU)\n", HIDDEN_SIZE);
  printf("  Output  : %d neuron   (activation: Sigmoid)\n\n", OUTPUT_SIZE);

  /* ────────────────────────────────────────────
   * 2. Inspect layer dimensions
   * ──────────────────────────────────────────── */
  printf("Layer Dimensions:\n");
  printf("  layer1.weights : (%d × %d)\n", net.layer1.weights.rows,
         net.layer1.weights.cols);
  printf("  layer1.bias    : (%d × %d)\n", net.layer1.bias.rows,
         net.layer1.bias.cols);
  printf("  layer2.weights : (%d × %d)\n", net.layer2.weights.rows,
         net.layer2.weights.cols);
  printf("  layer2.bias    : (%d × %d)\n\n", net.layer2.bias.rows,
         net.layer2.bias.cols);

  /* ────────────────────────────────────────────
   * 3. Forward pass on a random batch
   * ──────────────────────────────────────────── */
  printf("══════════════════════════════════════════\n");
  printf("  Batch Forward Pass  (%d samples)\n", BATCH_SIZE);
  printf("══════════════════════════════════════════\n");

  Matrix X_batch = random_matrix(BATCH_SIZE, INPUT_SIZE);
  Matrix out_batch = forward_network(&net, X_batch);

  printf("  Output probabilities (sigmoid → all in (0,1)):\n");
  for (int i = 0; i < BATCH_SIZE; i++)
    printf("    Sample %d : %.6f\n", i + 1, out_batch.data[i]);
  printf("\n");

  free_matrix(&X_batch);
  free_matrix(&out_batch);

  /* ────────────────────────────────────────────
   * 4. Step-by-step single-sample forward pass
   *    Shows each intermediate activation
   * ──────────────────────────────────────────── */
  printf("══════════════════════════════════════════\n");
  printf("  Step-by-Step Single Sample Pass\n");
  printf("══════════════════════════════════════════\n");

  Matrix x_single = random_matrix(1, INPUT_SIZE);
  printf("  Input x (1 × %d):  first 4 values: [", INPUT_SIZE);
  for (int j = 0; j < 4; j++)
    printf(" %.3f", x_single.data[j]);
  printf(" ... ]\n\n");

  /* Layer 1: linear */
  Matrix Z1 = forward_dense(&net.layer1, x_single);
  printf("  Z1 = layer1(x)  (%d × %d):  first 4: [", Z1.rows, Z1.cols);
  for (int j = 0; j < 4; j++)
    printf(" %.3f", Z1.data[j]);
  printf(" ... ]\n");

  /* Layer 1: relu */
  Matrix A1 = apply_activation(Z1, relu);
  printf("  A1 = relu(Z1)   (%d × %d):  first 4: [", A1.rows, A1.cols);
  for (int j = 0; j < 4; j++)
    printf(" %.3f", A1.data[j]);
  printf(" ... ]\n");
  free_matrix(&Z1);

  /* Layer 2: linear */
  Matrix Z2 = forward_dense(&net.layer2, A1);
  printf("  Z2 = layer2(A1) (%d × %d):  value:   %.4f\n", Z2.rows, Z2.cols,
         Z2.data[0]);
  free_matrix(&A1);

  /* Layer 2: sigmoid */
  Matrix A2 = apply_activation(Z2, sigmoid);
  printf("  A2 = sigmoid(Z2) → output probability: %.6f\n\n", A2.data[0]);
  free_matrix(&Z2);
  free_matrix(&A2);
  free_matrix(&x_single);

  /* ────────────────────────────────────────────
   * 5. Cleanup
   * ──────────────────────────────────────────── */
  free_network(&net);

  printf("══════════════════════════════════════════\n");
  printf("  Forward pass complete. Memory freed.\n");
  printf("══════════════════════════════════════════\n\n");
  printf("  Note: Backpropagation training is a\n");
  printf("  planned future iteration.\n\n");

  return 0;
}
