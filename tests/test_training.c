/*
 * test_training.c — Training loop integration test
 *
 * Proves train_network actually reduces loss over time: trains the
 * same 3-4-1 network shape the gradient checker uses on a small,
 * linearly-separable-ish synthetic dataset, and asserts final loss is
 * meaningfully lower than initial loss.
 */

#include <math.h>
#include <stdio.h>

#include "loss.h"
#include "matrix.h"
#include "neural_network.h"
#include "test_utils.h"
#include "tests.h"

#define INPUT_SIZE 3
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define N_SAMPLES 20
#define EPOCHS 500
#define LEARNING_RATE 0.5f

/*
 * Deterministic synthetic dataset (no time-seeded RNG, so the test is
 * reproducible): features come from a fixed formula, and the label is
 * the sign of a linear combination of them — linearly separable by
 * construction, which a 3-4-1 network should be able to fit.
 */
static void build_dataset(Matrix *X, Matrix *y) {
  *X = create_matrix(N_SAMPLES, INPUT_SIZE);
  *y = create_matrix(N_SAMPLES, OUTPUT_SIZE);

  for (int i = 0; i < N_SAMPLES; i++) {
    float x0 = sinf((float)i * 0.7f);
    float x1 = cosf((float)i * 1.3f);
    float x2 = sinf((float)i * 0.4f + 1.0f);

    X->data[i * INPUT_SIZE + 0] = x0;
    X->data[i * INPUT_SIZE + 1] = x1;
    X->data[i * INPUT_SIZE + 2] = x2;

    float score = x0 + x1 - x2;
    y->data[i] = score > 0.0f ? 1.0f : 0.0f;
  }
}

void run_training_tests(void) {
  printf("-- Training loop integration test --\n");

  Matrix X, y;
  build_dataset(&X, &y);

  NeuralNetwork net = create_network(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

  Matrix initial_pred = forward_network(&net, X);
  float initial_loss = binary_cross_entropy(y, initial_pred);
  free_matrix(&initial_pred);

  train_network(&net, X, y, EPOCHS, LEARNING_RATE);

  Matrix final_pred = forward_network(&net, X);
  float final_loss = binary_cross_entropy(y, final_pred);
  free_matrix(&final_pred);

  printf("  initial loss: %.6f | final loss: %.6f\n", initial_loss,
         final_loss);

  /* "Meaningfully lower", not just lower — guards against a no-op
   * update that happens to nudge loss down by noise alone. */
  ASSERT_TRUE(final_loss < initial_loss * 0.7f);

  free_network(&net);
  free_matrix(&X);
  free_matrix(&y);
}
