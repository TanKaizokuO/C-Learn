/*
 * test_gradients.c — Backward pass gradient checking
 *
 * backward_network is a pure function of (net, cache, X, y) — the single
 * seam Tier 1's correctness is verified through. For every one of the
 * network's 21 parameters, compare the analytic gradient against the
 * numerical estimate (L(w+eps) - L(w-eps)) / 2*eps.
 *
 * eps=1e-3 and a relative-error tolerance of 1e-2 are calibrated for
 * float (not double) precision, since Matrix.data is float and the
 * textbook eps=1e-7 assumes double.
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
#define N_SAMPLES 5
#define GRAD_EPS 1e-3f
#define REL_TOL 1e-2f

static float loss_for_current_params(NeuralNetwork *net, Matrix X, Matrix y) {
  Matrix A2 = forward_network(net, X);
  float loss = binary_cross_entropy(y, A2);
  free_matrix(&A2);
  return loss;
}

static void check_param_gradient(NeuralNetwork *net, Matrix X, Matrix y,
                                  float *param, float analytic_grad,
                                  const char *label) {
  float original = *param;

  *param = original + GRAD_EPS;
  float loss_plus = loss_for_current_params(net, X, y);

  *param = original - GRAD_EPS;
  float loss_minus = loss_for_current_params(net, X, y);

  *param = original;

  float numerical_grad = (loss_plus - loss_minus) / (2.0f * GRAD_EPS);

  float diff = fabsf(numerical_grad - analytic_grad);
  float denom = fmaxf(fabsf(numerical_grad), fabsf(analytic_grad));
  float rel_error = denom > 1e-8f ? diff / denom : diff;

  if (rel_error > REL_TOL) {
    printf("  [FAIL] %s: analytic=%.6f numerical=%.6f rel_error=%.6f\n",
           label, analytic_grad, numerical_grad, rel_error);
  }
  ASSERT_TRUE(rel_error <= REL_TOL);
}

void run_gradient_tests(void) {
  printf("-- Gradient checking tests --\n");

  NeuralNetwork net = create_network(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

  /* Fixed, hand-picked weights/data — deterministic, no reliance on
   * random_matrix's time-based seed. */
  float w1_vals[INPUT_SIZE * HIDDEN_SIZE] = {
      0.10f, -0.20f, 0.30f, -0.15f, 0.25f, 0.05f,
      -0.35f, 0.40f, -0.10f, 0.20f, 0.15f, -0.05f,
  };
  for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
    net.layer1.weights.data[i] = w1_vals[i];

  /* Biased well away from zero: relu has a non-differentiable kink at
   * z = 0, and a weight/bias perturbation of GRAD_EPS can flip a
   * near-zero pre-activation's sign, corrupting the numerical estimate.
   * These margins keep every sample's Z1 comfortably on one side. */
  float b1_vals[HIDDEN_SIZE] = {0.5f, -0.5f, 0.5f, -0.5f};
  for (int i = 0; i < HIDDEN_SIZE; i++) net.layer1.bias.data[i] = b1_vals[i];

  float w2_vals[HIDDEN_SIZE * OUTPUT_SIZE] = {0.6f, -0.9f, 0.45f, 0.15f};
  for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
    net.layer2.weights.data[i] = w2_vals[i];

  net.layer2.bias.data[0] = 0.02f;

  Matrix X = create_matrix(N_SAMPLES, INPUT_SIZE);
  float x_vals[N_SAMPLES * INPUT_SIZE] = {
      0.5f, -0.2f, 0.1f, -0.3f, 0.4f, 0.2f, 0.1f, 0.1f,
      -0.4f, -0.5f, -0.1f, 0.3f, 0.2f, 0.3f, -0.2f,
  };
  for (int i = 0; i < N_SAMPLES * INPUT_SIZE; i++) X.data[i] = x_vals[i];

  Matrix y = create_matrix(N_SAMPLES, OUTPUT_SIZE);
  float y_vals[N_SAMPLES] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
  for (int i = 0; i < N_SAMPLES; i++) y.data[i] = y_vals[i];

  ForwardCache cache = forward_network_cached(&net, X);
  Gradients grads = backward_network(&net, cache, X, y);

  char label[64];

  for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
    snprintf(label, sizeof(label), "dW1[%d]", i);
    check_param_gradient(&net, X, y, &net.layer1.weights.data[i],
                          grads.dW1.data[i], label);
  }
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    snprintf(label, sizeof(label), "db1[%d]", i);
    check_param_gradient(&net, X, y, &net.layer1.bias.data[i],
                          grads.db1.data[i], label);
  }
  for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
    snprintf(label, sizeof(label), "dW2[%d]", i);
    check_param_gradient(&net, X, y, &net.layer2.weights.data[i],
                          grads.dW2.data[i], label);
  }
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    snprintf(label, sizeof(label), "db2[%d]", i);
    check_param_gradient(&net, X, y, &net.layer2.bias.data[i],
                          grads.db2.data[i], label);
  }

  free_forward_cache(&cache);
  free_gradients(&grads);
  free_matrix(&X);
  free_matrix(&y);
  free_network(&net);
}
