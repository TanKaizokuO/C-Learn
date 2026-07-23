/*
 * test_activations.c — Activation function tests
 *
 * Covers the existing forward relu/sigmoid (not previously tested) plus
 * the new relu_derivative/sigmoid_derivative, which backward_network
 * relies on to chain gradients through the hidden layer.
 */

#include <stdio.h>

#include "activations.h"
#include "test_utils.h"
#include "tests.h"

static void test_relu(void) {
  ASSERT_FLOAT_NEAR(relu(2.0f), 2.0f, 1e-6f);
  ASSERT_FLOAT_NEAR(relu(-3.0f), 0.0f, 1e-6f);
  ASSERT_FLOAT_NEAR(relu(0.0f), 0.0f, 1e-6f);
}

static void test_sigmoid(void) {
  ASSERT_FLOAT_NEAR(sigmoid(0.0f), 0.5f, 1e-6f);
  /* sigmoid(2) ~= 0.8807971 */
  ASSERT_FLOAT_NEAR(sigmoid(2.0f), 0.8807971f, 1e-5f);
  /* sigmoid(-2) ~= 0.1192029 */
  ASSERT_FLOAT_NEAR(sigmoid(-2.0f), 0.1192029f, 1e-5f);
}

static void test_relu_derivative(void) {
  ASSERT_FLOAT_NEAR(relu_derivative(2.0f), 1.0f, 1e-6f);
  ASSERT_FLOAT_NEAR(relu_derivative(-3.0f), 0.0f, 1e-6f);
  /* boundary: z = 0 is defined as 0 by convention */
  ASSERT_FLOAT_NEAR(relu_derivative(0.0f), 0.0f, 1e-6f);
}

static void test_sigmoid_derivative(void) {
  /* sigmoid'(0) = 0.5 * 0.5 = 0.25 */
  ASSERT_FLOAT_NEAR(sigmoid_derivative(0.0f), 0.25f, 1e-6f);
  /* sigmoid'(2) = sigmoid(2) * (1 - sigmoid(2)) ~= 0.1049936 */
  ASSERT_FLOAT_NEAR(sigmoid_derivative(2.0f), 0.1049936f, 1e-5f);
}

void run_activation_tests(void) {
  printf("-- Activation tests --\n");
  test_relu();
  test_sigmoid();
  test_relu_derivative();
  test_sigmoid_derivative();
}
