/*
 * test_loss.c — Loss characterization tests
 *
 * Pins down mse/binary_cross_entropy against known values, and proves
 * the BCE numerical-stability clamp holds at the 0/1 boundary rather
 * than trusting it by inspection.
 */

#include <math.h>
#include <stdio.h>

#include "loss.h"
#include "matrix.h"
#include "test_utils.h"
#include "tests.h"

static void test_mse(void) {
  Matrix y_true = create_matrix(1, 4);
  float true_vals[4] = {0.0f, 1.0f, 1.0f, 0.0f};
  for (int i = 0; i < 4; i++) y_true.data[i] = true_vals[i];

  Matrix y_pred = create_matrix(1, 4);
  float pred_vals[4] = {0.1f, 0.8f, 0.6f, 0.2f};
  for (int i = 0; i < 4; i++) y_pred.data[i] = pred_vals[i];

  /* diffs^2: 0.01, 0.04, 0.16, 0.04 -> sum 0.25 -> mean 0.0625 */
  ASSERT_FLOAT_NEAR(mse(y_true, y_pred), 0.0625f, 1e-5f);

  free_matrix(&y_true);
  free_matrix(&y_pred);
}

static void test_binary_cross_entropy(void) {
  Matrix y_true = create_matrix(1, 2);
  y_true.data[0] = 1.0f;
  y_true.data[1] = 0.0f;

  Matrix y_pred = create_matrix(1, 2);
  y_pred.data[0] = 0.9f;
  y_pred.data[1] = 0.1f;

  /* -log(0.9) for both terms -> mean = -log(0.9) ~= 0.10536052 */
  float expected = -logf(0.9f);
  ASSERT_FLOAT_NEAR(binary_cross_entropy(y_true, y_pred), expected, 1e-4f);

  free_matrix(&y_true);
  free_matrix(&y_pred);
}

static void test_bce_boundary_clamp(void) {
  /* y_true=1 with y_pred at/near 0: without clamping this is log(0) -> -inf */
  Matrix y_true_1 = create_matrix(1, 1);
  y_true_1.data[0] = 1.0f;
  Matrix y_pred_0 = create_matrix(1, 1);
  y_pred_0.data[0] = 0.0f;

  float loss_a = binary_cross_entropy(y_true_1, y_pred_0);
  ASSERT_TRUE(isfinite(loss_a));

  /* symmetric case: y_true=0 with y_pred at/near 1 -> log(1-1) without clamp */
  Matrix y_true_0 = create_matrix(1, 1);
  y_true_0.data[0] = 0.0f;
  Matrix y_pred_1 = create_matrix(1, 1);
  y_pred_1.data[0] = 1.0f;

  float loss_b = binary_cross_entropy(y_true_0, y_pred_1);
  ASSERT_TRUE(isfinite(loss_b));

  free_matrix(&y_true_1);
  free_matrix(&y_pred_0);
  free_matrix(&y_true_0);
  free_matrix(&y_pred_1);
}

void run_loss_tests(void) {
  printf("-- Loss tests --\n");
  test_mse();
  test_binary_cross_entropy();
  test_bce_boundary_clamp();
}
