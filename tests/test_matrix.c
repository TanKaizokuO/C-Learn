/*
 * test_matrix.c — Matrix core characterization tests
 *
 * Pins down matrix.h's public API against known, hand-computed values.
 * Bias broadcasting is not a matrix.c primitive (it's inlined inside
 * forward_dense), so it's out of scope here.
 */

#include <stdio.h>

#include "matrix.h"
#include "test_utils.h"
#include "tests.h"

static void test_matmul(void) {
  /* A (2x3) * B (3x2) -> (2x2), hand-computed */
  Matrix a = create_matrix(2, 3);
  float a_vals[6] = {1, 2, 3, 4, 5, 6};
  for (int i = 0; i < 6; i++) a.data[i] = a_vals[i];

  Matrix b = create_matrix(3, 2);
  float b_vals[6] = {7, 8, 9, 10, 11, 12};
  for (int i = 0; i < 6; i++) b.data[i] = b_vals[i];

  Matrix result = matmul(a, b);

  ASSERT_TRUE(result.rows == 2 && result.cols == 2);
  ASSERT_FLOAT_NEAR(result.data[0], 58.0f, 1e-5f);
  ASSERT_FLOAT_NEAR(result.data[1], 64.0f, 1e-5f);
  ASSERT_FLOAT_NEAR(result.data[2], 139.0f, 1e-5f);
  ASSERT_FLOAT_NEAR(result.data[3], 154.0f, 1e-5f);

  free_matrix(&a);
  free_matrix(&b);
  free_matrix(&result);
}

static void test_transpose(void) {
  /* M (2x3) -> M^T (3x2), non-square */
  Matrix m = create_matrix(2, 3);
  float m_vals[6] = {1, 2, 3, 4, 5, 6};
  for (int i = 0; i < 6; i++) m.data[i] = m_vals[i];

  Matrix t = transpose(m);

  ASSERT_TRUE(t.rows == 3 && t.cols == 2);
  float expected[6] = {1, 4, 2, 5, 3, 6};
  for (int i = 0; i < 6; i++)
    ASSERT_FLOAT_NEAR(t.data[i], expected[i], 1e-5f);

  free_matrix(&m);
  free_matrix(&t);
}

static void test_add_subtract(void) {
  Matrix a = create_matrix(2, 2);
  float a_vals[4] = {1, 2, 3, 4};
  for (int i = 0; i < 4; i++) a.data[i] = a_vals[i];

  Matrix b = create_matrix(2, 2);
  float b_vals[4] = {5, 6, 7, 8};
  for (int i = 0; i < 4; i++) b.data[i] = b_vals[i];

  Matrix sum = add(a, b);
  float expected_sum[4] = {6, 8, 10, 12};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_NEAR(sum.data[i], expected_sum[i], 1e-5f);

  Matrix diff = subtract(a, b);
  float expected_diff[4] = {-4, -4, -4, -4};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_NEAR(diff.data[i], expected_diff[i], 1e-5f);

  free_matrix(&a);
  free_matrix(&b);
  free_matrix(&sum);
  free_matrix(&diff);
}

static void test_elementwise_ops(void) {
  Matrix a = create_matrix(2, 2);
  float a_vals[4] = {1, 2, 3, 4};
  for (int i = 0; i < 4; i++) a.data[i] = a_vals[i];

  Matrix b = create_matrix(2, 2);
  float b_vals[4] = {5, 6, 7, 8};
  for (int i = 0; i < 4; i++) b.data[i] = b_vals[i];

  Matrix esum = elementwise_add(a, b);
  float expected_sum[4] = {6, 8, 10, 12};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_NEAR(esum.data[i], expected_sum[i], 1e-5f);

  Matrix eprod = elementwise_multiply(a, b);
  float expected_prod[4] = {5, 12, 21, 32};
  for (int i = 0; i < 4; i++)
    ASSERT_FLOAT_NEAR(eprod.data[i], expected_prod[i], 1e-5f);

  free_matrix(&a);
  free_matrix(&b);
  free_matrix(&esum);
  free_matrix(&eprod);
}

static void test_dot_product(void) {
  float a[3] = {1, 2, 3};
  float b[3] = {4, 5, 6};

  /* 1*4 + 2*5 + 3*6 = 32 */
  ASSERT_FLOAT_NEAR(dot_product(a, b, 3), 32.0f, 1e-5f);
}

void run_matrix_tests(void) {
  printf("-- Matrix core tests --\n");
  test_matmul();
  test_transpose();
  test_add_subtract();
  test_elementwise_ops();
  test_dot_product();
}
