/*
 * demo.c — End-to-End CML Demo
 *
 * Demonstrates all four library layers in sequence:
 *   1. Matrix creation & linear algebra
 *   2. Tensor utilities (zeros, ones, random) + activations
 *   3. Loss functions  (MSE, BCE)
 *   4. Gradient descent optimizer step
 */

#include <stdio.h>

#include "activations.h"
#include "loss.h"
#include "matrix.h"
#include "optimizer.h"

/* ─── Pretty section header ─────────────────── */
static void section(const char *title) {
  printf("\n══════════════════════════════════════════\n");
  printf("  %s\n", title);
  printf("══════════════════════════════════════════\n");
}

int main(void) {
  printf("CML — C Machine Learning Library Demo\n");

  /* ────────────────────────────────────────────
   * ITERATION 1 — Linear Algebra Core
   * ──────────────────────────────────────────── */
  section("Iteration 1 — Linear Algebra Core");

  /* Manual 2×3 matrix A */
  Matrix A = create_matrix(2, 3);
  float a_vals[] = {1, 2, 3, 4, 5, 6};
  for (int i = 0; i < 6; i++)
    A.data[i] = a_vals[i];
  print_matrix(&A, "A (2×3)");

  /* Manual 3×2 matrix B */
  Matrix B = create_matrix(3, 2);
  float b_vals[] = {7, 8, 9, 10, 11, 12};
  for (int i = 0; i < 6; i++)
    B.data[i] = b_vals[i];
  print_matrix(&B, "B (3×2)");

  /* Matrix multiplication: A (2×3) · B (3×2) → C (2×2) */
  Matrix C = matmul(A, B);
  print_matrix(&C, "C = A × B  (2×2)");

  /* Transpose */
  Matrix At = transpose(A);
  print_matrix(&At, "Aᵀ (3×2)");

  /* Scalar multiply */
  Matrix A2 = scalar_multiply(A, 2.0f);
  print_matrix(&A2, "A × 2");

  /* Dot product of first row of A with first column of B.
   * B is row-major (2 cols): col 0 elements are at indices 0, 2, 4. */
  float b_col0[3] = {B.data[0], B.data[2], B.data[4]};
  float dp = dot_product(A.data, b_col0, 3);
  printf("  dot(A[0,:], B[:,0]) = %.4f  (expect 58.0)\n\n", dp);

  /* ────────────────────────────────────────────
   * ITERATION 2 — Tensor Utilities + Activations
   * ──────────────────────────────────────────── */
  section("Iteration 2 — Tensor Utilities & Activations");

  Matrix Z = zeros(2, 3);
  print_matrix(&Z, "zeros(2,3)");

  Matrix O = ones(2, 3);
  print_matrix(&O, "ones(2,3)");

  Matrix R = random_matrix(3, 3);
  print_matrix(&R, "random_matrix(3,3) — values in [-1, 1]");

  /* Apply sigmoid to random matrix */
  Matrix S = random_matrix(2, 4);
  print_matrix(&S, "Before sigmoid");
  apply_function(&S, sigmoid);
  print_matrix(&S, "After sigmoid  (all values in (0,1))");

  /* Apply relu */
  Matrix Re = random_matrix(2, 4);
  print_matrix(&Re, "Before relu");
  apply_function(&Re, relu);
  print_matrix(&Re, "After relu     (negatives zeroed)");

  /* Element-wise multiply */
  Matrix EW1 = ones(2, 2);
  Matrix EW2 = ones(2, 2);
  EW2.data[0] = 5.0f;
  EW2.data[3] = 3.0f;
  Matrix EWP = elementwise_multiply(EW1, EW2);
  print_matrix(&EWP, "elementwise_multiply(ones, [5,1,1,3])");

  /* ────────────────────────────────────────────
   * ITERATION 3 — Loss Functions
   * ──────────────────────────────────────────── */
  section("Iteration 3 — Loss Functions");

  /* MSE example */
  Matrix y_true = create_matrix(1, 4);
  Matrix y_pred = create_matrix(1, 4);

  float yt[] = {1.0f, 0.0f, 1.0f, 0.0f};
  float yp[] = {0.9f, 0.1f, 0.8f, 0.2f};
  for (int i = 0; i < 4; i++) {
    y_true.data[i] = yt[i];
    y_pred.data[i] = yp[i];
  }

  float mse_val = mse(y_true, y_pred);
  printf("  y_true: [1, 0, 1, 0]\n");
  printf("  y_pred: [0.9, 0.1, 0.8, 0.2]\n");
  printf("  MSE    = %.6f\n\n", mse_val);

  /* BCE example — predictions already in (0,1) */
  float bce_val = binary_cross_entropy(y_true, y_pred);
  printf("  Binary Cross-Entropy = %.6f\n\n", bce_val);

  /* ────────────────────────────────────────────
   * ITERATION 4 — Gradient Descent Optimizer
   * ──────────────────────────────────────────── */
  section("Iteration 4 — Gradient Descent Optimizer");

  Matrix weights = ones(2, 2);
  Matrix gradients = create_matrix(2, 2);
  float g_vals[] = {0.5f, -0.3f, 0.8f, -0.1f};
  for (int i = 0; i < 4; i++)
    gradients.data[i] = g_vals[i];

  float lr = 0.1f;

  print_matrix(&weights, "Weights (before)");
  print_matrix(&gradients, "Gradients ∇W");
  printf("  Learning rate η = %.2f\n\n", lr);

  gradient_descent(&weights, gradients, lr);

  print_matrix(&weights, "Weights (after one GD step)");
  printf("  W = W - η·∇W  →  applied element-wise in place\n");

  /* ────────────────────────────────────────────
   * Cleanup
   * ──────────────────────────────────────────── */
  free_matrix(&A);
  free_matrix(&B);
  free_matrix(&C);
  free_matrix(&At);
  free_matrix(&A2);
  free_matrix(&Z);
  free_matrix(&O);
  free_matrix(&R);
  free_matrix(&S);
  free_matrix(&Re);
  free_matrix(&EW1);
  free_matrix(&EW2);
  free_matrix(&EWP);
  free_matrix(&y_true);
  free_matrix(&y_pred);
  free_matrix(&weights);
  free_matrix(&gradients);

  printf("\n══════════════════════════════════════════\n");
  printf("  All operations completed successfully.\n");
  printf("══════════════════════════════════════════\n\n");

  return 0;
}
