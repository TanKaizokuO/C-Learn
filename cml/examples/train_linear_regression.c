/*
 * train_linear_regression.c — Iteration 5 Demo
 *
 * Demonstrates a complete ML training pipeline in C:
 *
 *   1. Generate a synthetic dataset: y = 3x + 2 + noise
 *   2. Initialise a LinearRegression model
 *   3. Train with gradient descent
 *   4. Print learned weight and bias
 *   5. Run a quick forward-pass validation
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "linear_regression.h"
#include "matrix.h"

/* ─── Configuration ─────────────────────────── */
#define N_SAMPLES 100    /* number of training points      */
#define N_FEATURES 1     /* single feature  (1-D input)    */
#define TRUE_WEIGHT 3.0f /* ground-truth slope             */
#define TRUE_BIAS 2.0f   /* ground-truth intercept         */
#define NOISE_SCALE 0.1f /* Gaussian noise magnitude       */
#define EPOCHS 500
#define LEARNING_RATE 0.01f

/* ─── Minimal pseudo-Gaussian noise ─────────── */
/* Box-Muller approximation using rand() */
static float randn(void) {
  float u = ((float)rand() / (float)RAND_MAX) + 1e-8f;
  float v = ((float)rand() / (float)RAND_MAX);
  return sqrtf(-2.0f * logf(u)) * cosf(2.0f * 3.14159265f * v);
}

int main(void) {
  srand((unsigned int)time(NULL));

  /* ────────────────────────────────────────────
   * 1. Synthetic dataset:  y = 3x + 2 + ε
   * ──────────────────────────────────────────── */
  printf("══════════════════════════════════════════\n");
  printf("  Iteration 5 — Linear Regression in C   \n");
  printf("══════════════════════════════════════════\n\n");

  printf("Dataset:  y = %.1fx + %.1f  +  noise(σ=%.2f)\n", TRUE_WEIGHT,
         TRUE_BIAS, NOISE_SCALE);
  printf("Samples:  %d\n", N_SAMPLES);
  printf("Epochs:   %d   |   lr = %.4f\n\n", EPOCHS, LEARNING_RATE);

  /* X: inputs in [-1, 1], y: noisy labels */
  Matrix X = create_matrix(N_SAMPLES, N_FEATURES);
  Matrix y = create_matrix(N_SAMPLES, 1);

  for (int i = 0; i < N_SAMPLES; i++) {
    float xi = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    X.data[i] = xi;
    y.data[i] = TRUE_WEIGHT * xi + TRUE_BIAS + NOISE_SCALE * randn();
  }

  /* ────────────────────────────────────────────
   * 2. Create model and train
   * ──────────────────────────────────────────── */
  LinearRegression model = create_linear_regression(N_FEATURES);

  train_linear_regression(&model, X, y, EPOCHS, LEARNING_RATE);

  /* ────────────────────────────────────────────
   * 3. Print learned parameters
   * ──────────────────────────────────────────── */
  printf("══════════════════════════════════════════\n");
  printf("  Learned Parameters\n");
  printf("══════════════════════════════════════════\n");
  printf("  Learned Weight : %8.4f   (true: %.1f)\n", model.weights.data[0],
         TRUE_WEIGHT);
  printf("  Learned Bias   : %8.4f   (true: %.1f)\n\n", model.bias, TRUE_BIAS);

  /* ────────────────────────────────────────────
   * 4. Quick forward-pass validation
   * ──────────────────────────────────────────── */
  printf("══════════════════════════════════════════\n");
  printf("  Predictions on 5 sample points\n");
  printf("══════════════════════════════════════════\n");
  printf("  %8s  %8s  %8s\n", "x", "y_true", "y_pred");
  printf("  %-8s  %-8s  %-8s\n", "--------", "--------", "--------");

  for (int i = 0; i < 5; i++) {
    /* single-sample matrix */
    Matrix x_i = create_matrix(1, N_FEATURES);
    x_i.data[0] = X.data[i];

    Matrix pred = predict(&model, x_i);

    printf("  %8.4f  %8.4f  %8.4f\n", X.data[i], y.data[i], pred.data[0]);

    free_matrix(&x_i);
    free_matrix(&pred);
  }
  printf("\n");

  /* ────────────────────────────────────────────
   * 5. Cleanup
   * ──────────────────────────────────────────── */
  free_matrix(&X);
  free_matrix(&y);
  free_linear_regression(&model);

  printf("Training complete. Model memory freed.\n\n");
  return 0;
}
