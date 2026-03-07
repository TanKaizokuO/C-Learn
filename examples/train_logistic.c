/*
 * train_logistic.c — Logistic Regression Demo
 *
 * Iteration 6: Binary classification on a synthetic linearly separable dataset.
 *
 * Dataset:  2 features, 200 samples
 *   Class 1  if  x[0] + x[1] >  0.0
 *   Class 0  if  x[0] + x[1] <= 0.0
 *
 * Pipeline:
 *   1. Generate dataset
 *   2. Train logistic regression
 *   3. Compute and print accuracy
 *   4. Show sample predictions
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "logistic_regression.h"
#include "matrix.h"

/* ─── Configuration ─────────────────────────── */
#define N_SAMPLES 200
#define N_FEATURES 2
#define EPOCHS 1000
#define LEARNING_RATE 0.1f

int main(void) {
  srand((unsigned int)time(NULL));

  /* ────────────────────────────────────────────
   * 1. Generate linearly separable dataset
   *
   *   x[0], x[1] ∈ [-1, 1]  (uniform)
   *   y = 1  if  x[0] + x[1] > 0,  else y = 0
   * ──────────────────────────────────────────── */
  printf("══════════════════════════════════════════\n");
  printf("  Iteration 6 — Logistic Regression in C \n");
  printf("══════════════════════════════════════════\n\n");

  printf("Dataset : 2 features, %d samples\n", N_SAMPLES);
  printf("Rule    : y = 1  if  x0 + x1 > 0  else  0\n");
  printf("Epochs  : %d  |  lr = %.2f\n\n", EPOCHS, LEARNING_RATE);

  Matrix X = create_matrix(N_SAMPLES, N_FEATURES);
  Matrix y = create_matrix(N_SAMPLES, 1);

  for (int i = 0; i < N_SAMPLES; i++) {
    float x0 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    float x1 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    X.data[i * N_FEATURES + 0] = x0;
    X.data[i * N_FEATURES + 1] = x1;
    y.data[i] = (x0 + x1 > 0.0f) ? 1.0f : 0.0f;
  }

  /* ────────────────────────────────────────────
   * 2. Create model and train
   * ──────────────────────────────────────────── */
  LogisticRegression model = create_logistic_regression(N_FEATURES);
  train_logistic_regression(&model, X, y, EPOCHS, LEARNING_RATE);

  /* ────────────────────────────────────────────
   * 3. Compute accuracy on training set
   *    Threshold: y_pred > 0.5 → class 1, else 0
   * ──────────────────────────────────────────── */
  Matrix y_pred = predict_logistic(&model, X);

  int correct = 0;
  for (int i = 0; i < N_SAMPLES; i++) {
    float predicted_class = (y_pred.data[i] > 0.5f) ? 1.0f : 0.0f;
    if (predicted_class == y.data[i])
      correct++;
  }
  float accuracy = 100.0f * (float)correct / (float)N_SAMPLES;

  printf("══════════════════════════════════════════\n");
  printf("  Results\n");
  printf("══════════════════════════════════════════\n");
  printf("  Learned Weight[0] : %8.4f\n", model.weights.data[0]);
  printf("  Learned Weight[1] : %8.4f\n", model.weights.data[1]);
  printf("  Learned Bias      : %8.4f\n\n", model.bias);
  printf("  Accuracy : %d / %d = %.1f%%\n\n", correct, N_SAMPLES, accuracy);

  /* ────────────────────────────────────────────
   * 4. Sample predictions
   * ──────────────────────────────────────────── */
  printf("══════════════════════════════════════════\n");
  printf("  Sample Predictions (first 8)\n");
  printf("══════════════════════════════════════════\n");
  printf("  %6s  %6s  %6s  %6s  %6s\n", "x0", "x1", "y_true", "y_prob",
         "y_hat");
  printf("  %-6s  %-6s  %-6s  %-6s  %-6s\n", "------", "------", "------",
         "------", "------");

  for (int i = 0; i < 8; i++) {
    float x0 = X.data[i * N_FEATURES + 0];
    float x1 = X.data[i * N_FEATURES + 1];
    float label = y.data[i];
    float prob = y_pred.data[i];
    int yhat = (prob > 0.5f) ? 1 : 0;
    printf("  %6.3f  %6.3f  %6.0f  %6.4f  %6d\n", x0, x1, label, prob, yhat);
  }
  printf("\n");

  /* ────────────────────────────────────────────
   * 5. Cleanup
   * ──────────────────────────────────────────── */
  free_matrix(&X);
  free_matrix(&y);
  free_matrix(&y_pred);
  free_logistic_regression(&model);

  printf("Training complete. Model memory freed.\n\n");
  return 0;
}
