#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logistic_regression.h"
#include "matrix.h"

#define MAX_LINE_LENGTH 1024
#define N_SAMPLES 891
#define N_FEATURES 16
#define TRAIN_SAMPLES 712
#define TEST_SAMPLES 179

#define EPOCHS 1000
#define LEARNING_RATE 0.1f

#include <time.h>

int main(void) {
  clock_t start_time = clock();

  printf("Loading data from preprocessedDB.csv...\n");
  FILE *file = fopen("preprocessedDB.csv", "r");
  if (!file) {
    perror("Failed to open preprocessedDB.csv");
    return 1;
  }

  Matrix X = create_matrix(N_SAMPLES, N_FEATURES);
  Matrix y = create_matrix(N_SAMPLES, 1);

  char line[MAX_LINE_LENGTH];
  // Skip header
  if (!fgets(line, MAX_LINE_LENGTH, file)) {
    printf("Failed to read header.\n");
    return 1;
  }

  int row = 0;
  while (fgets(line, MAX_LINE_LENGTH, file) && row < N_SAMPLES) {
    char *token = strtok(line, ",");
    if (token) {
      y.data[row] = atof(token);
      int col = 0;
      token = strtok(NULL, ",");
      while (token && col < N_FEATURES) {
        X.data[row * N_FEATURES + col] = atof(token);
        token = strtok(NULL, ",");
        col++;
      }
    }
    row++;
  }
  fclose(file);

  printf("Loaded %d rows.\n", row);

  // Naive split (first 712 for train, rest for test)
  // Note: This won't match Python's random state 42 perfectly, but it's close
  // enough to verify the model.
  Matrix X_train = create_matrix(TRAIN_SAMPLES, N_FEATURES);
  Matrix y_train = create_matrix(TRAIN_SAMPLES, 1);
  Matrix X_test = create_matrix(TEST_SAMPLES, N_FEATURES);
  Matrix y_test = create_matrix(TEST_SAMPLES, 1);

  for (int i = 0; i < TRAIN_SAMPLES; i++) {
    y_train.data[i] = y.data[i];
    for (int j = 0; j < N_FEATURES; j++) {
      X_train.data[i * N_FEATURES + j] = X.data[i * N_FEATURES + j];
    }
  }

  for (int i = 0; i < TEST_SAMPLES; i++) {
    y_test.data[i] = y.data[TRAIN_SAMPLES + i];
    for (int j = 0; j < N_FEATURES; j++) {
      X_test.data[i * N_FEATURES + j] =
          X.data[(TRAIN_SAMPLES + i) * N_FEATURES + j];
    }
  }

  printf("Training Logistic Regression model on %d samples for %d epochs "
         "(LR=%.2f)...\n",
         TRAIN_SAMPLES, EPOCHS, LEARNING_RATE);
  LogisticRegression model = create_logistic_regression(N_FEATURES);
  train_logistic_regression(&model, X_train, y_train, EPOCHS, LEARNING_RATE);

  printf("Evaluating precision on %d test samples...\n", TEST_SAMPLES);
  Matrix y_pred = predict_logistic(&model, X_test);

  int correct = 0;
  int tp = 0, tn = 0, fp = 0, fn = 0;

  for (int i = 0; i < TEST_SAMPLES; i++) {
    float predicted_class = (y_pred.data[i] > 0.5f) ? 1.0f : 0.0f;
    float actual_class = y_test.data[i];

    if (predicted_class == actual_class) {
      correct++;
      if (actual_class == 1.0f)
        tp++;
      else
        tn++;
    } else {
      if (predicted_class == 1.0f)
        fp++;
      else
        fn++;
    }
  }

  float accuracy = (float)correct / TEST_SAMPLES;

  printf("\nAccuracy: %.4f\n\n", accuracy);
  printf("Confusion Matrix:\n");
  printf("[[%d %d]\n", tn, fp);
  printf(" [%d %d]]\n\n", fn, tp);

  // Cleanup
  free_matrix(&X);
  free_matrix(&y);
  free_matrix(&X_train);
  free_matrix(&y_train);
  free_matrix(&X_test);
  free_matrix(&y_test);
  free_matrix(&y_pred);
  free_logistic_regression(&model);

  clock_t end_time = clock();
  double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  printf("Execution time: %.5f seconds\n", time_taken);

  return 0;
}
