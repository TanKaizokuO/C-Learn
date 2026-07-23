/*
 * main.c — Test runner
 *
 * Calls each module's run_*_tests() function and prints a final summary.
 * Exits nonzero if any assertion failed, so `make test` fails CI-style.
 */

#include <stdio.h>

#include "test_utils.h"
#include "tests.h"

TestCounter g_test_counter = {0, 0};

int main(void) {
  printf("Running C-Learn test suite...\n\n");

  run_matrix_tests();
  run_loss_tests();
  run_activation_tests();
  run_gradient_tests();
  run_training_tests();

  printf("\n%d passed, %d failed\n", g_test_counter.passed,
         g_test_counter.failed);

  return g_test_counter.failed > 0 ? 1 : 0;
}
