/*
 * tests.h — Prototypes for each module's test entry point.
 *
 * Each tests/test_*.c file defines one run_*_tests() function; main.c
 * calls all of them and prints a final summary.
 */

#ifndef TESTS_H
#define TESTS_H

void run_matrix_tests(void);
void run_loss_tests(void);
void run_activation_tests(void);
void run_gradient_tests(void);
void run_training_tests(void);

#endif /* TESTS_H */
