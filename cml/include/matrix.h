/*
 * matrix.h — Core Matrix Data Structure and Operations
 *
 * Iteration 1: Linear Algebra Core
 * Iteration 2: Tensor Utilities (zeros, ones, random, elementwise, apply)
 *
 * Row-major storage: element (i,j) is at data[i * cols + j]
 */

#ifndef MATRIX_H
#define MATRIX_H

/* ─────────────────────────────────────────────
 * Core Data Structure
 * ───────────────────────────────────────────── */

typedef struct {
    int    rows;
    int    cols;
    float *data;  /* heap-allocated, row-major */
} Matrix;

/* ─────────────────────────────────────────────
 * Creation & Memory Management
 * ───────────────────────────────────────────── */

/* Allocate a matrix of given dimensions (uninitialized data). */
Matrix create_matrix(int rows, int cols);

/* Free the heap-allocated data buffer. Sets data to NULL. */
void   free_matrix(Matrix *m);

/* ─────────────────────────────────────────────
 * Initialization Helpers (Iteration 2)
 * ───────────────────────────────────────────── */

Matrix zeros(int rows, int cols);
Matrix ones(int rows, int cols);
Matrix random_matrix(int rows, int cols);   /* values in [-1, 1] */

/* ─────────────────────────────────────────────
 * Matrix Operations (Iteration 1)
 * ───────────────────────────────────────────── */

Matrix add(Matrix a, Matrix b);
Matrix subtract(Matrix a, Matrix b);
Matrix scalar_multiply(Matrix m, float scalar);
Matrix matmul(Matrix a, Matrix b);          /* a (m×k) · b (k×n) → (m×n) */
Matrix transpose(Matrix m);

/* ─────────────────────────────────────────────
 * Element-wise Operations (Iteration 2)
 * ───────────────────────────────────────────── */

Matrix elementwise_add(Matrix a, Matrix b);
Matrix elementwise_multiply(Matrix a, Matrix b);

/* Apply a unary function to every element in-place. */
void   apply_function(Matrix *m, float (*func)(float));

/* ─────────────────────────────────────────────
 * Vector Utilities
 * ───────────────────────────────────────────── */

/* Raw dot product of two float arrays of given length. */
float  dot_product(float *a, float *b, int size);

/* ─────────────────────────────────────────────
 * Helpers
 * ───────────────────────────────────────────── */

void print_matrix(const Matrix *m, const char *label);

#endif /* MATRIX_H */
