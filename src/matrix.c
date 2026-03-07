/*
 * matrix.c — Matrix Operations Implementation
 *
 * Covers Iteration 1 (linear algebra) and Iteration 2 (tensor utilities).
 *
 * Storage convention: element (i, j) → data[i * cols + j]  (row-major)
 */

#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ─────────────────────────────────────────────
 * Internal Helpers
 * ───────────────────────────────────────────── */

/*
 * safe_malloc — Allocate n bytes; abort on failure.
 * Keeps all callers lean — no per-call NULL checks needed.
 */
static void *safe_malloc(size_t n)
{
    void *ptr = malloc(n);
    if (!ptr) {
        fprintf(stderr, "[cml] fatal: out of memory (requested %zu bytes)\n", n);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

/*
 * check_same_dims — Validate that two matrices share dimensions.
 * Prints an error and aborts if they don't match.
 */
static void check_same_dims(Matrix a, Matrix b, const char *op)
{
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr,
                "[cml] dimension mismatch in '%s': "
                "(%d×%d) vs (%d×%d)\n",
                op, a.rows, a.cols, b.rows, b.cols);
        exit(EXIT_FAILURE);
    }
}

/* ─────────────────────────────────────────────
 * Creation & Memory Management
 * ───────────────────────────────────────────── */

/*
 * create_matrix — Allocate an uninitialized rows×cols matrix.
 */
Matrix create_matrix(int rows, int cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float *)safe_malloc((size_t)rows * cols * sizeof(float));
    return m;
}

/*
 * free_matrix — Release the data buffer and null the pointer.
 */
void free_matrix(Matrix *m)
{
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
        m->rows = 0;
        m->cols = 0;
    }
}

/* ─────────────────────────────────────────────
 * Initialization Helpers
 * ───────────────────────────────────────────── */

/* zeros — Return a rows×cols matrix filled with 0.0f */
Matrix zeros(int rows, int cols)
{
    Matrix m = create_matrix(rows, cols);
    memset(m.data, 0, (size_t)rows * cols * sizeof(float));
    return m;
}

/* ones — Return a rows×cols matrix filled with 1.0f */
Matrix ones(int rows, int cols)
{
    Matrix m = create_matrix(rows, cols);
    int n = rows * cols;
    for (int i = 0; i < n; i++)
        m.data[i] = 1.0f;
    return m;
}

/*
 * random_matrix — Return a rows×cols matrix with values in [-1, 1].
 * Seeds rand() once via a static flag.
 */
Matrix random_matrix(int rows, int cols)
{
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }

    Matrix m = create_matrix(rows, cols);
    int n = rows * cols;
    for (int i = 0; i < n; i++)
        m.data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    return m;
}

/* ─────────────────────────────────────────────
 * Matrix Operations
 * ───────────────────────────────────────────── */

/* add — Element-wise addition; requires identical dimensions. */
Matrix add(Matrix a, Matrix b)
{
    check_same_dims(a, b, "add");
    Matrix result = create_matrix(a.rows, a.cols);
    int n = a.rows * a.cols;
    for (int i = 0; i < n; i++)
        result.data[i] = a.data[i] + b.data[i];
    return result;
}

/* subtract — Element-wise subtraction; requires identical dimensions. */
Matrix subtract(Matrix a, Matrix b)
{
    check_same_dims(a, b, "subtract");
    Matrix result = create_matrix(a.rows, a.cols);
    int n = a.rows * a.cols;
    for (int i = 0; i < n; i++)
        result.data[i] = a.data[i] - b.data[i];
    return result;
}

/* scalar_multiply — Scale every element by a constant. */
Matrix scalar_multiply(Matrix m, float scalar)
{
    Matrix result = create_matrix(m.rows, m.cols);
    int n = m.rows * m.cols;
    for (int i = 0; i < n; i++)
        result.data[i] = m.data[i] * scalar;
    return result;
}

/*
 * matmul — Matrix multiplication: a (m×k) · b (k×n) → result (m×n).
 * Requires a.cols == b.rows.
 */
Matrix matmul(Matrix a, Matrix b)
{
    if (a.cols != b.rows) {
        fprintf(stderr,
                "[cml] dimension mismatch in 'matmul': "
                "a.cols=%d != b.rows=%d\n",
                a.cols, b.rows);
        exit(EXIT_FAILURE);
    }

    Matrix result = zeros(a.rows, b.cols);

    for (int i = 0; i < a.rows; i++) {
        for (int k = 0; k < a.cols; k++) {
            float a_ik = a.data[i * a.cols + k];
            for (int j = 0; j < b.cols; j++) {
                result.data[i * b.cols + j] += a_ik * b.data[k * b.cols + j];
            }
        }
    }
    return result;
}

/*
 * transpose — Return a new matrix that is the transpose of m.
 * element (i,j) of result = element (j,i) of m.
 */
Matrix transpose(Matrix m)
{
    Matrix result = create_matrix(m.cols, m.rows);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[j * m.rows + i] = m.data[i * m.cols + j];
    return result;
}

/* ─────────────────────────────────────────────
 * Element-wise Operations
 * ───────────────────────────────────────────── */

/* elementwise_add — Alias for add; kept for API clarity. */
Matrix elementwise_add(Matrix a, Matrix b)
{
    return add(a, b);
}

/* elementwise_multiply — Hadamard product; requires same dimensions. */
Matrix elementwise_multiply(Matrix a, Matrix b)
{
    check_same_dims(a, b, "elementwise_multiply");
    Matrix result = create_matrix(a.rows, a.cols);
    int n = a.rows * a.cols;
    for (int i = 0; i < n; i++)
        result.data[i] = a.data[i] * b.data[i];
    return result;
}

/*
 * apply_function — Apply a unary float→float function to every element
 * of m in place.  Compatible with relu, sigmoid, or any custom function.
 *
 * Example:
 *   apply_function(&m, sigmoid);
 */
void apply_function(Matrix *m, float (*func)(float))
{
    int n = m->rows * m->cols;
    for (int i = 0; i < n; i++)
        m->data[i] = func(m->data[i]);
}

/* ─────────────────────────────────────────────
 * Vector Utilities
 * ───────────────────────────────────────────── */

/*
 * dot_product — Sum of element-wise products of two float arrays.
 * a and b must both have at least `size` elements.
 */
float dot_product(float *a, float *b, int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
        sum += a[i] * b[i];
    return sum;
}

/* ─────────────────────────────────────────────
 * Print Helper
 * ───────────────────────────────────────────── */

/*
 * print_matrix — Pretty-print a matrix to stdout.
 * label can be NULL.
 */
void print_matrix(const Matrix *m, const char *label)
{
    if (label)
        printf("%s (%d×%d):\n", label, m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        printf("  [ ");
        for (int j = 0; j < m->cols; j++) {
            printf("%8.4f", m->data[i * m->cols + j]);
            if (j < m->cols - 1) printf(", ");
        }
        printf(" ]\n");
    }
    printf("\n");
}
