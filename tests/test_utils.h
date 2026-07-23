/*
 * test_utils.h — Hand-rolled test harness
 *
 * Zero-dependency assertion macros. Failures are recorded into a shared
 * counter instead of aborting, so one failing assertion doesn't hide
 * the rest of the suite.
 */

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <math.h>
#include <stdio.h>

typedef struct {
  int passed;
  int failed;
} TestCounter;

/* Defined once in tests/main.c; every test file shares this counter. */
extern TestCounter g_test_counter;

#define ASSERT_TRUE(cond)                                                    \
  do {                                                                       \
    if (cond) {                                                             \
      g_test_counter.passed++;                                              \
    } else {                                                                \
      g_test_counter.failed++;                                              \
      printf("  [FAIL] %s:%d: ASSERT_TRUE(%s)\n", __FILE__, __LINE__,       \
             #cond);                                                        \
    }                                                                       \
  } while (0)

#define ASSERT_FLOAT_NEAR(a, b, eps)                                        \
  do {                                                                       \
    float _assert_a = (float)(a);                                           \
    float _assert_b = (float)(b);                                           \
    float _assert_eps = (float)(eps);                                       \
    float _assert_diff = fabsf(_assert_a - _assert_b);                      \
    if (_assert_diff <= _assert_eps) {                                      \
      g_test_counter.passed++;                                              \
    } else {                                                                \
      g_test_counter.failed++;                                              \
      printf("  [FAIL] %s:%d: ASSERT_FLOAT_NEAR(%s, %s) -> %f vs %f "       \
             "(diff %f > eps %f)\n",                                        \
             __FILE__, __LINE__, #a, #b, _assert_a, _assert_b, _assert_diff,\
             _assert_eps);                                                  \
    }                                                                       \
  } while (0)

#endif /* TEST_UTILS_H */
