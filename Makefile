# Makefile — CML (C Machine Learning Library)
#
# Usage:
#   make              → build all binaries
#   make demo         → Iterations 1–4 demo
#   make train_lr     → Iteration 5: Linear Regression
#   make train_logistic → Iteration 6: Logistic Regression
#   make nn_demo      → Iteration 7: Neural Network
#   make test         → build and run the test suite (test_runner)
#   make clean        → remove compiled objects and binaries
#

CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -Iinclude
LIBS    = -lm

SRC_DIR  = src
INC_DIR  = include
EX_DIR   = examples
OBJ_DIR  = build
TEST_DIR = tests

SOURCES  = $(SRC_DIR)/matrix.c              \
           $(SRC_DIR)/activations.c          \
           $(SRC_DIR)/loss.c                 \
           $(SRC_DIR)/optimizer.c            \
           $(SRC_DIR)/linear_regression.c    \
           $(SRC_DIR)/logistic_regression.c  \
           $(SRC_DIR)/dense_layer.c          \
           $(SRC_DIR)/neural_network.c

OBJECTS  = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SOURCES))

TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c)

TARGETS  = demo train_lr train_logistic nn_demo

.PHONY: all clean test

all: $(OBJ_DIR) $(TARGETS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile each source file to an object
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Iterations 1–4 demo
demo: $(OBJECTS) $(EX_DIR)/demo.c
	$(CC) $(CFLAGS) $(OBJECTS) $(EX_DIR)/demo.c -o demo $(LIBS)

# Iteration 5 — Linear Regression
train_lr: $(OBJECTS) $(EX_DIR)/train_linear_regression.c
	$(CC) $(CFLAGS) $(OBJECTS) $(EX_DIR)/train_linear_regression.c -o train_lr $(LIBS)

# Iteration 6 — Logistic Regression
train_logistic: $(OBJECTS) $(EX_DIR)/train_logistic.c
	$(CC) $(CFLAGS) $(OBJECTS) $(EX_DIR)/train_logistic.c -o train_logistic $(LIBS)

# Iteration 7 — Neural Network
nn_demo: $(OBJECTS) $(EX_DIR)/neural_network_demo.c
	$(CC) $(CFLAGS) $(OBJECTS) $(EX_DIR)/neural_network_demo.c -o nn_demo $(LIBS)

clean:
	rm -rf $(OBJ_DIR) demo train_lr train_logistic nn_demo test_runner

test_titanic_c: $(OBJ_DIR) $(OBJECTS) $(EX_DIR)/titanic_logistic.c
	$(CC) $(CFLAGS) $(OBJECTS) $(EX_DIR)/titanic_logistic.c -o test_titanic_c $(LIBS)

# Test suite — builds one test_runner binary linking $(OBJECTS) against
# every tests/*.c file, then runs it.
test: $(OBJ_DIR) $(OBJECTS) $(TEST_SOURCES)
	$(CC) $(CFLAGS) -I$(TEST_DIR) $(OBJECTS) $(TEST_SOURCES) -o test_runner $(LIBS)
	./test_runner
