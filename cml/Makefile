# Makefile — CML (C Machine Learning Library)
#
# Usage:
#   make          → build demo and train_lr binaries
#   make demo     → build original iteration demo
#   make train_lr → build linear regression demo
#   make clean    → remove compiled objects and binaries
#

CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -Iinclude
LIBS    = -lm

SRC_DIR  = src
INC_DIR  = include
EX_DIR   = examples
OBJ_DIR  = build

SOURCES  = $(SRC_DIR)/matrix.c          \
           $(SRC_DIR)/activations.c      \
           $(SRC_DIR)/loss.c             \
           $(SRC_DIR)/optimizer.c        \
           $(SRC_DIR)/linear_regression.c

OBJECTS  = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SOURCES))

TARGETS  = demo train_lr

.PHONY: all clean

all: $(OBJ_DIR) $(TARGETS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile each source file to an object
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Iterations 1-4 demo
demo: $(OBJECTS) $(EX_DIR)/demo.c
	$(CC) $(CFLAGS) $(OBJECTS) $(EX_DIR)/demo.c -o demo $(LIBS)

# Iteration 5 — Linear Regression
train_lr: $(OBJECTS) $(EX_DIR)/train_linear_regression.c
	$(CC) $(CFLAGS) $(OBJECTS) $(EX_DIR)/train_linear_regression.c -o train_lr $(LIBS)

clean:
	rm -rf $(OBJ_DIR) demo train_lr
