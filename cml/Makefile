# Makefile — CML (C Machine Learning Library)
#
# Usage:
#   make        → build demo binary
#   make clean  → remove compiled objects and binary
#

CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -Iinclude
LIBS    = -lm

SRC_DIR  = src
INC_DIR  = include
EX_DIR   = examples
OBJ_DIR  = build

SOURCES  = $(SRC_DIR)/matrix.c      \
           $(SRC_DIR)/activations.c  \
           $(SRC_DIR)/loss.c         \
           $(SRC_DIR)/optimizer.c

OBJECTS  = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SOURCES))

TARGET   = demo

.PHONY: all clean

all: $(OBJ_DIR) $(TARGET)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Link all objects + demo.c into the binary
$(TARGET): $(OBJECTS) $(EX_DIR)/demo.c
	$(CC) $(CFLAGS) $(OBJECTS) $(EX_DIR)/demo.c -o $(TARGET) $(LIBS)

# Compile each source file to an object
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)
