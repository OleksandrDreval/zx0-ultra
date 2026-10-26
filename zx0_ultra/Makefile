# Subdirectory Makefile for zx0_ultra (outputs into ../bin)

CC       ?= gcc
CFLAGS   ?= -O3 -march=native -Wall -Wextra -Werror -std=c11
OMPFLAGS ?= -fopenmp
BIN_DIR  := ../bin

ifeq ($(OS),Windows_NT)
    EXE       := .exe
    MKDIR_BIN = if not exist "$(BIN_DIR)" mkdir "$(BIN_DIR)"
    CLEAN_CMD = if exist "$(BIN_DIR)\zx0_ultra$(EXE)" del /Q "$(BIN_DIR)\zx0_ultra$(EXE)"
else
    EXE       :=
    MKDIR_BIN = mkdir -p "$(BIN_DIR)"
    CLEAN_CMD = rm -f "$(BIN_DIR)/zx0_ultra$(EXE)"
endif

TARGET := $(BIN_DIR)/zx0_ultra$(EXE)
SRCS   := zx0_ultra_cli.c ultra_engine.c sprite_solver.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) zx0_ultra.h
	@$(MKDIR_BIN)
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $(SRCS)

clean:
	$(CLEAN_CMD)
