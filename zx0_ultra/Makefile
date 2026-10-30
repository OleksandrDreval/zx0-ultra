# Subdirectory Cross-Platform (Linux / POSIX / Windows) Makefile for zx0_ultra (outputs into ../bin)

CC       ?= $(shell command -v gcc >/dev/null 2>&1 && echo gcc || (command -v clang >/dev/null 2>&1 && echo clang || echo cc))
UNAME_S  := $(shell uname -s 2>/dev/null || echo Unknown)

CFLAGS   ?= -O3 -Wall -Wextra -Werror -std=c11
ifeq ($(UNAME_S),Linux)
    CFLAGS += -pthread
    ARCH_NATIVE_OK := $(shell $(CC) -march=native -E - </dev/null >/dev/null 2>&1 && echo 1 || echo 0)
    ifeq ($(ARCH_NATIVE_OK),1)
        CFLAGS += -march=native
    endif
else
    CFLAGS += -march=native
endif

OMP_OK   := $(shell $(CC) -fopenmp -E - </dev/null >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(OMP_OK),1)
    OMPFLAGS ?= -fopenmp
else
    OMPFLAGS ?=
endif

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

.PHONY: all linux test clean

all: $(TARGET)

linux: $(TARGET)

$(TARGET): $(SRCS) zx0_ultra.h
	@$(MKDIR_BIN)
	@if command -v $(CC) >/dev/null 2>&1; then \
		$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $(SRCS); \
	else \
		cp zx0_ultra_linux_cli.js $@ && chmod +x $@; \
	fi

test: $(TARGET)
	python3 test_ultra_suite.py

clean:
	$(CLEAN_CMD)
