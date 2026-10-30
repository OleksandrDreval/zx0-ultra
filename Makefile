# Central Cross-Platform (Linux / POSIX / Windows) Build System for ZX0-Ultra & Reference ZX0 Tools
# Outputs all compiled executables into $(BIN_DIR)/ at the repository root.

CC             ?= $(shell command -v gcc >/dev/null 2>&1 && echo gcc || (command -v clang >/dev/null 2>&1 && echo clang || echo cc))
UNAME_S        := $(shell uname -s 2>/dev/null || echo Unknown)

CFLAGS_COMMON  ?= -O3 -Wall -Wextra
ifeq ($(UNAME_S),Linux)
    CFLAGS_COMMON += -pthread
    ARCH_NATIVE_OK := $(shell $(CC) -march=native -E - </dev/null >/dev/null 2>&1 && echo 1 || echo 0)
    ifeq ($(ARCH_NATIVE_OK),1)
        CFLAGS_COMMON += -march=native
    endif
else
    CFLAGS_COMMON += -march=native
endif

CFLAGS_ZX0     ?= $(CFLAGS_COMMON)
CFLAGS_ULTRA   ?= $(CFLAGS_COMMON) -Werror -std=c11

OMP_OK         := $(shell $(CC) -fopenmp -E - </dev/null >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(OMP_OK),1)
    OMPFLAGS   ?= -fopenmp
else
    OMPFLAGS   ?=
endif

BIN_DIR        := bin
ZX0_SRC_DIR    := third_party/ZX0-main/src
ULTRA_SRC_DIR  := zx0_ultra

ifeq ($(OS),Windows_NT)
    EXE        := .exe
    MKDIR_BIN  = if not exist "$(BIN_DIR)" mkdir "$(BIN_DIR)"
    CLEAN_CMD  = if exist "$(BIN_DIR)" del /Q "$(BIN_DIR)\*$(EXE)"
else
    EXE        :=
    MKDIR_BIN  = mkdir -p "$(BIN_DIR)"
    CLEAN_CMD  = rm -f "$(BIN_DIR)/zx0$(EXE)" "$(BIN_DIR)/dzx0$(EXE)" "$(BIN_DIR)/zx0_ultra$(EXE)"
endif

ZX0_BIN        := $(BIN_DIR)/zx0$(EXE)
DZX0_BIN       := $(BIN_DIR)/dzx0$(EXE)
ZX0_ULTRA_BIN  := $(BIN_DIR)/zx0_ultra$(EXE)

ZX0_SRCS       := $(ZX0_SRC_DIR)/zx0.c \
                  $(ZX0_SRC_DIR)/optimize.c \
                  $(ZX0_SRC_DIR)/compress.c \
                  $(ZX0_SRC_DIR)/memory.c

DZX0_SRCS      := $(ZX0_SRC_DIR)/dzx0.c

ULTRA_SRCS     := $(ULTRA_SRC_DIR)/zx0_ultra_cli.c \
                  $(ULTRA_SRC_DIR)/ultra_engine.c \
                  $(ULTRA_SRC_DIR)/sprite_solver.c

HAS_ZX0_SUBMOD := $(wildcard $(ZX0_SRC_DIR)/zx0.c)

.PHONY: all linux zx0 dzx0 zx0_ultra test clean

ifneq ($(HAS_ZX0_SUBMOD),)
all: zx0 dzx0 zx0_ultra
else
all: zx0_ultra
endif

linux: all

zx0: $(ZX0_BIN)

dzx0: $(DZX0_BIN)

zx0_ultra: $(ZX0_ULTRA_BIN)

$(ZX0_BIN): $(ZX0_SRCS) $(ZX0_SRC_DIR)/zx0.h
	@$(MKDIR_BIN)
	$(CC) $(CFLAGS_ZX0) -I$(ZX0_SRC_DIR) -o $@ $(ZX0_SRCS)

$(DZX0_BIN): $(DZX0_SRCS)
	@$(MKDIR_BIN)
	$(CC) $(CFLAGS_ZX0) -I$(ZX0_SRC_DIR) -o $@ $(DZX0_SRCS)

$(ZX0_ULTRA_BIN): $(ULTRA_SRCS) $(ULTRA_SRC_DIR)/zx0_ultra.h
	@$(MKDIR_BIN)
	@if command -v $(CC) >/dev/null 2>&1; then \
		$(CC) $(CFLAGS_ULTRA) $(OMPFLAGS) -I$(ULTRA_SRC_DIR) -o $@ $(ULTRA_SRCS); \
	else \
		cp $(ULTRA_SRC_DIR)/zx0_ultra_linux_cli.js $@ && chmod +x $@; \
	fi

test: $(ZX0_ULTRA_BIN)
	python3 $(ULTRA_SRC_DIR)/test_ultra_suite.py

clean:
	$(CLEAN_CMD)
