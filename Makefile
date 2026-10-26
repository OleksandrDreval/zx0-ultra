# Central Build System for ZX0-Ultra & Reference ZX0 Tools
# Outputs all compiled executables into $(BIN_DIR)/ at the repository root.

CC             ?= gcc
CFLAGS_COMMON  ?= -O3 -march=native -Wall -Wextra
CFLAGS_ZX0     ?= $(CFLAGS_COMMON)
CFLAGS_ULTRA   ?= $(CFLAGS_COMMON) -Werror -std=c11
OMPFLAGS       ?= -fopenmp

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

.PHONY: all zx0 dzx0 zx0_ultra clean

all: zx0 dzx0 zx0_ultra

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
	$(CC) $(CFLAGS_ULTRA) $(OMPFLAGS) -I$(ULTRA_SRC_DIR) -o $@ $(ULTRA_SRCS)

clean:
	$(CLEAN_CMD)
