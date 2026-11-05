#!/usr/bin/env python3
"""
Cryptographic Unicorn x86-16 Verification & Honest Anti-Overfitting Benchmark Suite for ZX0-Ultra.

Assembles all 11 x86-16 decompressors (67B..128B) with NASM, compresses diverse real-world datasets in
datasets/ plus 4 algorithmic edge/trap regression cases and multi-asset superstring/wildcard packs
(7-sprite exhaustive + 10-asset Held-Karp ATSP), and verifies every unpacked buffer in Unicorn
16-bit real mode via SHA-256 memory hashing and pre/post-buffer canary guards.

Author: Oleksandr Dreval (MIT License)
SPDX-License-Identifier: MIT
"""

import argparse
import concurrent.futures
import hashlib
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

from unicorn import Uc, UC_ARCH_X86, UC_MODE_16
from unicorn.x86_const import (
    UC_X86_REG_AX,
    UC_X86_REG_BX,
    UC_X86_REG_CX,
    UC_X86_REG_DX,
    UC_X86_REG_SI,
    UC_X86_REG_DI,
    UC_X86_REG_BP,
    UC_X86_REG_SP,
)

ULTRA_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_DIR = os.path.abspath(os.path.join(ULTRA_DIR, ".."))
BIN_DIR = os.path.join(REPO_DIR, "bin")
DATASETS_DIR = os.path.join(REPO_DIR, "datasets")
ZX_THINGS_DIR = os.path.join(REPO_DIR, "ZX things")
ASM_DIR = os.path.join(ULTRA_DIR, "asm")

sys.path.insert(0, REPO_DIR)

# 4bpp sprite data for the pommel component of the sword.
SWORD_POMMEL = bytes([
    0x00, 0x11,
    0x11, 0x14,
    0x13, 0x42,
    0x13, 0x23,
    0x12, 0x33,
    0x12, 0x33,
    0x11, 0x12,
    0x00, 0x11,
])


# 4bpp sprite data for the grip component of the sword.
SWORD_GRIP = bytes([
    0x11, 0x11,
    0x76, 0x61,
    0x66, 0x17,
    0x61, 0x76,
    0x17, 0x66,
    0x11, 0x11,
])


# 4bpp sprite data for the guard component of the sword.
SWORD_GUARD = bytes([
    0x01, 0x11,
    0x01, 0x41,
    0x11, 0x41,
    0x14, 0x41,
    0x14, 0x41,
    0x14, 0x41,
    0x14, 0x41,
    0x14, 0x41,
    0x14, 0x41,
    0x13, 0x31,
    0x13, 0x31,
    0x13, 0x31,
    0x13, 0x31,
    0x13, 0x31,
    0x13, 0x31,
    0x11, 0x31,
    0x01, 0x31,
    0x01, 0x11,
])


# 4bpp sprite data for a segment of the sword's blade.
SWORD_BLADE = bytes([
    0x11,
    0x44,
    0x44,
    0x22,
    0x44,
    0x22,
    0x22,
    0x11,
])


# 4bpp sprite data for the tip component of the sword.
SWORD_TIP = bytes([
    0x11, 0x00, 0x00, 0x00,
    0x41, 0x11, 0x10, 0x00,
    0x44, 0x44, 0x11, 0x11,
    0x44, 0x44, 0x44, 0x41,
    0x22, 0x22, 0x22, 0x21,
    0x22, 0x22, 0x11, 0x11,
    0x21, 0x11, 0x10, 0x00,
    0x11, 0x00, 0x00, 0x00,
])


# 4bpp sprite data for the top half of the skull.
HALF_SKULL_SPRITE = bytes([
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x11, 0x11,
    0x00, 0x00, 0x00, 0x00, 0x11, 0x11, 0x33, 0x33,
    0x00, 0x00, 0x01, 0x11, 0x13, 0x33, 0x34, 0x44,
    0x00, 0x00, 0x11, 0x33, 0x34, 0x44, 0x44, 0x44,
    0x00, 0x01, 0x13, 0x33, 0x44, 0x44, 0x44, 0x44,
    0x00, 0x11, 0x33, 0x34, 0x44, 0x44, 0x44, 0x44,
    0x01, 0x13, 0x33, 0x34, 0x44, 0x44, 0x44, 0x44,
    0x01, 0x33, 0x34, 0x44, 0x44, 0x44, 0x44, 0x44,
    0x01, 0x33, 0x34, 0x44, 0x44, 0x44, 0x44, 0x44,
    0x01, 0x33, 0x34, 0x44, 0x44, 0x44, 0x44, 0x44,
    0x11, 0x33, 0x34, 0x44, 0x44, 0x44, 0x44, 0x44,
    0x13, 0x32, 0x33, 0x44, 0x44, 0x44, 0x44, 0x44,
    0x12, 0x32, 0x33, 0x44, 0x44, 0x44, 0x44, 0x44,
    0x12, 0x33, 0x33, 0x33, 0x44, 0x44, 0x44, 0x44,
    0x12, 0x22, 0x33, 0x44, 0x44, 0x33, 0x44, 0x44,
    0x12, 0x21, 0x44, 0x33, 0x33, 0x33, 0x33, 0x44,
    0x12, 0x21, 0x43, 0x33, 0x22, 0x22, 0x33, 0x33,
    0x12, 0x23, 0x32, 0x22, 0x22, 0x22, 0x23, 0x33,
    0x12, 0x23, 0x22, 0x22, 0x22, 0x22, 0x23, 0x33,
    0x12, 0x23, 0x22, 0x22, 0x22, 0x22, 0x23, 0x34,
    0x12, 0x23, 0x22, 0x22, 0x22, 0x22, 0x23, 0x44,
    0x11, 0x24, 0x22, 0x22, 0x22, 0x22, 0x34, 0x44,
    0x13, 0x34, 0x32, 0x22, 0x22, 0x22, 0x34, 0x44,
    0x13, 0x34, 0x43, 0x33, 0x23, 0x33, 0x34, 0x42,
    0x13, 0x33, 0x44, 0x44, 0x44, 0x44, 0x44, 0x22,
    0x13, 0x33, 0x33, 0x33, 0x32, 0x23, 0x44, 0x22,
    0x11, 0x33, 0x33, 0x33, 0x32, 0x23, 0x43, 0x22,
    0x01, 0x11, 0x13, 0x33, 0x22, 0x23, 0x32, 0x22,
    0x00, 0x00, 0x11, 0x11, 0x22, 0x34, 0x42, 0x23,
    0x00, 0x00, 0x00, 0x01, 0x12, 0x34, 0x43, 0x33,
    0x00, 0x00, 0x00, 0x00, 0x12, 0x34, 0x33, 0x44,
    0x00, 0x00, 0x00, 0x00, 0x12, 0x34, 0x43, 0x43,
    0x00, 0x00, 0x00, 0x00, 0x13, 0x44, 0x43, 0x43,
    0x00, 0x00, 0x00, 0x00, 0x11, 0x11, 0x13, 0x13,
    0x00, 0x00, 0x00, 0x00, 0x11, 0x34, 0x31, 0x41,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x11, 0x11, 0x11,
])


# 4bpp sprite data for the lower jaw of the skull.
HALF_JAW_SPRITE = bytes([
    0x01, 0x11, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x23, 0x21, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x23, 0x21, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x13, 0x21, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x13, 0x21, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x13, 0x21, 0x10, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x13, 0x32, 0x11, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x13, 0x32, 0x11, 0x11, 0x00, 0x00, 0x00,
    0x00, 0x13, 0x34, 0x21, 0x31, 0x11, 0x11, 0x11,
    0x00, 0x13, 0x34, 0x22, 0x13, 0x32, 0x42, 0x41,
    0x00, 0x13, 0x34, 0x22, 0x21, 0x32, 0x32, 0x41,
    0x00, 0x12, 0x33, 0x33, 0x23, 0x44, 0x11, 0x31,
    0x00, 0x11, 0x22, 0x44, 0x33, 0x33, 0x44, 0x14,
    0x00, 0x01, 0x12, 0x44, 0x44, 0x33, 0x34, 0x33,
    0x00, 0x00, 0x12, 0x24, 0x44, 0x44, 0x44, 0x44,
    0x00, 0x00, 0x11, 0x12, 0x44, 0x44, 0x44, 0x44,
    0x00, 0x00, 0x00, 0x12, 0x34, 0x44, 0x44, 0x44,
    0x00, 0x00, 0x00, 0x11, 0x13, 0x44, 0x44, 0x44,
    0x00, 0x00, 0x00, 0x00, 0x11, 0x24, 0x33, 0x34,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x11, 0x11, 0x11,
])


def find_nasm() -> str:
    which_nasm = shutil.which("nasm")
    if which_nasm:
        return which_nasm
    common_paths = [
        r"C:\Program Files\NASM\nasm.exe",
        r"C:\Program Files (x86)\NASM\nasm.exe",
        "/usr/bin/nasm",
        "/usr/local/bin/nasm",
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p
    return "nasm"


NASM_EXE = find_nasm()
ZX0_ULTRA_EXE = os.path.join(BIN_DIR, "zx0_ultra" if os.name != "nt" else "zx0_ultra.exe")
ZX0_ORIG_EXE = os.path.join(BIN_DIR, "zx0" if os.name != "nt" else "zx0.exe")

CODE_LOAD_ADDR = 0x0100
HALT_STUB_ADDR = 0x00F0
STREAM_LOAD_ADDR = 0x0200
DEFAULT_OUTPUT_ADDR = 0x4000
INITIAL_STACK_PTR = 0xFFFE
SEGMENT_MEM_SIZE = 0x10000
CANARY_GUARD_BYTES = 16
MAX_EMULATION_STEPS = 25_000_000


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assemble_flat_binary(asm_path: str, out_bin_path: str) -> bytes:
    subprocess.run([NASM_EXE, "-f", "bin", asm_path, "-o", out_bin_path], check=True)
    with open(out_bin_path, "rb") as fp:
        return fp.read()


def emulate_x86_decompressor(decoder_code: bytes, compressed_stream: bytes, expected_len: int) -> bytes:
    mu = Uc(UC_ARCH_X86, UC_MODE_16)
    mu.mem_map(0x0000, SEGMENT_MEM_SIZE)

    min_out_addr = ((STREAM_LOAD_ADDR + len(compressed_stream) + CANARY_GUARD_BYTES + 0x1F) & ~0x0F)
    out_addr = max(DEFAULT_OUTPUT_ADDR, min_out_addr)
    if out_addr + expected_len + CANARY_GUARD_BYTES > INITIAL_STACK_PTR - 64:
        out_addr = min_out_addr
    assert out_addr + expected_len + CANARY_GUARD_BYTES <= INITIAL_STACK_PTR - 32, (
        f"64KB segment overflow: stream={len(compressed_stream)}B, raw={expected_len}B"
    )

    total_canary_span = CANARY_GUARD_BYTES + expected_len + CANARY_GUARD_BYTES
    canary = bytes(((i * 73 + 0x5A) & 0xFF) for i in range(total_canary_span))
    mu.mem_write(out_addr - CANARY_GUARD_BYTES, canary)

    mu.mem_write(CODE_LOAD_ADDR, decoder_code)
    mu.mem_write(HALT_STUB_ADDR, b"\xf4")  # hlt
    mu.mem_write(STREAM_LOAD_ADDR, compressed_stream)

    # Standard MS-DOS .COM startup register state (AX=0, BX=0, CX=0x00FF, SI=stream, DI=out)
    mu.reg_write(UC_X86_REG_AX, 0x0000)
    mu.reg_write(UC_X86_REG_BX, 0x0000)
    mu.reg_write(UC_X86_REG_CX, 0x00FF)
    mu.reg_write(UC_X86_REG_DX, 0x1234)  # Non-zero DX to verify no hidden DH/DL=0 assumptions
    mu.reg_write(UC_X86_REG_BP, 0x091C)  # Non-zero BP to verify clean BP initialization
    mu.reg_write(UC_X86_REG_SI, STREAM_LOAD_ADDR)
    mu.reg_write(UC_X86_REG_DI, out_addr)
    mu.reg_write(UC_X86_REG_SP, INITIAL_STACK_PTR)
    mu.mem_write(INITIAL_STACK_PTR, struct.pack("<H", HALT_STUB_ADDR))

    mu.emu_start(CODE_LOAD_ADDR, HALT_STUB_ADDR, count=MAX_EMULATION_STEPS)

    final_di = mu.reg_read(UC_X86_REG_DI)
    final_sp = mu.reg_read(UC_X86_REG_SP)
    assert final_sp == ((INITIAL_STACK_PTR + 2) & 0xFFFF), f"Stack imbalance: SP={final_sp:#x}"
    assert final_di == out_addr + expected_len, (
        f"DI end pointer mismatch: got {final_di:#x}, expected {out_addr + expected_len:#x}"
    )

    guard_before = bytes(mu.mem_read(out_addr - CANARY_GUARD_BYTES, CANARY_GUARD_BYTES))
    guard_after = bytes(mu.mem_read(out_addr + expected_len, CANARY_GUARD_BYTES))
    assert guard_before == canary[:CANARY_GUARD_BYTES], "Output buffer underrun detected!"
    assert guard_after == canary[CANARY_GUARD_BYTES + expected_len :], "Output buffer overrun detected!"

    return bytes(mu.mem_read(out_addr, expected_len))


def find_asset(corpus: list[tuple[str, bytes]], pattern: str) -> bytes:
    for name, data in corpus:
        if pattern in name:
            return data
    raise KeyError(f"Asset matching pattern '{pattern}' not found in corpus")


# Canonical 20 representative datasets covering all architectural modalities
CANONICAL_20_PATTERNS = [
    "4bit_nibble_tiles",
    "6bit_indexed_sprites",
    "alt_stride_rle",
    "arxiv_api_json",
    "asm_8088_src",
    "audio_16bit_pcm",
    "bitmap_font",
    "bytecode_vm",
    "canterbury_alice_2048",
    "canterbury_ptt5_fax",
    "cga_2bpp_tile",
    "ega_planar_tiles",
    "hybrid_code_pal",
    "incompressible_sha256",
    "opl2_music",
    "pe_headers",
    "real_dos_com",
    "sin16_lut",
    "structs_1024",
    "vga_rgb_pal",
    "x86_code_1142",
]


def load_honest_datasets_and_edge_cases(
    run_all: bool = False,
    limit: int | None = None,
    filter_patterns: list[str] | None = None,
) -> tuple[list[tuple[str, bytes]], list[tuple[str, bytes]]]:
    all_files: list[tuple[str, bytes]] = []
    if os.path.isdir(DATASETS_DIR):
        for fname in sorted(os.listdir(DATASETS_DIR)):
            fpath = os.path.join(DATASETS_DIR, fname)
            if os.path.isfile(fpath) and fname.endswith(".bin"):
                label = fname[:-4]
                with open(fpath, "rb") as fp:
                    all_files.append((label, fp.read()))

    if filter_patterns:
        selected_files = [
            (label, raw)
            for label, raw in all_files
            if any(p.lower() in label.lower() for p in filter_patterns)
        ]
    elif run_all:
        selected_files = all_files
    else:
        # Default: select canonical representative files (up to 20)
        selected_files = []
        for pat in CANONICAL_20_PATTERNS:
            match = next(((lbl, raw) for lbl, raw in all_files if pat in lbl), None)
            if match and match not in selected_files:
                selected_files.append(match)
        if not selected_files:
            selected_files = all_files[:20]

    if limit and limit > 0:
        selected_files = selected_files[:limit]

    # Additional algorithmic edge & regression cases (always validated)
    edge_cases: list[tuple[str, bytes]] = []
    edge_cases.append(("edge_01_1byte", b"\x42"))

    trap_prefix = b"ABCDEFGHIJKLMNO" + b"ABCDEFGHIJKLMNO"
    trap_lit1 = b"1234567890abcdef"
    trap_match = b"2"
    trap_lit2 = b"ghijklmnpqrstuvw"
    trap_trail = bytes([(x * 17 + 3) % 256 for x in range(16)])
    edge_cases.append(("edge_02_heuristic_trap", trap_prefix + trap_lit1 + trap_match + trap_lit2 + trap_trail))

    # Hybrid code + RGB table testing Kadane O(N) arbitrary split point (750B code + 450B RGB table)
    try:
        code_src = find_asset(all_files, "x86_code")[:750]
    except KeyError:
        code_src = bytes((x * 37 + 11) % 256 for x in range(750))

    pal_tail = bytes(((i >> 2) & 63) if (c == 0) else (((i * 3) >> 3) & 63 if c == 1 else ((63 - (i >> 2)) & 63))
                     for i in range(150) for c in range(3))
    edge_cases.append(("edge_03_kadane_750_450", code_src + pal_tail))

    # Trailing 0xE8 boundary safety test (ensures X86E8_84B safely falls back when raw[-2:] contains 0xE8)
    edge_cases.append(("edge_04_trailing_e8_guard", code_src[:254] + b"\xe8\xe8"))

    return selected_files, edge_cases, all_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Cryptographic Unicorn Verification Suite for ZX0-Ultra")
    parser.add_argument("--all", action="store_true", help="Run verification across all 114 datasets in datasets/")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of corpus datasets to verify")
    parser.add_argument("filters", nargs="*", help="Optional dataset name substrings to filter by")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        dec_x86_main = assemble_flat_binary(
            os.path.join(ZX_THINGS_DIR, "unzx0_x86.asm"), os.path.join(tmpdir, "dx86.bin")
        )
        dec_67b = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_67b.asm"), os.path.join(tmpdir, "d67.bin")
        )
        dec_77b = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_adpk_77b.asm"), os.path.join(tmpdir, "d77.bin")
        )
        dec_78b = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_adpk_78b.asm"), os.path.join(tmpdir, "d78.bin")
        )
        dec_80b = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_strd_80b.asm"), os.path.join(tmpdir, "d80.bin")
        )
        dec_compat82 = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_compat_82b.asm"), os.path.join(tmpdir, "dc82.bin")
        )
        dec_strd82 = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_strd_82b.asm"), os.path.join(tmpdir, "ds82.bin")
        )
        dec_x86e8_84 = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_x86e8_84b.asm"), os.path.join(tmpdir, "de84.bin")
        )
        dec_univ_93 = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_univ_93b.asm"), os.path.join(tmpdir, "du93.bin")
        )
        dec_hybrid_102 = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_hybrid_102b.asm"), os.path.join(tmpdir, "dh102.bin")
        )
        dec_hybrid_104 = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_hybrid_104b.asm"), os.path.join(tmpdir, "dh104.bin")
        )
        dec_pro_110 = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_pro_110b.asm"), os.path.join(tmpdir, "dp110.bin")
        )
        dec_pro_111 = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_pro_111b.asm"), os.path.join(tmpdir, "dp111.bin")
        )
        dec_ultra_128 = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_ultra_128b.asm"), os.path.join(tmpdir, "du128.bin")
        )

        print("=" * 162)
        print("ASSEMBLED x86-16 DECOMPRESSOR FOOTPRINTS (NASM FLAT BINARIES vs 70B BASELINE & 128B MAX BUDGET)")
        print("=" * 162)
        print(f"  ZX things/unzx0_x86.asm          : {len(dec_x86_main):3d} bytes ({len(dec_x86_main)-70:+d}B vs 70B, 100% ZX0 v2 reference)")
        print(f"  zx0_ultra/asm/unzx0_67b.asm      : {len(dec_67b):3d} bytes ({len(dec_67b)-70:+d}B vs 70B, 100% ZX0 v2 compatible)")
        print(f"  zx0_ultra/asm/unzx0_adpk_77b.asm : {len(dec_77b):3d} bytes ({len(dec_77b)-70:+d}B vs 70B, Adaptive-k 1..12 offset LSB, -1B vs 78B)")
        print(f"  zx0_ultra/asm/unzx0_adpk_78b.asm : {len(dec_78b):3d} bytes ({len(dec_78b)-70:+d}B vs 70B, Adaptive-k 1..12 offset LSB)")
        print(f"  zx0_ultra/asm/unzx0_strd_80b.asm : {len(dec_80b):3d} bytes ({len(dec_80b)-70:+d}B vs 70B, Order-1 16-bit Stride-Delta)")
        print(f"  zx0_ultra/asm/unzx0_compat_82b.asm: {len(dec_compat82):3d} bytes ({len(dec_compat82)-70:+d}B vs 70B, 0B-ovh ZX0 compat + Stride)")
        print(f"  zx0_ultra/asm/unzx0_strd_82b.asm : {len(dec_strd82):3d} bytes ({len(dec_strd82)-70:+d}B vs 70B, Multi-Pass Order-1..5 Stride)")
        print(f"  zx0_ultra/asm/unzx0_x86e8_84b.asm: {len(dec_x86e8_84):3d} bytes ({len(dec_x86e8_84)-70:+d}B vs 70B, 0B-ovh ZX0 compat + x86 E8 Call)")
        print(f"  zx0_ultra/asm/unzx0_univ_93b.asm : {len(dec_univ_93):3d} bytes ({len(dec_univ_93)-70:+d}B vs 70B, 0B-ovh ZX0 compat + Stride + E8)")
        print(f"  zx0_ultra/asm/unzx0_hybrid_102b.asm: {len(dec_hybrid_102):3d} bytes ({len(dec_hybrid_102)-70:+d}B vs 70B, 0B-ovh ZX0 + E8/E9 + Free-BX 2D Stride 1..128, -2B vs 104B)")
        print(f"  zx0_ultra/asm/unzx0_hybrid_104b.asm: {len(dec_hybrid_104):3d} bytes ({len(dec_hybrid_104)-70:+d}B vs 70B, 0B-ovh ZX0 + E8/E9 + Free-BX 2D Stride)")
        print(f"  zx0_ultra/asm/unzx0_pro_110b.asm : {len(dec_pro_110):3d} bytes ({len(dec_pro_110)-70:+d}B vs 70B, 0B-ovh ZX0 + E8/E9 + 16b/8b 2D Stride, -1B vs 111B)")
        print(f"  zx0_ultra/asm/unzx0_pro_111b.asm : {len(dec_pro_111):3d} bytes ({len(dec_pro_111)-70:+d}B vs 70B, 0B-ovh ZX0 + E8/E9 + 16b/8b 2D Stride)")
        print(f"  zx0_ultra/asm/unzx0_ultra_128b.asm: {len(dec_ultra_128):3d} bytes ({len(dec_ultra_128)-70:+d}B vs 70B, Centered Split-Domain DH=1..127 + E8/E9 + 16b/8b 2D Stride)")

        assert len(dec_x86_main) == 67
        assert len(dec_67b) == 67
        assert len(dec_77b) == 77
        assert len(dec_78b) == 78
        assert len(dec_80b) == 80
        assert len(dec_compat82) == 82
        assert len(dec_strd82) == 82
        assert len(dec_x86e8_84) == 84
        assert len(dec_univ_93) == 93
        assert len(dec_hybrid_102) == 102
        assert len(dec_hybrid_104) == 104
        assert len(dec_pro_110) == 110
        assert len(dec_pro_111) == 111
        assert len(dec_ultra_128) == 128, f"unzx0_ultra_128b.asm must be exactly 128B (got {len(dec_ultra_128)}B)"

        honest_corpus, edge_cases, all_corpus = load_honest_datasets_and_edge_cases(
            run_all=args.all,
            limit=args.limit,
            filter_patterns=args.filters if args.filters else None,
        )

        def evaluate_and_verify_table(title: str, items: list[tuple[str, bytes]]) -> int:
            print("\n" + "=" * 162)
            print(title)
            print("=" * 162)
            header = (
                f"{'Dataset':32s} | {'Raw':>6s} | {'OrigZX0':>7s} | {'UltZX0':>6s} | "
                f"{'AdpK78':>6s} | {'Strd80':>6s} | {'Cmp82':>6s} | {'Strd82':>6s} | {'X86E84':>6s} | "
                f"{'Univ93':>6s} | {'Hyb104':>6s} | {'Pro111':>6s} | {'Ult128':>6s} | {'SHA-256':12s}"
            )
            print(header)
            print("-" * 162)

            def process_one(item: tuple[str, bytes]) -> tuple[str, list[int], str, int]:
                name, raw = item
                in_p = os.path.join(tmpdir, f"{name}.bin")
                with open(in_p, "wb") as fp:
                    fp.write(raw)

                expected_hash = sha256_hex(raw)
                local_checks = 0

                # 1. Upstream Einar Saukas zx0 (verified on 67B, 82B-compat, 84B-x86e8, 93B-univ, 102B/104B-hybrid, AND 110B/111B-pro!)
                orig_out = os.path.join(tmpdir, f"{name}.orig.zx0")
                subprocess.run(
                    [ZX0_ORIG_EXE, "-f", in_p, orig_out],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                c_orig = open(orig_out, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_67b, c_orig, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_compat82, c_orig, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_x86e8_84, c_orig, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_univ_93, c_orig, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_hybrid_102, c_orig, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_hybrid_104, c_orig, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_pro_110, c_orig, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_pro_111, c_orig, len(raw))) == expected_hash
                local_checks += 8

                # 2. ZX0-Ultra --zx0 (verified on unzx0_x86.asm, 67b, compat_82b, x86e8_84b, univ_93b, hybrid_102b, hybrid_104b, pro_110b, pro_111b)
                u_zx0_p = os.path.join(tmpdir, f"{name}.uzx0")
                subprocess.run([ZX0_ULTRA_EXE, "--zx0", in_p, u_zx0_p], check=True, stdout=subprocess.DEVNULL)
                c_uzx0 = open(u_zx0_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_x86_main, c_uzx0, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_67b, c_uzx0, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_compat82, c_uzx0, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_x86e8_84, c_uzx0, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_univ_93, c_uzx0, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_hybrid_102, c_uzx0, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_hybrid_104, c_uzx0, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_pro_110, c_uzx0, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_pro_111, c_uzx0, len(raw))) == expected_hash
                local_checks += 9

                # 3. ZX0-Ultra --adpk77 (verified on 77B unzx0_adpk_77b.asm AND 78B unzx0_adpk_78b.asm, k=1..12)
                u_k78_p = os.path.join(tmpdir, f"{name}.k78")
                subprocess.run([ZX0_ULTRA_EXE, "--adpk77", in_p, u_k78_p], check=True, stdout=subprocess.DEVNULL)
                c_k78 = open(u_k78_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_77b, c_k78, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_78b, c_k78, len(raw))) == expected_hash
                local_checks += 2

                # 4. ZX0-Ultra --strd80 (verified on 80B unzx0_strd_80b.asm)
                u_s80_p = os.path.join(tmpdir, f"{name}.s80")
                subprocess.run([ZX0_ULTRA_EXE, "--strd80", in_p, u_s80_p], check=True, stdout=subprocess.DEVNULL)
                c_s80 = open(u_s80_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_80b, c_s80, len(raw))) == expected_hash
                local_checks += 1

                # 5. ZX0-Ultra --compat82 (verified on 82B unzx0_compat_82b.asm)
                u_c82_p = os.path.join(tmpdir, f"{name}.c82")
                subprocess.run([ZX0_ULTRA_EXE, "--compat82", in_p, u_c82_p], check=True, stdout=subprocess.DEVNULL)
                c_c82 = open(u_c82_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_compat82, c_c82, len(raw))) == expected_hash
                local_checks += 1

                # 6. ZX0-Ultra --strd82 (verified on 82B unzx0_strd_82b.asm)
                u_s82_p = os.path.join(tmpdir, f"{name}.s82")
                subprocess.run([ZX0_ULTRA_EXE, "--strd82", in_p, u_s82_p], check=True, stdout=subprocess.DEVNULL)
                c_s82 = open(u_s82_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_strd82, c_s82, len(raw))) == expected_hash
                local_checks += 1

                # 7. ZX0-Ultra --x86e8 (verified on 84B unzx0_x86e8_84b.asm)
                u_e84_p = os.path.join(tmpdir, f"{name}.e84")
                subprocess.run([ZX0_ULTRA_EXE, "--x86e8", in_p, u_e84_p], check=True, stdout=subprocess.DEVNULL)
                c_e84 = open(u_e84_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_x86e8_84, c_e84, len(raw))) == expected_hash
                local_checks += 1

                # 8. ZX0-Ultra --univ93 (verified on 93B unzx0_univ_93b.asm)
                u_u93_p = os.path.join(tmpdir, f"{name}.u93")
                subprocess.run([ZX0_ULTRA_EXE, "--univ93", in_p, u_u93_p], check=True, stdout=subprocess.DEVNULL)
                c_u93 = open(u_u93_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_univ_93, c_u93, len(raw))) == expected_hash
                local_checks += 1

                # 9. ZX0-Ultra --hybrid108 (verified on 102B unzx0_hybrid_102b.asm AND 104B unzx0_hybrid_104b.asm)
                u_h108_p = os.path.join(tmpdir, f"{name}.h108")
                subprocess.run([ZX0_ULTRA_EXE, "--hybrid108", in_p, u_h108_p], check=True, stdout=subprocess.DEVNULL)
                c_h108 = open(u_h108_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_hybrid_102, c_h108, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_hybrid_104, c_h108, len(raw))) == expected_hash
                local_checks += 2

                # 10. ZX0-Ultra --pro111 (verified on 110B unzx0_pro_110b.asm AND 111B unzx0_pro_111b.asm)
                u_p111_p = os.path.join(tmpdir, f"{name}.p111")
                subprocess.run([ZX0_ULTRA_EXE, "--pro111", in_p, u_p111_p], check=True, stdout=subprocess.DEVNULL)
                c_p111 = open(u_p111_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_pro_110, c_p111, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_pro_111, c_p111, len(raw))) == expected_hash
                local_checks += 2

                # 11. ZX0-Ultra --ultra128 (verified on 128B unzx0_ultra_128b.asm & C CLI decompressor)
                u_u128_p = os.path.join(tmpdir, f"{name}.u128")
                subprocess.run([ZX0_ULTRA_EXE, "--ultra128", in_p, u_u128_p], check=True, stdout=subprocess.DEVNULL)
                c_u128 = open(u_u128_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_ultra_128, c_u128, len(raw))) == expected_hash
                local_checks += 1

                u_u128_dec_p = os.path.join(tmpdir, f"{name}.u128.dec")
                subprocess.run(
                    [ZX0_ULTRA_EXE, "--decompress", "--ultra128", "--raw-size", str(len(raw)), u_u128_p, u_u128_dec_p],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                assert sha256_hex(open(u_u128_dec_p, "rb").read()) == expected_hash
                local_checks += 1

                vals = [
                    len(raw), len(c_orig), len(c_uzx0), len(c_k78),
                    len(c_s80), len(c_c82), len(c_s82), len(c_e84), len(c_u93), len(c_h108), len(c_p111), len(c_u128),
                ]
                return name, vals, expected_hash, local_checks

            totals = [0] * 12
            checks = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
                results = list(ex.map(process_one, items))

            for name, vals, expected_hash, local_checks in results:
                checks += local_checks
                for idx, v in enumerate(vals):
                    totals[idx] += v
                print(
                    f"{name:32s} | {vals[0]:6d} | {vals[1]:7d} | {vals[2]:6d} | "
                    f"{vals[3]:6d} | {vals[4]:6d} | {vals[5]:6d} | {vals[6]:6d} | {vals[7]:6d} | {vals[8]:6d} | {vals[9]:6d} | {vals[10]:6d} | {vals[11]:6d} | {expected_hash[:12]}"
                )

            print("-" * 162)
            print(
                f"{f'TOTAL ({len(items)} files)':32s} | {totals[0]:6d} | {totals[1]:7d} | {totals[2]:6d} | "
                f"{totals[3]:6d} | {totals[4]:6d} | {totals[5]:6d} | {totals[6]:6d} | {totals[7]:6d} | {totals[8]:6d} | {totals[9]:6d} | {totals[10]:6d} | {totals[11]:6d} | PASS ({checks})"
            )
            return checks

        unicorn_checks = 0
        unicorn_checks += evaluate_and_verify_table(
            f"HONEST BENCHMARK & UNICORN x86-16 SHA-256 VERIFICATION ({len(honest_corpus)} DATASETS)",
            honest_corpus,
        )
        unicorn_checks += evaluate_and_verify_table(
            "ALGORITHMIC EDGE & REGRESSION SUITE (1B, HEURISTIC TRAP, KADANE SPLIT, E8 BOUNDARY GUARD)",
            edge_cases,
        )

        # Multi-Asset 7-Sprite Pack Solver (--pack) across all 10 formats
        print("\n" + "=" * 162)
        print("MULTI-CORE 7-SPRITE SUPERSTRING + WILDCARD BRIDGE SOLVER + UNICORN SLICE HASHING (ALL 10 FORMATS)")
        print("=" * 162)
        sprite_entries = [
            ("sword_pommel", SWORD_POMMEL),
            ("sword_grip", SWORD_GRIP),
            ("sword_guard", SWORD_GUARD),
            ("sword_blade", SWORD_BLADE),
            ("sword_tip", SWORD_TIP),
            ("half_skull_sprite", HALF_SKULL_SPRITE),
            ("half_jaw_sprite", HALF_JAW_SPRITE),
        ]
        sp_paths = []
        for sname, sbytes in sprite_entries:
            sp_p = os.path.join(tmpdir, f"{sname}.bin")
            with open(sp_p, "wb") as fp:
                fp.write(sbytes)
            sp_paths.append(sp_p)

        all_pack_formats = [
            ("--zx0", dec_67b, "ZX0-v2 (67B decoder)"),
            ("--adpk77", dec_78b, "ADPK-78B (78B decoder)"),
            ("--strd80", dec_80b, "STRD-80B (80B decoder)"),
            ("--compat82", dec_compat82, "COMPAT-82B (82B decoder)"),
            ("--strd82", dec_strd82, "STRD-82B (82B decoder)"),
            ("--x86e8", dec_x86e8_84, "X86E8-84B (84B decoder)"),
            ("--univ93", dec_univ_93, "UNIV-93B (93B decoder)"),
            ("--hybrid108", dec_hybrid_104, "HYBRID-104B (104B decoder)"),
            ("--pro111", dec_pro_111, "PRO-111B (111B decoder)"),
            ("--ultra128", dec_ultra_128, "ULTRA-128B (128B decoder)"),
        ]

        for flag, dec_bin, label in all_pack_formats:
            out_comp = os.path.join(tmpdir, f"pack_{flag}.bin")
            out_raw = os.path.join(tmpdir, f"pack_{flag}.raw")
            res = subprocess.run(
                [ZX0_ULTRA_EXE, "--pack", flag, "-o", out_comp, "--raw-out", out_raw, *sp_paths],
                check=True,
                capture_output=True,
                text=True,
            )
            print(res.stdout.strip())
            comp_bytes = open(out_comp, "rb").read()
            raw_layout = open(out_raw, "rb").read()
            unpacked = emulate_x86_decompressor(dec_bin, comp_bytes, len(raw_layout))
            assert sha256_hex(unpacked) == sha256_hex(raw_layout)
            for sname, sbytes in sprite_entries:
                assert sbytes in unpacked, f"Missing exact slice for {sname} in {label}!"
            unicorn_checks += 1
            print(
                f"  -> [UNICORN x86-16 SHA-256 VERIFIED] {label}: compressed={len(comp_bytes)}B, "
                f"decoder={len(dec_bin)}B, total(code+data)={len(comp_bytes)+len(dec_bin)}B "
                f"(vs baseline 70B+269B = 339B!)\n"
            )

        # 10-Asset Held-Karp Bitmask ATSP Multi-Asset Pack Test (>8 assets, non-sprite heterogeneous blocks)
        print("=" * 162)
        print("10-ASSET HELD-KARP BITMASK ATSP PACK SOLVER TEST (O(K^2 2^K) PAIRWISE-ZX0 + 2-OPT/RELOCATION)")
        print("=" * 162)

        pal_data = find_asset(all_corpus, "vga_rgb_pal")
        font_data = find_asset(all_corpus, "bitmap_font")
        code_data = find_asset(all_corpus, "x86_code")
        tile_data = find_asset(all_corpus, "vga_8bpp_tile")
        struct_data = find_asset(all_corpus, "structs")
        json_data = find_asset(all_corpus, "arxiv_api_json")
        pe_data = find_asset(all_corpus, "pe_headers")

        hk_assets = [
            ("pal_a", pal_data[:96]),
            ("font_a", font_data[:96]),
            ("code_a", code_data[:112]),
            ("tile_a", tile_data[:96]),
            ("struct_a", struct_data[:96]),
            ("pal_b", pal_data[64:160]),    # 32B overlap with pal_a!
            ("code_b", code_data[80:192]),   # 32B overlap with code_a!
            ("json_a", json_data[:120]),
            ("pe_hdr", pe_data[:96]),
            ("json_b", json_data[90:210]),  # 30B overlap with json_a!
        ]
        hk_paths = []
        for gname, gbytes in hk_assets:
            gp = os.path.join(tmpdir, f"hk_{gname}.bin")
            with open(gp, "wb") as fp:
                fp.write(gbytes)
            hk_paths.append(gp)

        hk_comp_p = os.path.join(tmpdir, "hk_pack.c82")
        hk_raw_p = os.path.join(tmpdir, "hk_pack.raw")
        hk_res = subprocess.run(
            [ZX0_ULTRA_EXE, "--pack", "--zx0", "-o", hk_comp_p, "--raw-out", hk_raw_p, *hk_paths],
            check=True,
            capture_output=True,
            text=True,
        )
        print(hk_res.stdout.strip())
        hk_comp = open(hk_comp_p, "rb").read()
        hk_raw = open(hk_raw_p, "rb").read()
        hk_unpacked = emulate_x86_decompressor(dec_67b, hk_comp, len(hk_raw))
        assert sha256_hex(hk_unpacked) == sha256_hex(hk_raw)
        for gname, gbytes in hk_assets:
            assert gbytes in hk_unpacked, f"Missing slice {gname} in 10-asset Held-Karp pack!"
        unicorn_checks += 1
        print(
            f"  -> [UNICORN x86-16 SHA-256 VERIFIED] 10-asset Held-Karp pack: "
            f"raw_sum={sum(len(b) for _, b in hk_assets)}B -> merged={len(hk_raw)}B -> compressed={len(hk_comp)}B "
            f"(SHA-256={sha256_hex(hk_unpacked)[:16]})"
        )

        print("\n" + "=" * 162)
        print(f"ALL {unicorn_checks} UNICORN x86-16 CRYPTOGRAPHIC SHA-256 VERIFICATIONS PASSED!")
        print("=" * 162)


if __name__ == "__main__":
    main()
