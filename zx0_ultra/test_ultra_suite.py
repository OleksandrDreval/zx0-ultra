#!/usr/bin/env python3
"""
Cryptographic Unicorn x86-16 Verification & Honest Anti-Overfitting Benchmark Suite for ZX0-Ultra.

Assembles all 7 x86-16 decompressors (67B..84B) with NASM, compresses all 20 real-world files in
datasets/ plus 4 algorithmic edge/trap regression cases and multi-asset superstring/wildcard packs
(7-sprite exhaustive + 10-asset Held-Karp ATSP), and verifies every unpacked buffer in Unicorn
16-bit real mode via SHA-256 memory hashing and pre/post-buffer canary guards.

Author: Oleksandr Dreval (MIT License)
SPDX-License-Identifier: MIT
"""

import hashlib
import math
import os
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

import struct

ULTRA_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_DIR = os.path.abspath(os.path.join(ULTRA_DIR, ".."))
BIN_DIR = os.path.join(REPO_DIR, "bin")
DATASETS_DIR = os.path.join(REPO_DIR, "datasets")
ZX_THINGS_DIR = os.path.join(REPO_DIR, "ZX things")
ASM_DIR = os.path.join(ULTRA_DIR, "asm")

sys.path.insert(0, REPO_DIR)
import sprite_data  # noqa: E402

NASM_EXE = r"C:\Program Files\NASM\nasm.exe"
ZX0_ULTRA_EXE = os.path.join(BIN_DIR, "zx0_ultra.exe")
ZX0_ORIG_EXE = os.path.join(BIN_DIR, "zx0.exe")

CODE_LOAD_ADDR = 0x0100
HALT_STUB_ADDR = 0x0180
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


def load_honest_datasets_and_edge_cases() -> tuple[list[tuple[str, bytes]], list[tuple[str, bytes]]]:
    honest_files: list[tuple[str, bytes]] = []
    for fname in sorted(os.listdir(DATASETS_DIR)):
        fpath = os.path.join(DATASETS_DIR, fname)
        if os.path.isfile(fpath) and fname.endswith(".bin"):
            label = fname[:-4]
            with open(fpath, "rb") as fp:
                honest_files.append((label, fp.read()))

    # Additional algorithmic edge & regression cases (kept separate from honest benchmark totals!)
    edge_cases: list[tuple[str, bytes]] = []
    edge_cases.append(("edge_01_1byte", b"\x42"))

    trap_prefix = b"ABCDEFGHIJKLMNO" + b"ABCDEFGHIJKLMNO"
    trap_lit1 = b"1234567890abcdef"
    trap_match = b"2"
    trap_lit2 = b"ghijklmnpqrstuvw"
    trap_trail = bytes([(x * 17 + 3) % 256 for x in range(16)])
    edge_cases.append(("edge_02_heuristic_trap", trap_prefix + trap_lit1 + trap_match + trap_lit2 + trap_trail))

    # Hybrid code + RGB table testing Kadane O(N) arbitrary split point (750B code + 450B RGB table)
    code_src = dict(honest_files)["03_x86_code_1142"][:750]
    pal_tail = bytes(((i >> 2) & 63) if (c == 0) else (((i * 3) >> 3) & 63 if c == 1 else ((63 - (i >> 2)) & 63))
                     for i in range(150) for c in range(3))
    edge_cases.append(("edge_03_kadane_750_450", code_src + pal_tail))

    # Trailing 0xE8 boundary safety test (ensures X86E8_84B safely falls back when raw[-2:] contains 0xE8)
    edge_cases.append(("edge_04_trailing_e8_guard", code_src[:254] + b"\xe8\xe8"))

    return honest_files, edge_cases


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dec_x86_main = assemble_flat_binary(
            os.path.join(ZX_THINGS_DIR, "unzx0_x86.asm"), os.path.join(tmpdir, "dx86.bin")
        )
        dec_67b = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_67b.asm"), os.path.join(tmpdir, "d67.bin")
        )
        dec_78b = assemble_flat_binary(
            os.path.join(ASM_DIR, "unzx0_adpk_77b.asm"), os.path.join(tmpdir, "d78.bin")
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

        print("=" * 124)
        print("ASSEMBLED x86-16 DECOMPRESSOR FOOTPRINTS (NASM FLAT BINARIES vs 70B BASELINE & 81B 8088 REF)")
        print("=" * 124)
        print(f"  ZX things/unzx0_x86.asm        : {len(dec_x86_main):3d} bytes ({len(dec_x86_main)-70:+d}B vs 70B, 100% ZX0 v2 compatible)")
        print(f"  zx0_ultra/asm/unzx0_67b.asm    : {len(dec_67b):3d} bytes ({len(dec_67b)-70:+d}B vs 70B, 100% ZX0 v2 compatible)")
        print(f"  zx0_ultra/asm/unzx0_adpk_77b   : {len(dec_78b):3d} bytes ({len(dec_78b)-70:+d}B vs 70B, Adaptive-k 1..12 offset LSB)")
        print(f"  zx0_ultra/asm/unzx0_strd_80b   : {len(dec_80b):3d} bytes ({len(dec_80b)-70:+d}B vs 70B, Order-1 16-bit Stride-Delta)")
        print(f"  zx0_ultra/asm/unzx0_compat_82b : {len(dec_compat82):3d} bytes ({len(dec_compat82)-70:+d}B vs 70B, 0B-ovh ZX0 compat + Stride)")
        print(f"  zx0_ultra/asm/unzx0_strd_82b   : {len(dec_strd82):3d} bytes ({len(dec_strd82)-70:+d}B vs 70B, Multi-Pass Order-1..5 Stride)")
        print(f"  zx0_ultra/asm/unzx0_x86e8_84b  : {len(dec_x86e8_84):3d} bytes ({len(dec_x86e8_84)-70:+d}B vs 70B, 0B-ovh ZX0 compat + x86 E8 Call)")

        assert len(dec_x86_main) == 67
        assert len(dec_67b) == 67
        assert len(dec_78b) == 78
        assert len(dec_80b) == 80
        assert len(dec_compat82) == 82
        assert len(dec_strd82) == 82
        assert len(dec_x86e8_84) == 84

        honest_corpus, edge_cases = load_honest_datasets_and_edge_cases()

        def evaluate_and_verify_table(title: str, items: list[tuple[str, bytes]]) -> int:
            print("\n" + "=" * 124)
            print(title)
            print("=" * 124)
            header = (
                f"{'Dataset':28s} | {'Raw':>6s} | {'OrigZX0':>7s} | {'UltZX0':>6s} | "
                f"{'AdpK78':>6s} | {'Strd80':>6s} | {'Cmp82':>6s} | {'Strd82':>6s} | {'X86E84':>6s} | {'SHA-256':14s}"
            )
            print(header)
            print("-" * 124)

            totals = [0] * 8
            checks = 0

            for name, raw in items:
                in_p = os.path.join(tmpdir, f"{name}.bin")
                with open(in_p, "wb") as fp:
                    fp.write(raw)

                expected_hash = sha256_hex(raw)

                # 1. Upstream Einar Saukas zx0.exe (verified on 67B, 82B-compat, AND 84B-x86e8!)
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
                checks += 3

                # 2. ZX0-Ultra --zx0 (verified on unzx0_x86.asm, unzx0_67b, unzx0_compat_82b, unzx0_x86e8_84b)
                u_zx0_p = os.path.join(tmpdir, f"{name}.uzx0")
                subprocess.run([ZX0_ULTRA_EXE, "--zx0", in_p, u_zx0_p], check=True, stdout=subprocess.DEVNULL)
                c_uzx0 = open(u_zx0_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_x86_main, c_uzx0, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_67b, c_uzx0, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_compat82, c_uzx0, len(raw))) == expected_hash
                assert sha256_hex(emulate_x86_decompressor(dec_x86e8_84, c_uzx0, len(raw))) == expected_hash
                checks += 4

                # 3. ZX0-Ultra --adpk77 (verified on 78B unzx0_adpk_77b.asm, k=1..12)
                u_k78_p = os.path.join(tmpdir, f"{name}.k78")
                subprocess.run([ZX0_ULTRA_EXE, "--adpk77", in_p, u_k78_p], check=True, stdout=subprocess.DEVNULL)
                c_k78 = open(u_k78_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_78b, c_k78, len(raw))) == expected_hash
                checks += 1

                # 4. ZX0-Ultra --strd80 (verified on 80B unzx0_strd_80b.asm)
                u_s80_p = os.path.join(tmpdir, f"{name}.s80")
                subprocess.run([ZX0_ULTRA_EXE, "--strd80", in_p, u_s80_p], check=True, stdout=subprocess.DEVNULL)
                c_s80 = open(u_s80_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_80b, c_s80, len(raw))) == expected_hash
                checks += 1

                # 5. ZX0-Ultra --compat82 (verified on 82B unzx0_compat_82b.asm)
                u_c82_p = os.path.join(tmpdir, f"{name}.c82")
                subprocess.run([ZX0_ULTRA_EXE, "--compat82", in_p, u_c82_p], check=True, stdout=subprocess.DEVNULL)
                c_c82 = open(u_c82_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_compat82, c_c82, len(raw))) == expected_hash
                checks += 1

                # 6. ZX0-Ultra --strd82 (verified on 82B unzx0_strd_82b.asm)
                u_s82_p = os.path.join(tmpdir, f"{name}.s82")
                subprocess.run([ZX0_ULTRA_EXE, "--strd82", in_p, u_s82_p], check=True, stdout=subprocess.DEVNULL)
                c_s82 = open(u_s82_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_strd82, c_s82, len(raw))) == expected_hash
                checks += 1

                # 7. ZX0-Ultra --x86e8 (verified on 84B unzx0_x86e8_84b.asm)
                u_e84_p = os.path.join(tmpdir, f"{name}.e84")
                subprocess.run([ZX0_ULTRA_EXE, "--x86e8", in_p, u_e84_p], check=True, stdout=subprocess.DEVNULL)
                c_e84 = open(u_e84_p, "rb").read()
                assert sha256_hex(emulate_x86_decompressor(dec_x86e8_84, c_e84, len(raw))) == expected_hash
                checks += 1

                vals = [
                    len(raw), len(c_orig), len(c_uzx0), len(c_k78),
                    len(c_s80), len(c_c82), len(c_s82), len(c_e84),
                ]
                for idx, v in enumerate(vals):
                    totals[idx] += v

                print(
                    f"{name:28s} | {vals[0]:6d} | {vals[1]:7d} | {vals[2]:6d} | "
                    f"{vals[3]:6d} | {vals[4]:6d} | {vals[5]:6d} | {vals[6]:6d} | {vals[7]:6d} | {expected_hash[:14]}"
                )

            print("-" * 124)
            print(
                f"{f'TOTAL ({len(items)} files)':28s} | {totals[0]:6d} | {totals[1]:7d} | {totals[2]:6d} | "
                f"{totals[3]:6d} | {totals[4]:6d} | {totals[5]:6d} | {totals[6]:6d} | {totals[7]:6d} | PASS ({checks})"
            )
            return checks

        unicorn_checks = 0
        unicorn_checks += evaluate_and_verify_table(
            f"HONEST BENCHMARK & UNICORN x86-16 SHA-256 VERIFICATION ({len(honest_corpus)} REAL DATASETS IN datasets/)",
            honest_corpus,
        )
        unicorn_checks += evaluate_and_verify_table(
            "ALGORITHMIC EDGE & REGRESSION SUITE (1B, HEURISTIC TRAP, KADANE SPLIT, E8 BOUNDARY GUARD)",
            edge_cases,
        )

        # Multi-Asset 7-Sprite Pack Solver (--pack)
        print("\n" + "=" * 124)
        print("MULTI-CORE 7-SPRITE SUPERSTRING + WILDCARD BRIDGE SOLVER + UNICORN SLICE HASHING")
        print("=" * 124)
        sprite_entries = [
            ("sword_pommel", sprite_data.SWORD_POMMEL),
            ("sword_grip", sprite_data.SWORD_GRIP),
            ("sword_guard", sprite_data.SWORD_GUARD),
            ("sword_blade", sprite_data.SWORD_BLADE),
            ("sword_tip", sprite_data.SWORD_TIP),
            ("half_skull_sprite", sprite_data.HALF_SKULL_SPRITE),
            ("half_jaw_sprite", sprite_data.HALF_JAW_SPRITE),
        ]
        sp_paths = []
        for sname, sbytes in sprite_entries:
            sp_p = os.path.join(tmpdir, f"{sname}.bin")
            with open(sp_p, "wb") as fp:
                fp.write(sbytes)
            sp_paths.append(sp_p)

        for flag, dec_bin, label in [
            ("--zx0", dec_67b, "ZX0-v2 (67B decoder)"),
            ("--adpk77", dec_78b, "ADPK-78B (78B decoder)"),
            ("--compat82", dec_compat82, "COMPAT-82B (82B decoder)"),
            ("--x86e8", dec_x86e8_84, "X86E8-84B (84B decoder)"),
        ]:
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
        print("=" * 124)
        print("10-ASSET HELD-KARP BITMASK ATSP PACK SOLVER TEST (O(K^2 2^K) PAIRWISE-ZX0 + 2-OPT/RELOCATION)")
        print("=" * 124)
        corpus_dict = dict(honest_corpus)
        hk_assets = [
            ("pal_a", corpus_dict["07_vga_rgb_pal_768"][:96]),
            ("font_a", corpus_dict["09_bitmap_font_768"][:96]),
            ("code_a", corpus_dict["03_x86_code_1142"][:112]),
            ("tile_a", corpus_dict["05_vga_8bpp_tile"][:96]),
            ("struct_a", corpus_dict["08_structs_1024"][:96]),
            ("pal_b", corpus_dict["07_vga_rgb_pal_768"][64:160]),    # 32B overlap with pal_a!
            ("code_b", corpus_dict["03_x86_code_1142"][80:192]),     # 32B overlap with code_a!
            ("json_a", corpus_dict["17_arxiv_api_json_4096"][:120]),
            ("pe_hdr", corpus_dict["18_pe_headers_2048"][:96]),
            ("json_b", corpus_dict["17_arxiv_api_json_4096"][90:210]),  # 30B overlap with json_a!
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

        print("\n" + "=" * 124)
        print(f"ALL {unicorn_checks} UNICORN x86-16 CRYPTOGRAPHIC SHA-256 VERIFICATIONS PASSED!")
        print("=" * 124)


if __name__ == "__main__":
    main()
