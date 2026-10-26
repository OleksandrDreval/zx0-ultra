#!/usr/bin/env python3
"""
Cryptographic Unicorn x86-16 Verification & Anti-Overfitting Benchmark Suite for ZX0-Ultra.

Assembles all x86-16 decompressors with NASM, compresses 17 diverse datasets (sprites, machine code,
source text, UTF-8 prose, 2D VGA tiles, 320px Mode 13h scanlines, 16-bit sine & audio LUTs, RGB
palettes, C struct arrays, 1bpp fonts, RLE/stride stress, literal-split DP traps, hybrid code+table
binaries, and 1-byte edge cases) plus multi-asset superstring/wildcard packs, and verifies every
decompressed buffer in Unicorn 16-bit real mode via SHA-256 memory hashing.

Author: Oleksandr Dreval (MIT License)
SPDX-License-Identifier: MIT
"""

import hashlib
import math
import os
import random
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
DEMO_DIR = os.path.abspath(os.path.join(ULTRA_DIR, ".."))
BIN_DIR = os.path.join(DEMO_DIR, "bin")
ZX0_REF_SRC_DIR = os.path.join(DEMO_DIR, "third_party", "ZX0-main", "src")
ZX_THINGS_DIR = os.path.join(DEMO_DIR, "ZX things")
ASM_DIR = os.path.join(ULTRA_DIR, "asm")

sys.path.insert(0, DEMO_DIR)
import sprite_data  # noqa: E402

NASM_EXE = r"C:\Program Files\NASM\nasm.exe"
ZX0_ULTRA_EXE = os.path.join(BIN_DIR, "zx0_ultra.exe")
ZX0_ORIG_EXE = os.path.join(BIN_DIR, "zx0.exe")

CODE_LOAD_ADDR = 0x0100
HALT_STUB_ADDR = 0x0500
STREAM_LOAD_ADDR = 0x1000
OUTPUT_LOAD_ADDR = 0x5000
INITIAL_STACK_PTR = 0xFFFE
SEGMENT_MEM_SIZE = 0x10000
CANARY_GUARD_BYTES = 16
MAX_EMULATION_STEPS = 5_000_000


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assemble_flat_binary(asm_path: str, out_bin_path: str) -> bytes:
    subprocess.run([NASM_EXE, "-f", "bin", asm_path, "-o", out_bin_path], check=True)
    with open(out_bin_path, "rb") as fp:
        return fp.read()


def emulate_x86_decompressor(decoder_code: bytes, compressed_stream: bytes, expected_len: int) -> bytes:
    mu = Uc(UC_ARCH_X86, UC_MODE_16)
    mu.mem_map(0x0000, SEGMENT_MEM_SIZE)

    # Pre-fill output area with deterministic non-zero canary bytes so unwritten bytes cannot pass by luck
    canary = bytes(((i * 73 + 0x5A) & 0xFF) for i in range(expected_len + 64))
    mu.mem_write(OUTPUT_LOAD_ADDR, canary)

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
    mu.reg_write(UC_X86_REG_DI, OUTPUT_LOAD_ADDR)
    mu.reg_write(UC_X86_REG_SP, INITIAL_STACK_PTR)
    mu.mem_write(INITIAL_STACK_PTR, struct.pack("<H", HALT_STUB_ADDR))

    mu.emu_start(CODE_LOAD_ADDR, HALT_STUB_ADDR, count=MAX_EMULATION_STEPS)

    final_di = mu.reg_read(UC_X86_REG_DI)
    final_sp = mu.reg_read(UC_X86_REG_SP)
    assert final_sp == ((INITIAL_STACK_PTR + 2) & 0xFFFF), f"Stack imbalance: SP={final_sp:#x}"
    assert final_di == OUTPUT_LOAD_ADDR + expected_len, (
        f"DI end pointer mismatch: got {final_di:#x}, expected {OUTPUT_LOAD_ADDR + expected_len:#x}"
    )

    # Verify post-end canary guard was never overwritten
    guard_after = bytes(mu.mem_read(OUTPUT_LOAD_ADDR + expected_len, CANARY_GUARD_BYTES))
    assert guard_after == canary[expected_len : expected_len + CANARY_GUARD_BYTES], (
        "Output buffer overrun detected!"
    )

    return bytes(mu.mem_read(OUTPUT_LOAD_ADDR, expected_len))


def build_diverse_corpus() -> list[tuple[str, bytes]]:
    rng = random.Random(20261025)
    corpus: list[tuple[str, bytes]] = []

    # 1. Overlapped 7-sprite layout currently in static_demo.asm (547B)
    base_sprites = [
        ("sword_grip", sprite_data.SWORD_GRIP),
        ("half_skull_sprite", sprite_data.HALF_SKULL_SPRITE),
        ("sword_blade", sprite_data.SWORD_BLADE),
        ("sword_guard", sprite_data.SWORD_GUARD),
        ("half_jaw_sprite", sprite_data.HALF_JAW_SPRITE),
        ("sword_tip", sprite_data.SWORD_TIP),
        ("sword_pommel", sprite_data.SWORD_POMMEL),
    ]
    merged_547 = bytearray(base_sprites[0][1])
    for _, sp in base_sprites[1:]:
        best_ov = 0
        for ov in range(min(len(merged_547), len(sp)) - 1, 0, -1):
            if merged_547[-ov:] == sp[:ov]:
                best_ov = ov
                break
        merged_547.extend(sp[best_ov:])
    corpus.append(("01_sprites_547", bytes(merged_547)))

    # 2. Unoverlapped 7-sprite concatenation (552B)
    corpus.append(("02_sprites_552", b"".join(sp for _, sp in base_sprites)))

    # 3. Real 16-bit x86 machine code assembled from static_demo.asm (first 1,200B of code)
    with tempfile.TemporaryDirectory() as td:
        com_bin = os.path.join(td, "demo.com")
        subprocess.run(
            [NASM_EXE, "-f", "bin", os.path.join(DEMO_DIR, "static_demo.asm"), "-o", com_bin],
            cwd=DEMO_DIR,
            check=True,
        )
        demo_bytes = open(com_bin, "rb").read()
    x86_code_1200 = demo_bytes[:1200]
    corpus.append(("03_x86_code_1200", x86_code_1200))

    # 4. C source text (optimize.c, ~4.95KB)
    c_src = open(os.path.join(ZX0_REF_SRC_DIR, "optimize.c"), "rb").read()
    corpus.append(("04_c_source_text", c_src))

    # 5. Procedural 32x32 8bpp VGA tile texture (1,024B)
    tile = bytes(((x * 7 + y * 13 + ((x ^ y) & 7)) & 0x3F) for y in range(32) for x in range(32))
    corpus.append(("05_vga_8bpp_tile", tile))

    # 6. 256-entry 16-bit fixed-point sine LUT (512B)
    sin_lut = b"".join(
        struct.pack("<h", int(round(32767.0 * math.sin(2.0 * math.pi * i / 256.0))))
        for i in range(256)
    )
    corpus.append(("06_sin16_lut_512", sin_lut))

    # 7. 256-color VGA 6-bit RGB gradient palette (768B)
    pal = bytearray()
    for i in range(256):
        pal.extend(((i >> 2) & 63, ((i * 3) >> 3) & 63, (63 - (i >> 2)) & 63))
    vga_pal_768 = bytes(pal)
    corpus.append(("07_vga_rgb_pal_768", vga_pal_768))

    # 8. Array of 128 8-byte entity records / structs (1,024B)
    structs = b"".join(
        struct.pack("<HHBBH", 100 + i * 4, 200 - i, i & 7, 1, 0x1000 + i * 16)
        for i in range(128)
    )
    corpus.append(("08_structs_1024", structs))

    # 9. 96-glyph 8x8 1bpp bitmap font patterns (768B)
    font = bytearray()
    for g in range(96):
        for r in range(8):
            font.append(((g * 17 + r * 31) ^ (0xAA if (r & 1) else 0x55)) & 0x7E if 0 < r < 7 else 0)
    corpus.append(("09_bitmap_font_768", bytes(font)))

    # 10. Alternating RLE & stride-4 pattern stress dataset (600B)
    alt = bytearray()
    for blk in range(15):
        alt.extend([blk * 11 & 0xFF] * 20)
        for k in range(20):
            alt.append((blk + (k & 3) * 40) & 0xFF)
    corpus.append(("10_alt_stride_rle", bytes(alt)))

    # 11. Synthetic Literal-Split / Match-Bridging DP Corpus (800B)
    motif_a = bytes(rng.randrange(1, 255) for _ in range(24))
    motif_b = bytes(rng.randrange(1, 255) for _ in range(18))
    trap = bytearray()
    for step in range(20):
        trap.extend(motif_a[:10])
        trap.append(motif_a[0])
        trap.extend(motif_a[11:])
        trap.extend(motif_b[:8])
        trap.append((step * 37 + 3) & 0xFF)
        trap.extend(motif_b[9:])
    corpus.append(("11_dp_lit_bridge_trap", bytes(trap)))

    # 12. Mode 13h 320-pixel wide scanline buffer (320 x 12 = 3,840B, testing 16-bit stride = 320)
    scanlines = bytearray()
    for y in range(12):
        for x in range(320):
            scanlines.append((x // 5 + y + ((x * 3) & 7)) & 0xFF)
    corpus.append(("12_mode13h_320x12", bytes(scanlines)))

    # 13. 16-bit PCM damped harmonic audio wave (1,024B = 512 samples)
    pcm = b"".join(
        struct.pack(
            "<h",
            int(round(24000.0 * math.exp(-t / 400.0) * math.sin(2.0 * math.pi * t / 32.0))),
        )
        for t in range(512)
    )
    corpus.append(("13_pcm16_audio_1024", pcm))

    # 14. Mixed English + Ukrainian technical prose in UTF-8 (1,410B)
    prose = (
        "Алгоритм стиснення ZX0 використовує динамічне програмування на графі станів та гамма-код Еліаса. "
        "Exact Pareto frontier pruning guarantees global optimality in O(N*W) time without heuristics! "
        "У реальному режимі x86 декомпресор займає лише 67–82 байти машинного коду для MS-DOS .COM.\n"
    ).encode("utf-8") * 5
    corpus.append(("14_utf8_prose_1410", prose))

    # 15. Minimal 1-byte edge-case file
    corpus.append(("15_edge_1byte", b"\x42"))

    # 16. Reference Heuristic Literal-Split Trap (from benchmark_zx0.py, 79B)
    # Demonstrates ZX0-Ultra --zx0 beating upstream Einar Saukas zx0.exe on pure ZX0 v2!
    trap_prefix = b"ABCDEFGHIJKLMNO" + b"ABCDEFGHIJKLMNO"
    trap_lit1 = b"1234567890abcdef"
    trap_match = b"2"
    trap_lit2 = b"ghijklmnpqrstuvw"
    trap_trail = bytes([(x * 17 + 3) % 256 for x in range(16)])
    corpus.append(("16_ref_heuristic_trap", trap_prefix + trap_lit1 + trap_match + trap_lit2 + trap_trail))

    # 17. Hybrid Binary: 600B x86 machine code + 600B RGB gradient palette (1,200B)
    # Demonstrates piecewise tail filtering (leaving code prefix untouched while delta-filtering palette tail)!
    corpus.append(("17_hybrid_code_pal", x86_code_1200[:600] + vga_pal_768[:600]))

    return corpus


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

        print("=" * 112)
        print("ASSEMBLED x86-16 DECOMPRESSOR FOOTPRINTS (NASM FLAT BINARIES vs 70B BASELINE & 81B 8088 REF)")
        print("=" * 112)
        print(f"  ZX things/unzx0_x86.asm        : {len(dec_x86_main):3d} bytes ({len(dec_x86_main)-70:+d}B vs 70B, 100% ZX0 v2 compatible)")
        print(f"  ZX0-Ultra unzx0_67b.asm        : {len(dec_67b):3d} bytes ({len(dec_67b)-70:+d}B vs 70B, 100% ZX0 v2 compatible)")
        print(f"  ZX0-Ultra unzx0_adpk_77b.asm   : {len(dec_78b):3d} bytes ({len(dec_78b)-70:+d}B vs 70B, Adaptive-k 1..8 offset LSB)")
        print(f"  ZX0-Ultra unzx0_strd_80b.asm   : {len(dec_80b):3d} bytes ({len(dec_80b)-70:+d}B vs 70B, Order-1 16-bit Stride-Delta)")
        print(f"  ZX0-Ultra unzx0_compat_82b.asm : {len(dec_compat82):3d} bytes ({len(dec_compat82)-70:+d}B vs 70B, 0-overhead ZX0 compat + Stride-Delta)")
        print(f"  ZX0-Ultra unzx0_strd_82b.asm   : {len(dec_strd82):3d} bytes ({len(dec_strd82)-70:+d}B vs 70B, Multi-Pass Order-1..5 Stride-Delta)")

        assert len(dec_x86_main) == 67
        assert len(dec_67b) == 67
        assert len(dec_78b) == 78
        assert len(dec_80b) == 80
        assert len(dec_compat82) == 82
        assert len(dec_strd82) == 82

        corpus = build_diverse_corpus()

        print("\n" + "=" * 112)
        print("BENCHMARK & UNICORN x86-16 SHA-256 VERIFICATION ACROSS 17 DIVERSE DATASETS")
        print("=" * 112)
        header = (
            f"{'Dataset':22s} | {'Raw':>5s} | {'OrigZX0':>7s} | {'UltZX0':>6s} | "
            f"{'AdpK78':>6s} | {'Strd80':>6s} | {'Cmp82':>6s} | {'Strd82':>6s} | {'SHA-256':16s}"
        )
        print(header)
        print("-" * 112)

        totals = [0, 0, 0, 0, 0, 0, 0]
        unicorn_checks = 0

        for name, raw in corpus:
            in_p = os.path.join(tmpdir, f"{name}.bin")
            with open(in_p, "wb") as fp:
                fp.write(raw)

            expected_hash = sha256_hex(raw)

            # 1. Upstream Einar Saukas zx0.exe (verified on BOTH 67B unzx0_67b AND 82B unzx0_compat_82b!)
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
            unicorn_checks += 2

            # 2. ZX0-Ultra --zx0 (tested on unzx0_x86.asm, unzx0_67b.asm, and unzx0_compat_82b.asm!)
            u_zx0_p = os.path.join(tmpdir, f"{name}.uzx0")
            subprocess.run([ZX0_ULTRA_EXE, "--zx0", in_p, u_zx0_p], check=True, stdout=subprocess.DEVNULL)
            c_uzx0 = open(u_zx0_p, "rb").read()
            assert sha256_hex(emulate_x86_decompressor(dec_x86_main, c_uzx0, len(raw))) == expected_hash
            assert sha256_hex(emulate_x86_decompressor(dec_67b, c_uzx0, len(raw))) == expected_hash
            assert sha256_hex(emulate_x86_decompressor(dec_compat82, c_uzx0, len(raw))) == expected_hash
            unicorn_checks += 3

            # 3. ZX0-Ultra --adpk77 (tested on 78B unzx0_adpk_77b.asm)
            u_k78_p = os.path.join(tmpdir, f"{name}.k78")
            subprocess.run([ZX0_ULTRA_EXE, "--adpk77", in_p, u_k78_p], check=True, stdout=subprocess.DEVNULL)
            c_k78 = open(u_k78_p, "rb").read()
            assert sha256_hex(emulate_x86_decompressor(dec_78b, c_k78, len(raw))) == expected_hash
            unicorn_checks += 1

            # 4. ZX0-Ultra --strd80 (tested on 80B unzx0_strd_80b.asm)
            u_s80_p = os.path.join(tmpdir, f"{name}.s80")
            subprocess.run([ZX0_ULTRA_EXE, "--strd80", in_p, u_s80_p], check=True, stdout=subprocess.DEVNULL)
            c_s80 = open(u_s80_p, "rb").read()
            assert sha256_hex(emulate_x86_decompressor(dec_80b, c_s80, len(raw))) == expected_hash
            unicorn_checks += 1

            # 5. ZX0-Ultra --compat82 (tested on 82B unzx0_compat_82b.asm)
            u_c82_p = os.path.join(tmpdir, f"{name}.c82")
            subprocess.run([ZX0_ULTRA_EXE, "--compat82", in_p, u_c82_p], check=True, stdout=subprocess.DEVNULL)
            c_c82 = open(u_c82_p, "rb").read()
            assert sha256_hex(emulate_x86_decompressor(dec_compat82, c_c82, len(raw))) == expected_hash
            unicorn_checks += 1

            # 6. ZX0-Ultra --strd82 (tested on 82B unzx0_strd_82b.asm)
            u_s82_p = os.path.join(tmpdir, f"{name}.s82")
            subprocess.run([ZX0_ULTRA_EXE, "--strd82", in_p, u_s82_p], check=True, stdout=subprocess.DEVNULL)
            c_s82 = open(u_s82_p, "rb").read()
            assert sha256_hex(emulate_x86_decompressor(dec_strd82, c_s82, len(raw))) == expected_hash
            unicorn_checks += 1

            vals = [len(raw), len(c_orig), len(c_uzx0), len(c_k78), len(c_s80), len(c_c82), len(c_s82)]
            for idx, v in enumerate(vals):
                totals[idx] += v

            print(
                f"{name:22s} | {vals[0]:5d} | {vals[1]:7d} | {vals[2]:6d} | "
                f"{vals[3]:6d} | {vals[4]:6d} | {vals[5]:6d} | {vals[6]:6d} | {expected_hash[:16]}"
            )

        print("-" * 112)
        print(
            f"{'TOTAL (17 datasets)':22s} | {totals[0]:5d} | {totals[1]:7d} | {totals[2]:6d} | "
            f"{totals[3]:6d} | {totals[4]:6d} | {totals[5]:6d} | {totals[6]:6d} | PASS ({unicorn_checks} runs)"
        )

        # Multi-Asset 7-Sprite Pack Solver (--pack) across --zx0, --adpk77, and --compat82
        print("\n" + "=" * 112)
        print("MULTI-CORE 7-SPRITE SUPERSTRING + WILDCARD BRIDGE SOLVER + UNICORN SLICE HASHING")
        print("=" * 112)
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

        # Anti-Overfitting General Multi-Asset Pack Test (5 non-sprite heterogeneous blocks)
        print("=" * 112)
        print("GENERAL HETEROGENEOUS 5-ASSET PACK SOLVER TEST (NON-SPRITE ANTI-OVERFITTING CHECK)")
        print("=" * 112)
        corpus_dict = dict(corpus)
        gen_assets = [
            ("pal_slice", corpus_dict["07_vga_rgb_pal_768"][:192]),
            ("font_slice", corpus_dict["09_bitmap_font_768"][:128]),
            ("code_slice", corpus_dict["03_x86_code_1200"][:160]),
            ("tile_slice", corpus_dict["05_vga_8bpp_tile"][:128]),
            ("struct_slice", corpus_dict["08_structs_1024"][:128]),
        ]
        gen_paths = []
        for gname, gbytes in gen_assets:
            gp = os.path.join(tmpdir, f"{gname}.bin")
            with open(gp, "wb") as fp:
                fp.write(gbytes)
            gen_paths.append(gp)

        gen_comp_p = os.path.join(tmpdir, "gen_pack.c82")
        gen_raw_p = os.path.join(tmpdir, "gen_pack.raw")
        subprocess.run(
            [ZX0_ULTRA_EXE, "--pack", "--compat82", "-o", gen_comp_p, "--raw-out", gen_raw_p, *gen_paths],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        gen_comp = open(gen_comp_p, "rb").read()
        gen_raw = open(gen_raw_p, "rb").read()
        gen_unpacked = emulate_x86_decompressor(dec_compat82, gen_comp, len(gen_raw))
        assert sha256_hex(gen_unpacked) == sha256_hex(gen_raw)
        for gname, gbytes in gen_assets:
            assert gbytes in gen_unpacked, f"Missing slice {gname} in general multi-asset pack!"
        unicorn_checks += 1
        print(
            f"  [UNICORN x86-16 SHA-256 VERIFIED] 5 heterogeneous non-sprite blocks: "
            f"raw={sum(len(b) for _, b in gen_assets)}B -> merged={len(gen_raw)}B -> compressed={len(gen_comp)}B "
            f"(SHA-256={sha256_hex(gen_unpacked)[:16]})"
        )

        print("\n" + "=" * 112)
        print(f"ALL {unicorn_checks} UNICORN x86-16 CRYPTOGRAPHIC SHA-256 VERIFICATIONS PASSED!")
        print("=" * 112)


if __name__ == "__main__":
    main()
