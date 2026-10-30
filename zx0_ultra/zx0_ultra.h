/*
 * zx0_ultra.h - Public C11 API for ZX0-Ultra Exact Optimal Parser, OpenMP Transform Search,
 *               and Multi-Asset Superstring/Wildcard Pack Optimizer.
 *
 * Based on the ZX0 compression format & engine by Einar Saukas (c) 2021 (BSD 3-Clause).
 * Exact multi-state parser architecture, extended bitstream formats, and multi-asset API
 * by Oleksandr Dreval (MIT License).
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ZX0_ULTRA_H
#define ZX0_ULTRA_H

#include <stddef.h>
#include <stdint.h>

/* Standard ZX0 v2 architectural constants */
#define ZX0_INITIAL_OFFSET          1
#define ZX0_DEFAULT_LSB_BITS        7
#define ZX0_MAX_OFFSET              32640
#define ZX0_EOF_MSB_VALUE           256
#define ZX0_EOF_ALT_VALUE           257
#define ZX0_EOF_E8_VALUE            281
#define ZX0_LITERAL_BITS_PER_BYTE   8

/* Monotonic literal-frontier capacity: Elias gamma concavity guarantees <= 12 undominated split points */
#define ULTRA_MAX_FRONTIER_SIZE     24
#define ULTRA_MAX_FILTER_PASSES     5

/* Supported bitstream formats corresponding to verified x86-16 assembly decompressors */
typedef enum {
    ULTRA_FORMAT_ZX0_V2     = 0, /* 100% compatible with 67B/70B unzx0_x86.asm & original dzx0 */
    ULTRA_FORMAT_ADPK_77B   = 1, /* 78-byte Adaptive-k offset LSB width (k in 1..12) */
    ULTRA_FORMAT_STRD_80B   = 2, /* 80-byte 1st-order arbitrary 16-bit Stride-Delta post-filter */
    ULTRA_FORMAT_STRD_82B   = 3, /* 82-byte Multi-pass (Order-1..5) Stride-Delta post-filter */
    ULTRA_FORMAT_COMPAT_82B = 4, /* 82-byte Zero-Overhead ZX0-compatible + Order-1 Stride-Delta */
    ULTRA_FORMAT_X86E8_84B  = 5, /* 84-byte Zero-Overhead ZX0-compatible + x86 E8 Call-Target filter */
    ULTRA_FORMAT_UNIV_93B   = 6  /* 93-byte Universal Zero-Overhead ZX0 + Stride-Delta + x86 E8 filter */
} UltraFormat;

typedef struct {
    UltraFormat format;
    int offset_lsb_bits;                        /* 7 for standard ZX0; 1..12 for ULTRA_FORMAT_ADPK_77B */
    int filter_passes;                          /* 0 = raw, 1..5 = active post-filter passes */
    int filter_stride;                          /* Stride distance in bytes (0 for E8 call filter) */
    int filter_counts[ULTRA_MAX_FILTER_PASSES]; /* Filtered tail byte count per integration pass */
    int compressed_bits;                        /* Exact payload bit count prior to byte alignment */
} UltraMetadata;

/*
 * Compress an input buffer using exact unpruned multi-state DP and OpenMP parameter search.
 * If wildcard_mask is non-NULL, positions with wildcard_mask[i] != 0 are treated as flexible
 * "don't-care" bytes whose resolved values are written into resolved_input_out (if non-NULL).
 */
uint8_t *ultra_compress_buffer(
    const uint8_t *input_data,
    int input_size,
    const uint8_t *wildcard_mask,
    UltraFormat format,
    int max_offset_limit,
    size_t *compressed_size_out,
    UltraMetadata *metadata_out,
    uint8_t *resolved_input_out
);

/*
 * Decompress any ZX0_V2, ADPK_77B, STRD_80B, or STRD_82B buffer in memory for self-verification.
 * Returns newly allocated output buffer of size expected_output_size, or NULL on error.
 */
uint8_t *ultra_decompress_buffer(
    const uint8_t *compressed_data,
    size_t compressed_size,
    size_t expected_output_size,
    UltraFormat format
);

/* Multi-asset sprite descriptor for permutation, boundary-overlap, and wildcard-bridge optimization */
typedef struct {
    const char *name;
    const uint8_t *data;
    int length;
} UltraAsset;

typedef struct {
    int permutation[16];
    int asset_offsets[16];
    int total_uncompressed_size;
    size_t compressed_size;
    uint8_t *compressed_stream;
    uint8_t *uncompressed_buffer;
    UltraMetadata metadata;
} UltraPackSolution;

/*
 * Find the globally minimal single-stream arrangement of asset_count assets using OpenMP
 * across permutations, partial/maximal suffix-prefix overlaps, containment, and wildcard bridges.
 */
int ultra_solve_multi_asset_pack(
    const UltraAsset *assets,
    int asset_count,
    UltraFormat format,
    int enable_wildcard_bridges,
    UltraPackSolution *solution_out
);

void ultra_free_pack_solution(UltraPackSolution *solution);

#endif /* ZX0_ULTRA_H */
