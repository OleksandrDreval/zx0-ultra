/*
 * zx0_ultra_cli.c - Command-Line Interface for ZX0-Ultra Compressor, Decompressor,
 *                   and Multi-Core Multi-Asset Superstring/Wildcard Pack Solver.
 *
 * Author & Implementation: Oleksandr Dreval (MIT License).
 * SPDX-License-Identifier: MIT
 */

#include "zx0_ultra.h"

#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
static inline double omp_get_wtime(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}
static inline int omp_get_max_threads(void) {
    return 1;
}
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *read_entire_file(const char *path, size_t *size_out) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", path);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len <= 0) {
        fprintf(stderr, "Error: Input file '%s' is empty or unreadable\n", path);
        fclose(fp);
        return NULL;
    }
    rewind(fp);

    uint8_t *buf = (uint8_t *)malloc((size_t)len);
    if (buf == NULL) {
        fclose(fp);
        return NULL;
    }
    if (fread(buf, 1, (size_t)len, fp) != (size_t)len) {
        fprintf(stderr, "Error: Short read on '%s'\n", path);
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *size_out = (size_t)len;
    return buf;
}

static int write_entire_file(const char *path, const uint8_t *data, size_t size) {
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open output file '%s' for writing\n", path);
        return 0;
    }
    if (size > 0 && fwrite(data, 1, size, fp) != size) {
        fprintf(stderr, "Error: Failed writing to '%s'\n", path);
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

static const char *format_label(UltraFormat fmt) {
    switch (fmt) {
        case ULTRA_FORMAT_ZX0_V2:      return "ZX0-v2 (67B/70B compat)";
        case ULTRA_FORMAT_ADPK_77B:    return "ADPK-78B (Adaptive-k 1..12)";
        case ULTRA_FORMAT_STRD_80B:    return "STRD-80B (Order-1 Stride)";
        case ULTRA_FORMAT_STRD_82B:    return "STRD-82B (Order-M Stride)";
        case ULTRA_FORMAT_COMPAT_82B:  return "COMPAT-82B (0B-Ovh + Stride)";
        case ULTRA_FORMAT_X86E8_84B:   return "X86E8-84B (0B-Ovh + E8 Call)";
        case ULTRA_FORMAT_UNIV_93B:    return "UNIV-93B (0B-Ovh + Stride + E8)";
        case ULTRA_FORMAT_HYBRID_108B: return "HYBRID-108B (0B-Ovh + E8/E9 + 2D Stride)";
        case ULTRA_FORMAT_PRO_111B:    return "PRO-111B (0B-Ovh + E8/E9 + 16b/8b 2D Stride)";
        case ULTRA_FORMAT_ULTRA_128B:  return "ULTRA-128B (VarLit + E8/E9 + 16b/8b 2D Stride)";
        default:                       return "Unknown";
    }
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "ZX0-Ultra: Multi-Core Exact Optimal Compressor & Superstring Pack Solver\n"
        "Usage:\n"
        "  %s [options] <input_file> <output_file>\n"
        "  %s --decompress [format_flag] --raw-size <N> <input.zx0u> <output.bin>\n"
        "  %s --pack [format_flag] [--no-wildcards] -o <output.zx0u> [--raw-out <layout.bin>] <asset1> <asset2> ...\n"
        "\nFormat flags:\n"
        "  --zx0        Standard ZX0 v2 bitstream (default; 100%% compatible with 67B/70B unzx0_x86.asm)\n"
        "  --adpk77     Adaptive k-bit (k=1..12) offset LSB bitstream (78-byte unzx0_adpk_77b.asm)\n"
        "  --strd80     1st-order arbitrary 16-bit Stride-Delta post-filter (80-byte unzx0_strd_80b.asm)\n"
        "  --compat82   Zero-overhead ZX0-compatible + Order-1 Stride-Delta (82-byte unzx0_compat_82b.asm)\n"
        "  --strd82     Multi-pass Order-1..5 Stride-Delta post-filter (82-byte unzx0_strd_82b.asm)\n"
        "  --x86e8      Zero-overhead ZX0-compatible + x86 E8 Call-Target filter (84-byte unzx0_x86e8_84b.asm)\n"
        "  --univ93     Universal Zero-overhead ZX0 + Stride-Delta + x86 E8 filter (93-byte unzx0_univ_93b.asm)\n"
        "  --hybrid108  Multi-Stage Hybrid 2D 0B-Ovh ZX0 + E8/E9 + Free-BX Stride (108-byte unzx0_hybrid_108b.asm)\n"
        "  --pro111     Zero-Overhead ZX0 + E8/E9 + 16-Bit/8-Bit 2D Stride (111-byte unzx0_pro_111b.asm)\n"
        "  --ultra128   Flagship Variable-Bit Literals + E8/E9 + 16-Bit/8-Bit 2D Stride (128-byte unzx0_ultra_128b.asm)\n",
        prog, prog, prog
    );
}

#define CLI_MAX_PACK_ASSETS 14

static int handle_pack_command(
    UltraFormat format,
    int enable_wildcards,
    const char *out_comp_path,
    const char *out_raw_path,
    char **asset_paths,
    int asset_count
) {
    if (asset_count < 1 || asset_count > CLI_MAX_PACK_ASSETS || out_comp_path == NULL) {
        fprintf(stderr, "Error: --pack requires -o <output> and between 1 and %d asset files\n",
                CLI_MAX_PACK_ASSETS);
        return 1;
    }

    UltraAsset assets[CLI_MAX_PACK_ASSETS];
    uint8_t *buffers[CLI_MAX_PACK_ASSETS] = {NULL};

    for (int i = 0; i < asset_count; i++) {
        size_t sz = 0;
        buffers[i] = read_entire_file(asset_paths[i], &sz);
        if (buffers[i] == NULL) {
            for (int k = 0; k < i; k++) {
                free(buffers[k]);
            }
            return 1;
        }
        assets[i].name = asset_paths[i];
        assets[i].data = buffers[i];
        assets[i].length = (int)sz;
    }

    double t0 = omp_get_wtime();
    UltraPackSolution sol;
    int ok = ultra_solve_multi_asset_pack(assets, asset_count, format, enable_wildcards, &sol);
    double elapsed = omp_get_wtime() - t0;

    if (!ok) {
        fprintf(stderr, "Error: Multi-asset pack optimization failed\n");
        for (int i = 0; i < asset_count; i++) {
            free(buffers[i]);
        }
        return 1;
    }

    int sum_raw = 0;
    for (int i = 0; i < asset_count; i++) {
        sum_raw += assets[i].length;
    }

    printf("[%s] Solved %d-asset pack in %.3fs across %d threads:\n",
           format_label(format), asset_count, elapsed, omp_get_max_threads());
    printf("  Raw sum: %d B -> Merged layout: %d B -> Compressed: %zu B (%d payload bits)\n",
           sum_raw, sol.total_uncompressed_size, sol.compressed_size, sol.metadata.compressed_bits);
    if (format == ULTRA_FORMAT_ADPK_77B) {
        printf("  Selected offset_lsb_bits (k): %d\n", sol.metadata.offset_lsb_bits);
    } else if ((format == ULTRA_FORMAT_X86E8_84B ||
                (format == ULTRA_FORMAT_UNIV_93B && sol.metadata.filter_stride == 0)) &&
               sol.metadata.filter_passes > 0) {
        printf("  Selected x86 E8 Call-Target tail: %d B\n", sol.metadata.filter_counts[0]);
    } else if (sol.metadata.filter_passes > 0) {
        printf("  Selected Stride-Delta: passes=%d, stride=%d, tail=%d B\n",
               sol.metadata.filter_passes, sol.metadata.filter_stride, sol.metadata.filter_counts[0]);
    }
    for (int i = 0; i < asset_count; i++) {
        printf("  Asset[%d] %-28s : offset +%4d (0x%03X), len %3d B\n",
               i, assets[i].name, sol.asset_offsets[i], sol.asset_offsets[i], assets[i].length);
    }

    int rc = 0;
    if (!write_entire_file(out_comp_path, sol.compressed_stream, sol.compressed_size)) {
        rc = 1;
    }
    if (rc == 0 && out_raw_path != NULL) {
        if (!write_entire_file(out_raw_path, sol.uncompressed_buffer, (size_t)sol.total_uncompressed_size)) {
            rc = 1;
        }
    }

    ultra_free_pack_solution(&sol);
    for (int i = 0; i < asset_count; i++) {
        free(buffers[i]);
    }
    return rc;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    UltraFormat format = ULTRA_FORMAT_ZX0_V2;
    int decompress_mode = 0;
    int pack_mode = 0;
    int enable_wildcards = 1;
    size_t expected_raw_size = 0;
    const char *pack_out_path = NULL;
    const char *pack_raw_out_path = NULL;

    char *positional[20];
    int pos_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--zx0") == 0) {
            format = ULTRA_FORMAT_ZX0_V2;
        } else if (strcmp(argv[i], "--adpk77") == 0) {
            format = ULTRA_FORMAT_ADPK_77B;
        } else if (strcmp(argv[i], "--strd80") == 0) {
            format = ULTRA_FORMAT_STRD_80B;
        } else if (strcmp(argv[i], "--compat82") == 0) {
            format = ULTRA_FORMAT_COMPAT_82B;
        } else if (strcmp(argv[i], "--strd82") == 0) {
            format = ULTRA_FORMAT_STRD_82B;
        } else if (strcmp(argv[i], "--x86e8") == 0) {
            format = ULTRA_FORMAT_X86E8_84B;
        } else if (strcmp(argv[i], "--univ93") == 0) {
            format = ULTRA_FORMAT_UNIV_93B;
        } else if (strcmp(argv[i], "--hybrid108") == 0 || strcmp(argv[i], "--hybrid110") == 0) {
            format = ULTRA_FORMAT_HYBRID_108B;
        } else if (strcmp(argv[i], "--pro111") == 0) {
            format = ULTRA_FORMAT_PRO_111B;
        } else if (strcmp(argv[i], "--ultra128") == 0) {
            format = ULTRA_FORMAT_ULTRA_128B;
        } else if (strcmp(argv[i], "--decompress") == 0 || strcmp(argv[i], "-d") == 0) {
            decompress_mode = 1;
        } else if (strcmp(argv[i], "--pack") == 0) {
            pack_mode = 1;
        } else if (strcmp(argv[i], "--no-wildcards") == 0) {
            enable_wildcards = 0;
        } else if (strcmp(argv[i], "--raw-size") == 0 && i + 1 < argc) {
            expected_raw_size = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            pack_out_path = argv[++i];
        } else if (strcmp(argv[i], "--raw-out") == 0 && i + 1 < argc) {
            pack_raw_out_path = argv[++i];
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        } else {
            if (pos_count < 20) {
                positional[pos_count++] = argv[i];
            }
        }
    }

    if (pack_mode) {
        return handle_pack_command(
            format, enable_wildcards, pack_out_path, pack_raw_out_path, positional, pos_count
        );
    }

    if (pos_count != 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *in_path = positional[0];
    const char *out_path = positional[1];

    if (decompress_mode) {
        if (expected_raw_size == 0) {
            fprintf(stderr, "Error: --decompress requires --raw-size <N>\n");
            return 1;
        }
        size_t comp_size = 0;
        uint8_t *comp_buf = read_entire_file(in_path, &comp_size);
        if (comp_buf == NULL) {
            return 1;
        }
        uint8_t *decomp = ultra_decompress_buffer(comp_buf, comp_size, expected_raw_size, format);
        free(comp_buf);
        if (decomp == NULL) {
            fprintf(stderr, "Error: Decompression failed or stream corrupted\n");
            return 1;
        }
        int ok = write_entire_file(out_path, decomp, expected_raw_size);
        free(decomp);
        return ok ? 0 : 1;
    }

    size_t raw_size = 0;
    uint8_t *raw_buf = read_entire_file(in_path, &raw_size);
    if (raw_buf == NULL) {
        return 1;
    }

    double t0 = omp_get_wtime();
    size_t comp_size = 0;
    UltraMetadata meta;
    uint8_t *comp_buf = ultra_compress_buffer(
        raw_buf, (int)raw_size, NULL, format, ZX0_MAX_OFFSET, &comp_size, &meta, NULL
    );
    double elapsed = omp_get_wtime() - t0;

    printf("[%s] Compressed %zu -> %zu bytes (%.2f%%, %d bits) in %.3fs",
           format_label(format), raw_size, comp_size,
           100.0 * (double)comp_size / (double)raw_size,
           meta.compressed_bits, elapsed);
    if (format == ULTRA_FORMAT_ULTRA_128B && meta.literal_bits > 0 && meta.literal_bits < 8) {
        printf(" [lit_bits=%d]", meta.literal_bits);
    }
    if (format == ULTRA_FORMAT_ADPK_77B) {
        printf(" [k=%d]", meta.offset_lsb_bits);
    } else if ((format == ULTRA_FORMAT_X86E8_84B ||
                ((format == ULTRA_FORMAT_UNIV_93B || format == ULTRA_FORMAT_HYBRID_108B || format == ULTRA_FORMAT_PRO_111B || format == ULTRA_FORMAT_ULTRA_128B) && meta.filter_stride == 0)) &&
               meta.filter_passes > 0) {
        if ((format == ULTRA_FORMAT_HYBRID_108B || format == ULTRA_FORMAT_PRO_111B || format == ULTRA_FORMAT_ULTRA_128B) && meta.filter_passes > 2) {
            printf(" [e8_tail=%d, stride=%d, passes=%d, tail=%d]",
                   meta.filter_counts[0], meta.filter_strides[2], meta.filter_passes - 2, meta.filter_counts[2]);
        } else {
            printf(" [e8_tail=%d]", meta.filter_counts[0]);
        }
    } else if (meta.filter_passes > 0) {
        printf(" [stride=%d, passes=%d, tail=%d]", meta.filter_stride, meta.filter_passes, meta.filter_counts[0]);
    }
    printf("\n");

    int ok = write_entire_file(out_path, comp_buf, comp_size);
    free(comp_buf);
    free(raw_buf);
    return ok ? 0 : 1;
}
