/*
 * sprite_solver.c - Multi-Core Multi-Asset Superstring, Partial/Maximal Boundary Overlap,
 *                   Substring Containment, and Wildcard-Bridge Optimizer.
 *
 * Author & Implementation: Oleksandr Dreval (MIT License).
 * SPDX-License-Identifier: MIT
 */

#include "zx0_ultra.h"

#include <assert.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PACK_ASSETS             12
#define ELITE_BYTE_SLACK            4
#define MAX_ELITE_PERMUTATIONS      256
#define MAX_WILDCARD_BRIDGE_LEN     4

static int compute_max_overlap(const UltraAsset *left, const UltraAsset *right) {
    int limit = left->length < right->length ? left->length : right->length;
    /* Do not merge completely here; full containment is handled separately */
    if (limit == right->length && left->length == right->length) {
        limit--;
    }
    for (int ov = limit; ov >= 1; ov--) {
        if (memcmp(left->data + left->length - ov, right->data, (size_t)ov) == 0) {
            return ov;
        }
    }
    return 0;
}

static int find_contained_offset(const UltraAsset *haystack, const UltraAsset *needle) {
    if (needle->length > haystack->length) {
        return -1;
    }
    int max_start = haystack->length - needle->length;
    for (int pos = 0; pos <= max_start; pos++) {
        if (memcmp(haystack->data + pos, needle->data, (size_t)needle->length) == 0) {
            return pos;
        }
    }
    return -1;
}

/*
 * Gap specification between consecutive active assets:
 *   gap_spec >= 0 : suffix-prefix overlap of gap_spec bytes (0 = direct concatenation)
 *   gap_spec <  0 : insert (-gap_spec) flexible wildcard bridge bytes between the two assets
 */
static int build_candidate_buffer(
    const UltraAsset *assets,
    const int *active_perm,
    int active_count,
    const int *gap_specs,
    uint8_t *raw_out,
    uint8_t *wildcard_out,
    int *active_offsets_out
) {
    int cursor = 0;
    for (int idx = 0; idx < active_count; idx++) {
        const UltraAsset *cur = &assets[active_perm[idx]];
        if (idx == 0) {
            active_offsets_out[idx] = 0;
            memcpy(raw_out, cur->data, (size_t)cur->length);
            memset(wildcard_out, 0, (size_t)cur->length);
            cursor = cur->length;
        } else {
            int spec = gap_specs[idx - 1];
            if (spec >= 0) {
                int ov = spec;
                active_offsets_out[idx] = cursor - ov;
                int tail_len = cur->length - ov;
                if (tail_len > 0) {
                    memcpy(raw_out + cursor, cur->data + ov, (size_t)tail_len);
                    memset(wildcard_out + cursor, 0, (size_t)tail_len);
                    cursor += tail_len;
                }
            } else {
                int bridge_len = -spec;
                memset(raw_out + cursor, 0, (size_t)bridge_len);
                memset(wildcard_out + cursor, 1, (size_t)bridge_len);
                cursor += bridge_len;
                active_offsets_out[idx] = cursor;
                memcpy(raw_out + cursor, cur->data, (size_t)cur->length);
                memset(wildcard_out + cursor, 0, (size_t)cur->length);
                cursor += cur->length;
            }
        }
    }
    return cursor;
}

static int next_lexicographic_permutation(int *arr, int n) {
    int i = n - 2;
    while (i >= 0 && arr[i] >= arr[i + 1]) {
        i--;
    }
    if (i < 0) {
        return 0;
    }
    int j = n - 1;
    while (arr[j] <= arr[i]) {
        j--;
    }
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;

    int left = i + 1;
    int right = n - 1;
    while (left < right) {
        tmp = arr[left];
        arr[left] = arr[right];
        arr[right] = tmp;
        left++;
        right--;
    }
    return 1;
}

typedef struct {
    int perm[MAX_PACK_ASSETS];
    size_t best_bytes;
    int best_bits;
} PermScore;

static void update_global_best(
    UltraPackSolution *best_sol,
    int asset_count,
    const int *contained_in,
    const int *contained_rel_off,
    const int *active_perm,
    int active_count,
    const int *active_offsets,
    int total_raw_size,
    uint8_t *resolved_raw,
    uint8_t *comp_stream,
    size_t comp_size,
    const UltraMetadata *meta
) {
    int is_better = 0;
    if (best_sol->compressed_stream == NULL || comp_size < best_sol->compressed_size) {
        is_better = 1;
    } else if (comp_size == best_sol->compressed_size) {
        if (meta->compressed_bits < best_sol->metadata.compressed_bits ||
            (meta->compressed_bits == best_sol->metadata.compressed_bits &&
             total_raw_size < best_sol->total_uncompressed_size)) {
            is_better = 1;
        }
    }

    if (is_better) {
        free(best_sol->compressed_stream);
        free(best_sol->uncompressed_buffer);

        best_sol->compressed_stream = comp_stream;
        best_sol->compressed_size = comp_size;
        best_sol->uncompressed_buffer = resolved_raw;
        best_sol->total_uncompressed_size = total_raw_size;
        best_sol->metadata = *meta;

        for (int i = 0; i < active_count; i++) {
            int orig_id = active_perm[i];
            best_sol->permutation[i] = orig_id;
            best_sol->asset_offsets[orig_id] = active_offsets[i];
        }
        int out_p = active_count;
        for (int a = 0; a < asset_count; a++) {
            if (contained_in[a] >= 0) {
                best_sol->permutation[out_p++] = a;
                best_sol->asset_offsets[a] = best_sol->asset_offsets[contained_in[a]] + contained_rel_off[a];
            }
        }
    } else {
        free(comp_stream);
        free(resolved_raw);
    }
}

int ultra_solve_multi_asset_pack(
    const UltraAsset *assets,
    int asset_count,
    UltraFormat format,
    int enable_wildcard_bridges,
    UltraPackSolution *solution_out
) {
    if (assets == NULL || asset_count <= 0 || asset_count > MAX_PACK_ASSETS || solution_out == NULL) {
        return 0;
    }
    memset(solution_out, 0, sizeof(*solution_out));

    /* Step 1: Detect any assets completely contained inside a larger asset */
    int contained_in[MAX_PACK_ASSETS];
    int contained_rel_off[MAX_PACK_ASSETS];
    int active_ids[MAX_PACK_ASSETS];
    int active_count = 0;

    for (int i = 0; i < asset_count; i++) {
        contained_in[i] = -1;
        contained_rel_off[i] = 0;
        for (int j = 0; j < asset_count; j++) {
            if (i == j) {
                continue;
            }
            if (assets[j].length > assets[i].length ||
                (assets[j].length == assets[i].length && j < i)) {
                int rel = find_contained_offset(&assets[j], &assets[i]);
                if (rel >= 0) {
                    contained_in[i] = j;
                    contained_rel_off[i] = rel;
                    break;
                }
            }
        }
        if (contained_in[i] < 0) {
            active_ids[active_count++] = i;
        }
    }

    int max_raw_capacity = 0;
    for (int i = 0; i < active_count; i++) {
        max_raw_capacity += assets[active_ids[i]].length + MAX_WILDCARD_BRIDGE_LEN + 4;
    }

    /* Precompute pairwise maximal suffix-prefix overlaps */
    int max_ov_table[MAX_PACK_ASSETS][MAX_PACK_ASSETS];
    memset(max_ov_table, 0, sizeof(max_ov_table));
    for (int i = 0; i < asset_count; i++) {
        for (int j = 0; j < asset_count; j++) {
            if (i != j) {
                max_ov_table[i][j] = compute_max_overlap(&assets[i], &assets[j]);
            }
        }
    }

    /* Enumerate all permutations of active_ids */
    size_t total_perms = 1;
    for (int k = 2; k <= active_count; k++) {
        total_perms *= (size_t)k;
    }
    int (*all_perms)[MAX_PACK_ASSETS] = malloc(total_perms * sizeof(*all_perms));
    PermScore *perm_scores = (PermScore *)malloc(total_perms * sizeof(PermScore));
    assert(all_perms != NULL && perm_scores != NULL);

    int cur_perm[MAX_PACK_ASSETS];
    memcpy(cur_perm, active_ids, (size_t)active_count * sizeof(int));
    size_t p_idx = 0;
    do {
        memcpy(all_perms[p_idx], cur_perm, (size_t)active_count * sizeof(int));
        p_idx++;
    } while (next_lexicographic_permutation(cur_perm, active_count));
    assert(p_idx == total_perms);

    /*
     * Phase 1: Parallel screening of all permutations (both maximal-overlap and zero-overlap).
     * When total_perms > MAX_ELITE_PERMUTATIONS and format uses multi-stride search, screen with
     * exact ZX0_V2 first to avoid redundant multi-stride sweeps on non-competitive permutations,
     * then evaluate the full target format exhaustively across all elite candidates in Phase 2.
     */
    int use_fast_phase1_screen = (total_perms > MAX_ELITE_PERMUTATIONS) &&
                                 (format == ULTRA_FORMAT_STRD_80B ||
                                  format == ULTRA_FORMAT_STRD_82B ||
                                  format == ULTRA_FORMAT_COMPAT_82B);
    UltraFormat phase1_format = use_fast_phase1_screen ? ULTRA_FORMAT_ZX0_V2 : format;

    #pragma omp parallel for schedule(dynamic, 16)
    for (int idx = 0; idx < (int)total_perms; idx++) {
        uint8_t *raw_buf = (uint8_t *)malloc((size_t)max_raw_capacity);
        uint8_t *wc_buf = (uint8_t *)malloc((size_t)max_raw_capacity);
        int gap_specs[MAX_PACK_ASSETS];
        int active_offs[MAX_PACK_ASSETS];
        assert(raw_buf != NULL && wc_buf != NULL);

        size_t local_best_bytes = (size_t)-1;
        int local_best_bits = 0x3fffffff;

        for (int mode = 0; mode < 2; mode++) {
            int has_any_ov = 0;
            for (int g = 0; g < active_count - 1; g++) {
                int ov = max_ov_table[all_perms[idx][g]][all_perms[idx][g + 1]];
                gap_specs[g] = (mode == 0) ? ov : 0;
                if (ov > 0) {
                    has_any_ov = 1;
                }
            }
            if (mode == 1 && !has_any_ov) {
                continue;
            }

            int raw_len = build_candidate_buffer(
                assets, all_perms[idx], active_count, gap_specs, raw_buf, wc_buf, active_offs
            );
            uint8_t *resolved = (uint8_t *)malloc((size_t)raw_len);
            assert(resolved != NULL);

            size_t comp_sz = 0;
            UltraMetadata meta;
            uint8_t *comp = ultra_compress_buffer(
                raw_buf, raw_len, NULL, phase1_format, ZX0_MAX_OFFSET, &comp_sz, &meta, resolved
            );

            if (comp_sz < local_best_bytes ||
                (comp_sz == local_best_bytes && meta.compressed_bits < local_best_bits)) {
                local_best_bytes = comp_sz;
                local_best_bits = meta.compressed_bits;
            }

            #pragma omp critical
            {
                update_global_best(
                    solution_out, asset_count, contained_in, contained_rel_off,
                    all_perms[idx], active_count, active_offs, raw_len,
                    resolved, comp, comp_sz, &meta
                );
            }
        }

        memcpy(perm_scores[idx].perm, all_perms[idx], (size_t)active_count * sizeof(int));
        perm_scores[idx].best_bytes = local_best_bytes;
        perm_scores[idx].best_bits = local_best_bits;

        free(raw_buf);
        free(wc_buf);
    }

    /* Collect elite permutations within ELITE_BYTE_SLACK of Phase 1 best */
    size_t phase1_min_bytes = solution_out->compressed_size;
    if (use_fast_phase1_screen) {
        ultra_free_pack_solution(solution_out);
    }
    int (*elite_perms)[MAX_PACK_ASSETS] = malloc(MAX_ELITE_PERMUTATIONS * sizeof(*elite_perms));
    assert(elite_perms != NULL);
    int elite_count = 0;

    for (size_t threshold = phase1_min_bytes;
         threshold <= phase1_min_bytes + ELITE_BYTE_SLACK && elite_count < MAX_ELITE_PERMUTATIONS;
         threshold++) {
        for (size_t idx = 0; idx < total_perms && elite_count < MAX_ELITE_PERMUTATIONS; idx++) {
            if (perm_scores[idx].best_bytes == threshold) {
                memcpy(elite_perms[elite_count++], perm_scores[idx].perm, (size_t)active_count * sizeof(int));
            }
        }
    }

    /* Phase 2: Exhaustive partial-overlap bitmasks + 1..4 byte Wildcard Bridges on elite permutations */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int e = 0; e < elite_count; e++) {
        const int *perm = elite_perms[e];
        uint8_t *raw_buf = (uint8_t *)malloc((size_t)max_raw_capacity);
        uint8_t *wc_buf = (uint8_t *)malloc((size_t)max_raw_capacity);
        int base_gaps[MAX_PACK_ASSETS];
        int active_offs[MAX_PACK_ASSETS];
        assert(raw_buf != NULL && wc_buf != NULL);

        /* 2A: Enumerate all partial overlap combinations (0 .. max_ov_g) across all boundaries */
        int total_ov_combos = 1;
        int radices[MAX_PACK_ASSETS];
        for (int g = 0; g < active_count - 1; g++) {
            radices[g] = max_ov_table[perm[g]][perm[g + 1]] + 1;
            if (total_ov_combos <= 256) {
                total_ov_combos *= radices[g];
            } else {
                radices[g] = 1;
            }
        }

        for (int combo = 0; combo < total_ov_combos; combo++) {
            int rem = combo;
            for (int g = 0; g < active_count - 1; g++) {
                base_gaps[g] = rem % radices[g];
                rem /= radices[g];
            }

            int raw_len = build_candidate_buffer(
                assets, perm, active_count, base_gaps, raw_buf, wc_buf, active_offs
            );
            uint8_t *resolved = (uint8_t *)malloc((size_t)raw_len);
            assert(resolved != NULL);
            size_t comp_sz = 0;
            UltraMetadata meta;
            uint8_t *comp = ultra_compress_buffer(raw_buf, raw_len, NULL, format, ZX0_MAX_OFFSET, &comp_sz, &meta, resolved);

            #pragma omp critical
            {
                update_global_best(
                    solution_out, asset_count, contained_in, contained_rel_off,
                    perm, active_count, active_offs, raw_len,
                    resolved, comp, comp_sz, &meta
                );
            }
        }

        /* 2B: Evaluate Wildcard ("Don't-Care") Synthetic Bridges of 1..MAX_WILDCARD_BRIDGE_LEN bytes */
        if (enable_wildcard_bridges) {
            for (int g = 0; g < active_count - 1; g++) {
                base_gaps[g] = max_ov_table[perm[g]][perm[g + 1]];
            }
            for (int bridge_slot = 0; bridge_slot < active_count - 1; bridge_slot++) {
                int saved = base_gaps[bridge_slot];
                for (int wlen = 1; wlen <= MAX_WILDCARD_BRIDGE_LEN; wlen++) {
                    base_gaps[bridge_slot] = -wlen;
                    int raw_len = build_candidate_buffer(
                        assets, perm, active_count, base_gaps, raw_buf, wc_buf, active_offs
                    );
                    uint8_t *resolved = (uint8_t *)malloc((size_t)raw_len);
                    assert(resolved != NULL);
                    size_t comp_sz = 0;
                    UltraMetadata meta;
                    uint8_t *comp = ultra_compress_buffer(
                        raw_buf, raw_len, wc_buf, format, ZX0_MAX_OFFSET, &comp_sz, &meta, resolved
                    );

                    #pragma omp critical
                    {
                        update_global_best(
                            solution_out, asset_count, contained_in, contained_rel_off,
                            perm, active_count, active_offs, raw_len,
                            resolved, comp, comp_sz, &meta
                        );
                    }
                }
                base_gaps[bridge_slot] = saved;
            }
        }

        free(raw_buf);
        free(wc_buf);
    }

    free(elite_perms);
    free(perm_scores);
    free(all_perms);

    /* Mandatory cryptographic-grade slice verification of every original asset */
    assert(solution_out->compressed_stream != NULL);
    uint8_t *decomp = ultra_decompress_buffer(
        solution_out->compressed_stream,
        solution_out->compressed_size,
        (size_t)solution_out->total_uncompressed_size,
        format
    );
    assert(decomp != NULL);
    for (int a = 0; a < asset_count; a++) {
        int off = solution_out->asset_offsets[a];
        assert(off >= 0 && off + assets[a].length <= solution_out->total_uncompressed_size);
        assert(memcmp(decomp + off, assets[a].data, (size_t)assets[a].length) == 0);
    }
    free(decomp);

    return 1;
}

void ultra_free_pack_solution(UltraPackSolution *solution) {
    if (solution != NULL) {
        free(solution->compressed_stream);
        free(solution->uncompressed_buffer);
        memset(solution, 0, sizeof(*solution));
    }
}
