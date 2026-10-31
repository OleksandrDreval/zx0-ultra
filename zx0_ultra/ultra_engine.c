/*
 * ultra_engine.c - Exact Unpruned Multi-State Automaton DP, Bitstream Encoder,
 *                  OpenMP Transform Search, and In-Memory Self-Verifier.
 *
 * Based on the ZX0 compression engine by Einar Saukas (c) 2021 (BSD 3-Clause).
 * Exact Pareto-frontier optimal parser, wildcard materialization, and multi-core
 * Stride-Delta / Adaptive-k optimizations by Oleksandr Dreval (MIT License).
 *
 * SPDX-License-Identifier: MIT
 */

#include "zx0_ultra.h"

#include <assert.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_in_parallel(void) { return 0; }
#endif
#include <stdlib.h>
#include <string.h>

#define COST_INFINITY               0x3fffffff
#define EXACT_FRONTIER_CAPACITY     32
#define INITIAL_NODE_POOL_CAPACITY  65536
#define NODE_POOL_GC_THRESHOLD      (2 * 1024 * 1024)
#define EOF_CONTROL_AND_ELIAS_BITS  18
#define FILTER_WORD_BYTES           2
#define ADPK_HEADER_BYTES           1
#define MAX_ADPK_LSB_BITS           12
#define MIN_MATCH_LENGTH_NEW_OFFSET 2
#define X86_OPCODE_CALL_REL16       0xE8
#define X86_OPCODE_JMP_REL16        0xE9

typedef struct {
    int prev_node;
    int end_pos;
    int offset; /* 0 denotes a Literal run; > 0 denotes a Match with this offset */
} ParseNode;

typedef struct {
    ParseNode *nodes;
    int count;
    int capacity;
} NodePool;

typedef struct {
    int pos;
    int base_cost;
    int node_idx;
} FrontierEntry;

/* Cache-line aligned layout: count + entries[0..4] reside in a single 64-byte cache line */
typedef struct {
    int count;
    FrontierEntry entries[EXACT_FRONTIER_CAPACITY];
} MonotonicFrontier;

typedef struct {
    int end_pos;
    int length;
    int offset; /* 0 = Literal, > 0 = Match */
} Token;

static inline int compute_elias_gamma_bits(int value) {
    assert(value >= 1);
#if defined(__GNUC__) || defined(__clang__)
    return 1 + 2 * (31 - __builtin_clz((unsigned int)value));
#else
    int bits = 1;
    while (value >>= 1) {
        bits += 2;
    }
    return bits;
#endif
}

static inline int compute_new_offset_overhead_bits(int offset, int offset_lsb_bits) {
    int msb_group = ((offset - 1) >> offset_lsb_bits) + 1;
    return 1 + compute_elias_gamma_bits(msb_group) + offset_lsb_bits;
}

static void node_pool_init(NodePool *pool, int initial_capacity) {
    pool->capacity = initial_capacity > 0 ? initial_capacity : INITIAL_NODE_POOL_CAPACITY;
    pool->count = 0;
    pool->nodes = (ParseNode *)malloc((size_t)pool->capacity * sizeof(ParseNode));
    assert(pool->nodes != NULL);
}

static void node_pool_free(NodePool *pool) {
    free(pool->nodes);
    pool->nodes = NULL;
    pool->count = 0;
    pool->capacity = 0;
}

static int node_pool_push(NodePool *pool, int prev_node, int end_pos, int offset) {
    if (pool->count >= pool->capacity) {
        pool->capacity *= 2;
        pool->nodes = (ParseNode *)realloc(pool->nodes, (size_t)pool->capacity * sizeof(ParseNode));
        assert(pool->nodes != NULL);
    }
    int idx = pool->count++;
    pool->nodes[idx].prev_node = prev_node;
    pool->nodes[idx].end_pos = end_pos;
    pool->nodes[idx].offset = offset;
    return idx;
}

/*
 * Exact O(Live) Topological Mark-and-Compact Garbage Collector for NodePool.
 * Reclaims dead ParseNodes displaced from monotonic frontiers, bounding per-thread memory to ~24 MB
 * even on highly repetitive 64 KB files across 16 simultaneous OpenMP threads.
 */
static void node_pool_compact(
    NodePool *pool,
    int *best_any_node,
    int upto_pos,
    MonotonicFrontier *lit_frontiers,
    MonotonicFrontier *rep_frontiers,
    const uint8_t *rep_nonempty,
    int max_window
) {
    if (pool->count == 0) {
        return;
    }
    int *remap = (int *)malloc((size_t)pool->count * sizeof(int));
    assert(remap != NULL);
    for (int i = 0; i < pool->count; i++) {
        remap[i] = -1;
    }

    for (int p = 1; p <= upto_pos; p++) {
        int nid = best_any_node[p];
        if (nid >= 0) {
            remap[nid] = 1;
        }
    }
    for (int off = 0; off <= max_window; off++) {
        int lc = lit_frontiers[off].count;
        for (int k = 0; k < lc; k++) {
            int nid = lit_frontiers[off].entries[k].node_idx;
            if (nid >= 0) {
                remap[nid] = 1;
            }
        }
        if (off >= 1 && rep_nonempty[off]) {
            int rc = rep_frontiers[off].count;
            for (int k = 0; k < rc; k++) {
                int nid = rep_frontiers[off].entries[k].node_idx;
                if (nid >= 0) {
                    remap[nid] = 1;
                }
            }
        }
    }

    for (int idx = pool->count - 1; idx >= 0; idx--) {
        if (remap[idx] >= 0) {
            int prev = pool->nodes[idx].prev_node;
            if (prev >= 0) {
                assert(prev < idx);
                remap[prev] = 1;
            }
        }
    }

    int new_count = 0;
    for (int idx = 0; idx < pool->count; idx++) {
        if (remap[idx] >= 0) {
            ParseNode node = pool->nodes[idx];
            if (node.prev_node >= 0) {
                node.prev_node = remap[node.prev_node];
            }
            pool->nodes[new_count] = node;
            remap[idx] = new_count++;
        }
    }
    pool->count = new_count;

    for (int p = 1; p <= upto_pos; p++) {
        if (best_any_node[p] >= 0) {
            best_any_node[p] = remap[best_any_node[p]];
        }
    }
    for (int off = 0; off <= max_window; off++) {
        int lc = lit_frontiers[off].count;
        for (int k = 0; k < lc; k++) {
            int nid = lit_frontiers[off].entries[k].node_idx;
            if (nid >= 0) {
                lit_frontiers[off].entries[k].node_idx = remap[nid];
            }
        }
        if (off >= 1 && rep_nonempty[off]) {
            int rc = rep_frontiers[off].count;
            for (int k = 0; k < rc; k++) {
                int nid = rep_frontiers[off].entries[k].node_idx;
                if (nid >= 0) {
                    rep_frontiers[off].entries[k].node_idx = remap[nid];
                }
            }
        }
    }

    free(remap);
}

/*
 * Maintain strictly increasing (pos, base_cost) invariant.
 * Any older state with base_cost >= new_base_cost is provably dominated for all future positions
 * because Elias gamma bit length is monotonically non-decreasing with run length.
 */
static inline void frontier_push(MonotonicFrontier *frontier, int pos, int base_cost, int node_idx) {
    int c = frontier->count;
    while (c > 0 && frontier->entries[c - 1].base_cost >= base_cost) {
        c--;
    }
    frontier->count = c;
    if (c > 0 && (base_cost - frontier->entries[0].base_cost) > 31) {
        return;
    }
    if (c < EXACT_FRONTIER_CAPACITY) {
        frontier->entries[c].pos = pos;
        frontier->entries[c].base_cost = base_cost;
        frontier->entries[c].node_idx = node_idx;
        frontier->count = c + 1;
    }
}

static inline int frontier_evaluate_min(
    const MonotonicFrontier *frontier,
    int current_pos,
    int *best_node_out
) {
    int best_cost = COST_INFINITY;
    int best_node = -1;
    int c = frontier->count;
    for (int k = 0; k < c; k++) {
        int run_len = current_pos - frontier->entries[k].pos;
        int candidate = frontier->entries[k].base_cost + 1 + compute_elias_gamma_bits(run_len);
        if (candidate < best_cost) {
            best_cost = candidate;
            best_node = frontier->entries[k].node_idx;
        }
    }
    *best_node_out = best_node;
    return best_cost;
}

static int can_match_byte(
    const uint8_t *input_data,
    const uint8_t *wildcard_mask,
    int target_index,
    int offset
) {
    int source_index = target_index - offset;
    if (source_index < 0) {
        return 0;
    }
    if (wildcard_mask != NULL) {
        /* Never copy from an unresolved wildcard position during pass 1 */
        if (wildcard_mask[source_index] != 0) {
            return 0;
        }
        /* A wildcard target matches any concrete historical byte */
        if (wildcard_mask[target_index] != 0) {
            return 1;
        }
    }
    return input_data[target_index] == input_data[source_index];
}

typedef struct {
    Token *tokens;
    int token_count;
    int payload_bits;
} ParsePlan;

static inline int detect_buffer_literal_bits(
    const uint8_t *input_data,
    int input_size,
    const uint8_t *wildcard_mask
) {
    uint8_t max_val = 0;
    for (int i = 0; i < input_size; i++) {
        if (wildcard_mask != NULL && wildcard_mask[i] != 0) {
            continue;
        }
        if (input_data[i] > max_val) {
            max_val = input_data[i];
        }
    }
    if (max_val < 16) return 4;
    if (max_val < 32) return 5;
    if (max_val < 64) return 6;
    if (max_val < 128) return 7;
    return 8;
}

/*
 * Exact O(N * W) Multi-State Automaton DP Solver with Zero-Loss Dominated-Literal
 * Pruning, Global O(1) Literal Frontier (lit_frontiers[0]), Single-Cache-Line Frontiers,
 * and Topological Mark-and-Compact GC.
 * If required_final_offset > 0, only terminal states whose active last_offset equals
 * required_final_offset are accepted (enabling 0-byte stride passing via register BX).
 * If required_final_offset < 0, variable bit-width literals (4..8 bits) for ULTRA_128B are used.
 */
static int solve_exact_parse(
    const uint8_t *input_data,
    int input_size,
    const uint8_t *wildcard_mask,
    int offset_lsb_bits,
    int max_offset_limit,
    int required_final_offset,
    ParsePlan *plan_out
) {
    assert(input_size > 0);
    int is_ultra128 = (required_final_offset < 0);
    int lit_bits_per_byte = is_ultra128
                          ? detect_buffer_literal_bits(input_data, input_size, wildcard_mask)
                          : ZX0_LITERAL_BITS_PER_BYTE;
    if (required_final_offset < 0) {
        required_final_offset = 0;
    }
    int max_window = input_size - 1;
    if (max_window > max_offset_limit) {
        max_window = max_offset_limit;
    }
    int max_msb_offset = ((ZX0_EOF_MSB_VALUE - 1) << offset_lsb_bits);
    if (max_window > max_msb_offset) {
        max_window = max_msb_offset;
    }
    if (max_window < ZX0_INITIAL_OFFSET) {
        max_window = ZX0_INITIAL_OFFSET;
    }
    if (required_final_offset > max_window) {
        return 0;
    }

    MonotonicFrontier *lit_frontiers = (MonotonicFrontier *)calloc((size_t)(max_window + 1), sizeof(MonotonicFrontier));
    MonotonicFrontier *rep_frontiers = (MonotonicFrontier *)calloc((size_t)(max_window + 1), sizeof(MonotonicFrontier));
    int *lit_min_base = (int *)malloc((size_t)(max_window + 1) * sizeof(int));
    uint8_t *rep_nonempty = (uint8_t *)calloc((size_t)(max_window + 1), sizeof(uint8_t));
    int *new_off_overhead = (int *)malloc((size_t)(max_window + 1) * sizeof(int));
    int *match_lengths = (int *)calloc((size_t)(max_window + 1), sizeof(int));
    int *best_any_cost = (int *)malloc((size_t)(input_size + 1) * sizeof(int));
    int *best_any_node = (int *)malloc((size_t)(input_size + 1) * sizeof(int));
    int *best_len_cost = (int *)malloc((size_t)(input_size + 1) * sizeof(int));
    int *best_len_start = (int *)malloc((size_t)(input_size + 1) * sizeof(int));
    int *final_lit_cost = (int *)malloc((size_t)(max_window + 1) * sizeof(int));
    int *final_lit_node = (int *)malloc((size_t)(max_window + 1) * sizeof(int));
    int *final_mat_cost = (int *)malloc((size_t)(max_window + 1) * sizeof(int));
    int *final_mat_node = (int *)malloc((size_t)(max_window + 1) * sizeof(int));

    assert(lit_frontiers && rep_frontiers && lit_min_base && rep_nonempty && new_off_overhead && match_lengths);
    assert(best_any_cost && best_any_node && best_len_cost && best_len_start);
    assert(final_lit_cost && final_lit_node && final_mat_cost && final_mat_node);

    new_off_overhead[0] = 0;
    for (int off = 0; off <= max_window; off++) {
        if (off >= 1) {
            new_off_overhead[off] = compute_new_offset_overhead_bits(off, offset_lsb_bits);
        }
        lit_min_base[off] = COST_INFINITY;
        final_lit_cost[off] = COST_INFINITY;
        final_lit_node[off] = -1;
        final_mat_cost[off] = COST_INFINITY;
        final_mat_node[off] = -1;
    }

    NodePool pool;
    node_pool_init(&pool, input_size * 4 + 1024);
    int gc_threshold = NODE_POOL_GC_THRESHOLD;

    /* Seed initial state at pos = 0 with base_cost = -1 so first literal block has 0 control bits */
    frontier_push(&lit_frontiers[ZX0_INITIAL_OFFSET], 0, -1, -1);
    lit_min_base[ZX0_INITIAL_OFFSET] = -1;
    frontier_push(&lit_frontiers[0], 0, -1, -1);
    lit_min_base[0] = -1;
    best_any_cost[0] = COST_INFINITY;
    best_any_node[0] = -1;

    for (int pos = 1; pos <= input_size; pos++) {
        if (pool.count >= gc_threshold) {
            node_pool_compact(
                &pool, best_any_node, pos - 1, lit_frontiers, rep_frontiers, rep_nonempty, max_window
            );
            if (pool.count * 2 > gc_threshold) {
                gc_threshold = pool.count * 2;
            }
        }

        best_any_cost[pos] = COST_INFINITY;
        best_any_node[pos] = -1;

        int active_max_off = pos - 1;
        if (active_max_off > max_window) {
            active_max_off = max_window;
        }
        int loop_max_off = active_max_off < ZX0_INITIAL_OFFSET ? ZX0_INITIAL_OFFSET : active_max_off;
        int computed_best_len = 1;
        int literal_pos_bits = lit_bits_per_byte * pos;
        uint8_t cur_byte = input_data[pos - 1];
        uint8_t nxt_byte = (pos < input_size) ? input_data[pos] : 0;

        /* O(1) Global Literal Frontier evaluation covers best_any_cost[pos] across all offsets */
        int glob_src_node = -1;
        int glob_raw = frontier_evaluate_min(&lit_frontiers[0], pos, &glob_src_node);
        if (glob_raw < COST_INFINITY) {
            best_any_cost[pos] = glob_raw + literal_pos_bits;
            best_any_node[pos] = node_pool_push(&pool, glob_src_node, pos, 0);
        }

        int best_mat_cost_at_pos = COST_INFINITY;
        int best_mat_node_at_pos = -1;

        for (int off = 1; off <= loop_max_off; off++) {
            int cur_matches = (wildcard_mask == NULL)
                            ? (off <= active_max_off && input_data[pos - 1 - off] == cur_byte)
                            : (off <= active_max_off && can_match_byte(input_data, wildcard_mask, pos - 1, off));
            int next_pos_matches = (pos == input_size)
                                 ? 1
                                 : ((wildcard_mask == NULL)
                                    ? (off <= pos && input_data[pos - off] == nxt_byte)
                                    : (off <= pos && can_match_byte(input_data, wildcard_mask, pos, off)));
            int ml = 0;
            if (cur_matches) {
                ml = ++match_lengths[off];
            } else {
                if (match_lengths[off] != 0) {
                    match_lengths[off] = 0;
                }
                if (rep_nonempty[off]) {
                    rep_frontiers[off].count = 0;
                    rep_nonempty[off] = 0;
                }
                if (!next_pos_matches) {
                    continue;
                }
            }

            int min_base = lit_min_base[off];

            /* 1. Evaluate Literal run ending at pos only when needed for LAST_OFFSET or EOF */
            int lit_src_node = -1;
            int lit_cost = COST_INFINITY;
            int created_lit_node = -1;
            if (min_base < COST_INFINITY && (pos == input_size || next_pos_matches)) {
                int raw_eval = frontier_evaluate_min(&lit_frontiers[off], pos, &lit_src_node);
                if (raw_eval < COST_INFINITY) {
                    lit_cost = raw_eval + literal_pos_bits;
                    created_lit_node = node_pool_push(&pool, lit_src_node, pos, 0);
                    if (lit_cost < best_any_cost[pos]) {
                        best_any_cost[pos] = lit_cost;
                        best_any_node[pos] = created_lit_node;
                    }
                }
            }

            /* 2. Evaluate Match (LAST_OFFSET or NEW_OFFSET) ending at pos with offset = off */
            int mat_cost = COST_INFINITY;
            int mat_prev_node = -1;

            if (ml >= 1 && rep_nonempty[off]) {
                int rep_src_node = -1;
                int rep_cost = frontier_evaluate_min(&rep_frontiers[off], pos, &rep_src_node);
                if (rep_cost < mat_cost) {
                    mat_cost = rep_cost;
                    mat_prev_node = rep_src_node;
                }
            }

            if (ml >= MIN_MATCH_LENGTH_NEW_OFFSET) {
                while (computed_best_len < ml) {
                    int next_len = computed_best_len + 1;
                    int start_pos = pos - next_len;
                    int cand_cost = COST_INFINITY;
                    if (start_pos >= 1 && best_any_cost[start_pos] < COST_INFINITY) {
                        cand_cost = best_any_cost[start_pos] + compute_elias_gamma_bits(next_len - 1);
                    }
                    if (next_len == MIN_MATCH_LENGTH_NEW_OFFSET || cand_cost <= best_len_cost[computed_best_len]) {
                        best_len_cost[next_len] = cand_cost;
                        best_len_start[next_len] = start_pos;
                    } else {
                        best_len_cost[next_len] = best_len_cost[computed_best_len];
                        best_len_start[next_len] = best_len_start[computed_best_len];
                    }
                    computed_best_len = next_len;
                }
                if (best_len_cost[ml] < COST_INFINITY) {
                    int new_off_cost = best_len_cost[ml] + new_off_overhead[off];
                    if (new_off_cost < mat_cost) {
                        mat_cost = new_off_cost;
                        mat_prev_node = best_any_node[best_len_start[ml]];
                    }
                }
            }

            int created_mat_node = -1;
            if (mat_cost < COST_INFINITY) {
                int mat_base_cost = mat_cost - literal_pos_bits;
                int enters_lit_frontier = (min_base == COST_INFINITY) ||
                                          ((mat_base_cost - min_base) <= 31);
                if (pos == input_size || mat_cost < best_any_cost[pos] || enters_lit_frontier) {
                    created_mat_node = node_pool_push(&pool, mat_prev_node, pos, off);
                    if (mat_cost < best_any_cost[pos]) {
                        best_any_cost[pos] = mat_cost;
                        best_any_node[pos] = created_mat_node;
                    }
                    if (mat_cost < best_mat_cost_at_pos) {
                        best_mat_cost_at_pos = mat_cost;
                        best_mat_node_at_pos = created_mat_node;
                    }
                    if (enters_lit_frontier) {
                        frontier_push(&lit_frontiers[off], pos, mat_base_cost, created_mat_node);
                        lit_min_base[off] = lit_frontiers[off].entries[0].base_cost;
                    }
                }
            }

            /* Push literal state after match evaluation so LAST_OFFSET matches have length >= 1 */
            if (created_lit_node >= 0 && next_pos_matches) {
                frontier_push(&rep_frontiers[off], pos, lit_cost, created_lit_node);
                rep_nonempty[off] = (uint8_t)(rep_frontiers[off].count > 0);
            }

            if (pos == input_size) {
                final_lit_cost[off] = lit_cost;
                final_lit_node[off] = created_lit_node;
                final_mat_cost[off] = mat_cost;
                final_mat_node[off] = created_mat_node;
            }
        }

        if (best_mat_cost_at_pos < COST_INFINITY) {
            frontier_push(&lit_frontiers[0], pos, best_mat_cost_at_pos - literal_pos_bits, best_mat_node_at_pos);
            lit_min_base[0] = lit_frontiers[0].entries[0].base_cost;
        }
    }

    int terminal_cost = COST_INFINITY;
    int terminal_node = -1;
    if (required_final_offset > 0) {
        if (final_lit_cost[required_final_offset] < terminal_cost) {
            terminal_cost = final_lit_cost[required_final_offset];
            terminal_node = final_lit_node[required_final_offset];
        }
        if (final_mat_cost[required_final_offset] < terminal_cost) {
            terminal_cost = final_mat_cost[required_final_offset];
            terminal_node = final_mat_node[required_final_offset];
        }
    } else {
        terminal_cost = best_any_cost[input_size];
        terminal_node = best_any_node[input_size];
    }

    int success = (terminal_cost < COST_INFINITY && terminal_node >= 0);
    if (success && plan_out != NULL) {
        int chain_len = 0;
        for (int cur = terminal_node; cur >= 0; cur = pool.nodes[cur].prev_node) {
            chain_len++;
        }
        Token *tokens = (Token *)malloc((size_t)chain_len * sizeof(Token));
        assert(tokens != NULL);

        int idx = chain_len - 1;
        for (int cur = terminal_node; cur >= 0; cur = pool.nodes[cur].prev_node) {
            int prev = pool.nodes[cur].prev_node;
            int start_pos = (prev >= 0) ? pool.nodes[prev].end_pos : 0;
            tokens[idx].end_pos = pool.nodes[cur].end_pos;
            tokens[idx].length = pool.nodes[cur].end_pos - start_pos;
            tokens[idx].offset = pool.nodes[cur].offset;
            idx--;
        }
        plan_out->tokens = tokens;
        plan_out->token_count = chain_len;
        plan_out->payload_bits = terminal_cost + EOF_CONTROL_AND_ELIAS_BITS;
    }

    node_pool_free(&pool);
    free(lit_frontiers);
    free(rep_frontiers);
    free(lit_min_base);
    free(rep_nonempty);
    free(new_off_overhead);
    free(match_lengths);
    free(best_any_cost);
    free(best_any_node);
    free(best_len_cost);
    free(best_len_start);
    free(final_lit_cost);
    free(final_lit_node);
    free(final_mat_cost);
    free(final_mat_node);

    return success;
}

/* Materialize flexible wildcard bytes forward from the chosen parse plan */
static void materialize_wildcards_from_plan(
    uint8_t *buffer,
    const uint8_t *wildcard_mask,
    const ParsePlan *plan
) {
    int pos = 0;
    for (int t = 0; t < plan->token_count; t++) {
        int len = plan->tokens[t].length;
        int off = plan->tokens[t].offset;
        if (off == 0) {
            for (int k = 0; k < len; k++) {
                if (wildcard_mask[pos + k] != 0) {
                    buffer[pos + k] = 0x00;
                }
            }
        } else {
            for (int k = 0; k < len; k++) {
                buffer[pos + k] = buffer[pos + k - off];
            }
        }
        pos += len;
    }
}

typedef struct {
    uint8_t *data;
    size_t size;
    size_t capacity;
    size_t bit_byte_index;
    int bit_mask;
    int backtrack;
    int is_ultra128;
} BitWriter;

static void bit_writer_init(BitWriter *bw, size_t initial_capacity, int initial_backtrack) {
    bw->capacity = initial_capacity > 64 ? initial_capacity : 64;
    bw->data = (uint8_t *)calloc(bw->capacity, 1);
    assert(bw->data != NULL);
    bw->size = 0;
    bw->bit_byte_index = 0;
    bw->bit_mask = 0;
    bw->backtrack = initial_backtrack;
    bw->is_ultra128 = 0;
}

static void bit_writer_put_byte(BitWriter *bw, uint8_t value) {
    if (bw->size >= bw->capacity) {
        bw->capacity *= 2;
        bw->data = (uint8_t *)realloc(bw->data, bw->capacity);
        assert(bw->data != NULL);
    }
    bw->data[bw->size++] = value;
}

static void bit_writer_put_u16_le(BitWriter *bw, uint16_t value) {
    bit_writer_put_byte(bw, (uint8_t)(value & 0xFF));
    bit_writer_put_byte(bw, (uint8_t)((value >> 8) & 0xFF));
}

static void bit_writer_put_bit(BitWriter *bw, int bit) {
    if (bw->backtrack) {
        if (bit) {
            assert(bw->size > 0);
            bw->data[bw->size - 1] |= 1;
        }
        bw->backtrack = 0;
    } else {
        if (bw->bit_mask == 0 || (bw->is_ultra128 && bw->bit_byte_index == 0 && bw->bit_mask == 1)) {
            bw->bit_mask = 128;
            bw->bit_byte_index = bw->size;
            bit_writer_put_byte(bw, 0);
        }
        if (bit) {
            bw->data[bw->bit_byte_index] |= (uint8_t)bw->bit_mask;
        }
        bw->bit_mask >>= 1;
    }
}

static void bit_writer_put_elias(BitWriter *bw, int value, int invert_data_bits) {
    assert(value >= 1);
    int mask = 2;
    while (mask <= value) {
        mask <<= 1;
    }
    mask >>= 1;
    while (mask >>= 1) {
        bit_writer_put_bit(bw, 0);
        int bit = (value & mask) ? 1 : 0;
        bit_writer_put_bit(bw, invert_data_bits ? !bit : bit);
    }
    bit_writer_put_bit(bw, 1);
}

static uint8_t *encode_plan_to_stream(
    const uint8_t *working_input,
    const ParsePlan *plan,
    UltraFormat format,
    int offset_lsb_bits,
    const UltraMetadata *meta,
    size_t *out_size
) {
    int use_zx0_lsb_byte = (format != ULTRA_FORMAT_ADPK_77B);
    int total_raw_len = (plan->token_count > 0) ? plan->tokens[plan->token_count - 1].end_pos : 0;
    int lit_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                 ? ((meta->literal_bits >= 4 && meta->literal_bits <= 8)
                    ? meta->literal_bits
                    : detect_buffer_literal_bits(working_input, total_raw_len, NULL))
                 : ZX0_LITERAL_BITS_PER_BYTE;

    BitWriter bw;
    bit_writer_init(
        &bw,
        (size_t)(plan->payload_bits / 8 + 48),
        (use_zx0_lsb_byte && format != ULTRA_FORMAT_ULTRA_128B) ? 1 : 0
    );

    if (format == ULTRA_FORMAT_ADPK_77B) {
        bit_writer_put_byte(&bw, (uint8_t)offset_lsb_bits);
    } else if (format == ULTRA_FORMAT_ULTRA_128B) {
        bw.is_ultra128 = 1;
        bit_writer_put_byte(&bw, 0x01); /* Byte 0: bits 7..1 = first 7 bits, bit 0 = sentinel 1 */
        bit_writer_put_byte(&bw, (uint8_t)(1 << (8 - lit_bits))); /* Byte 1: DH sentinel */
        bw.bit_byte_index = 0;
        bw.bit_mask = 128;
    }

    int last_offset = ZX0_INITIAL_OFFSET;
    int input_pos = 0;

    for (int t = 0; t < plan->token_count; t++) {
        int length = plan->tokens[t].length;
        int offset = plan->tokens[t].offset;

        if (offset == 0) {
            if ((use_zx0_lsb_byte && format != ULTRA_FORMAT_ULTRA_128B) || t > 0) {
                bit_writer_put_bit(&bw, 0);
            }
            bit_writer_put_elias(&bw, length, 0);
            if (format == ULTRA_FORMAT_ULTRA_128B) {
                for (int i = 0; i < length; i++) {
                    uint8_t b = working_input[input_pos++];
                    for (int bit_idx = lit_bits - 1; bit_idx >= 0; bit_idx--) {
                        bit_writer_put_bit(&bw, (b >> bit_idx) & 1);
                    }
                }
            } else {
                for (int i = 0; i < length; i++) {
                    bit_writer_put_byte(&bw, working_input[input_pos++]);
                }
            }
        } else if (offset == last_offset) {
            bit_writer_put_bit(&bw, 0);
            bit_writer_put_elias(&bw, length, 0);
            input_pos += length;
        } else {
            bit_writer_put_bit(&bw, 1);
            int msb_group = ((offset - 1) >> offset_lsb_bits) + 1;
            bit_writer_put_elias(&bw, msb_group, 1);

            if (use_zx0_lsb_byte) {
                int lsb_rem = (offset - 1) & 0x7F;
                bit_writer_put_byte(&bw, (uint8_t)((127 - lsb_rem) << 1));
                bw.backtrack = 1;
            } else {
                int mask_max = (1 << offset_lsb_bits) - 1;
                int inv_lsb = mask_max - ((offset - 1) & mask_max);
                for (int b = offset_lsb_bits - 1; b >= 0; b--) {
                    bit_writer_put_bit(&bw, (inv_lsb >> b) & 1);
                }
            }
            bit_writer_put_elias(&bw, length - 1, 0);
            input_pos += length;
            last_offset = offset;
        }
    }

    /* Emit ZX0 end-of-stream marker */
    bit_writer_put_bit(&bw, 1);
    int eof_symbol = ZX0_EOF_MSB_VALUE;
    if ((format == ULTRA_FORMAT_COMPAT_82B || format == ULTRA_FORMAT_X86E8_84B) && meta->filter_passes > 0) {
        eof_symbol = ZX0_EOF_ALT_VALUE;
    } else if ((format == ULTRA_FORMAT_UNIV_93B || format == ULTRA_FORMAT_HYBRID_108B) && meta->filter_passes > 0) {
        eof_symbol = (meta->filter_stride == 0) ? ZX0_EOF_E8_VALUE : ZX0_EOF_ALT_VALUE;
    } else if ((format == ULTRA_FORMAT_ULTRA_128B || format == ULTRA_FORMAT_PRO_111B) && meta->filter_passes > 0) {
        if (meta->filter_stride == 0) {
            int is_two_opcodes = (meta->filter_passes >= 2 && meta->filter_strides[1] == 0);
            eof_symbol = is_two_opcodes ? ZX0_EOF_ULTRA_E8E9_VALUE : ZX0_EOF_ULTRA_E8_VALUE;
        } else {
            eof_symbol = ZX0_EOF_ALT_VALUE;
        }
    }
    bit_writer_put_elias(&bw, eof_symbol, 1);

    /* Emit post-filter control words */
    if (format == ULTRA_FORMAT_STRD_80B) {
        uint16_t count_word = (meta->filter_passes > 0) ? (uint16_t)meta->filter_counts[0] : 0;
        bit_writer_put_u16_le(&bw, count_word);
    } else if (format == ULTRA_FORMAT_COMPAT_82B ||
               format == ULTRA_FORMAT_X86E8_84B ||
               format == ULTRA_FORMAT_UNIV_93B) {
        if (meta->filter_passes > 0) {
            bit_writer_put_u16_le(&bw, (uint16_t)meta->filter_counts[0]);
        }
    } else if (format == ULTRA_FORMAT_STRD_82B) {
        /*
         * Decoder applies integration passes in stream order; encoder applied forward differences
         * in forward order, so emit counts in reverse pass order for exact inversion.
         */
        for (int p = meta->filter_passes - 1; p >= 0; p--) {
            bit_writer_put_u16_le(&bw, (uint16_t)meta->filter_counts[p]);
        }
        bit_writer_put_u16_le(&bw, 0);
    } else if (format == ULTRA_FORMAT_HYBRID_108B) {
        if (meta->filter_passes > 0) {
            for (int p = 0; p < meta->filter_passes; p++) {
                bit_writer_put_u16_le(&bw, (uint16_t)meta->filter_counts[p]);
                int st = meta->filter_strides[p];
                if (st > 0) {
                    bit_writer_put_byte(&bw, (uint8_t)((256 - st) & 0xFF));
                }
            }
            bit_writer_put_u16_le(&bw, 0);
        }
    } else if (format == ULTRA_FORMAT_ULTRA_128B || format == ULTRA_FORMAT_PRO_111B) {
        if (meta->filter_passes > 0) {
            for (int p = 0; p < meta->filter_passes; p++) {
                bit_writer_put_u16_le(&bw, (uint16_t)meta->filter_counts[p]);
                int st = meta->filter_strides[p];
                if (st > 0) {
                    uint16_t packed_w = (uint16_t)((meta->filter_word_modes[p] ? 0x8000u : 0x0000u) |
                                                   ((unsigned int)(256 - st) & 0xFFu));
                    bit_writer_put_u16_le(&bw, packed_w);
                }
            }
            bit_writer_put_u16_le(&bw, 0);
        }
    }

    *out_size = bw.size;
    return bw.data;
}

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t index;
    int bit_mask;
    int bit_value;
    int backtrack;
    int last_byte;
    int is_first_ultra_byte;
} BitReader;

static int bit_reader_get_byte(BitReader *br) {
    if (br->index >= br->size) {
        return -1;
    }
    br->last_byte = br->data[br->index++];
    return br->last_byte;
}

static int bit_reader_get_bit(BitReader *br) {
    if (br->backtrack) {
        br->backtrack = 0;
        return br->last_byte & 1;
    }
    br->bit_mask >>= 1;
    if (br->bit_mask == 0) {
        br->bit_mask = 128;
        br->bit_value = bit_reader_get_byte(br);
        if (br->bit_value < 0) {
            return -1;
        }
    }
    int bit = (br->bit_value & br->bit_mask) ? 1 : 0;
    if (br->is_first_ultra_byte && br->bit_mask == 2) {
        br->bit_mask = 1;
        br->is_first_ultra_byte = 0;
    }
    return bit;
}

static int bit_reader_get_elias(BitReader *br, int invert_data_bits) {
    int value = 1;
    for (;;) {
        int stop = bit_reader_get_bit(br);
        if (stop < 0) {
            return -1;
        }
        if (stop == 1) {
            break;
        }
        int data_bit = bit_reader_get_bit(br);
        if (data_bit < 0) {
            return -1;
        }
        value = (value << 1) | (data_bit ^ invert_data_bits);
    }
    return value;
}

uint8_t *ultra_decompress_buffer(
    const uint8_t *compressed_data,
    size_t compressed_size,
    size_t expected_output_size,
    UltraFormat format
) {
    if (compressed_data == NULL || compressed_size == 0 || expected_output_size == 0) {
        return NULL;
    }

    BitReader br;
    br.data = compressed_data;
    br.size = compressed_size;
    br.index = 0;
    br.bit_mask = 0;
    br.bit_value = 0;
    br.backtrack = 0;
    br.last_byte = 0;
    br.is_first_ultra_byte = 0;

    int offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
    int ultra_dh_sentinel = 1;
    if (format == ULTRA_FORMAT_ADPK_77B) {
        int hdr = bit_reader_get_byte(&br);
        if (hdr < 1 || hdr > MAX_ADPK_LSB_BITS) {
            return NULL;
        }
        offset_lsb_bits = hdr;
    } else if (format == ULTRA_FORMAT_ULTRA_128B) {
        int dl_init = bit_reader_get_byte(&br);
        int dh_init = bit_reader_get_byte(&br);
        if (dl_init < 0 || dh_init <= 0 || dh_init >= 128) {
            return NULL;
        }
        br.bit_value = dl_init;
        br.bit_mask = 256;
        br.is_first_ultra_byte = (dl_init != 0x80) ? 1 : 0;
        if (dl_init == 0x80) {
            br.bit_mask = 0;
        }
        ultra_dh_sentinel = dh_init;
    }

    uint8_t *out = (uint8_t *)malloc(expected_output_size);
    if (out == NULL) {
        return NULL;
    }
    size_t out_pos = 0;
    int last_offset = ZX0_INITIAL_OFFSET;
    int eof_msb_read = ZX0_EOF_MSB_VALUE;

    for (;;) {
        /* Copy literals */
        int lit_len = bit_reader_get_elias(&br, 0);
        if (lit_len <= 0 || out_pos + (size_t)lit_len > expected_output_size) {
            free(out);
            return NULL;
        }
        if (format == ULTRA_FORMAT_ULTRA_128B) {
            for (int i = 0; i < lit_len; i++) {
                int al = ultra_dh_sentinel;
                while (al < 256) {
                    int bit = bit_reader_get_bit(&br);
                    if (bit < 0) {
                        free(out);
                        return NULL;
                    }
                    al = (al << 1) | bit;
                }
                out[out_pos++] = (uint8_t)(al & 0xFF);
            }
        } else {
            for (int i = 0; i < lit_len; i++) {
                int b = bit_reader_get_byte(&br);
                if (b < 0) {
                    free(out);
                    return NULL;
                }
                out[out_pos++] = (uint8_t)b;
            }
        }

        int ctrl = bit_reader_get_bit(&br);
        if (ctrl < 0) {
            free(out);
            return NULL;
        }
        if (ctrl == 0) {
            /* Copy from last offset */
            int rep_len = bit_reader_get_elias(&br, 0);
            if (rep_len <= 0 || (size_t)last_offset > out_pos || out_pos + (size_t)rep_len > expected_output_size) {
                free(out);
                return NULL;
            }
            for (int i = 0; i < rep_len; i++) {
                out[out_pos] = out[out_pos - (size_t)last_offset];
                out_pos++;
            }
            ctrl = bit_reader_get_bit(&br);
            if (ctrl < 0) {
                free(out);
                return NULL;
            }
            if (ctrl == 0) {
                continue;
            }
        }

        /* Copy from new offset (repeating while trailing ctrl bit == 1) */
        for (;;) {
            int msb = bit_reader_get_elias(&br, 1);
            if (msb < 0) {
                free(out);
                return NULL;
            }
            if (msb >= ZX0_EOF_MSB_VALUE) {
                eof_msb_read = msb;
                goto reached_eof;
            }

            if (format != ULTRA_FORMAT_ADPK_77B) {
                int lsb_byte = bit_reader_get_byte(&br);
                if (lsb_byte < 0) {
                    free(out);
                    return NULL;
                }
                last_offset = msb * 128 - (lsb_byte >> 1);
                br.backtrack = 1;
            } else {
                uint16_t bx_u16 = (uint16_t)(0x10000u - (unsigned int)msb);
                for (int b = 0; b < offset_lsb_bits; b++) {
                    int bit = bit_reader_get_bit(&br);
                    if (bit < 0) {
                        free(out);
                        return NULL;
                    }
                    bx_u16 = (uint16_t)((bx_u16 << 1) | (uint16_t)bit);
                }
                last_offset = (bx_u16 == 0) ? 65536 : (int)(0x10000u - (unsigned int)bx_u16);
            }

            int match_len = bit_reader_get_elias(&br, 0) + 1;
            if (match_len < 2 || last_offset <= 0 || (size_t)last_offset > out_pos ||
                out_pos + (size_t)match_len > expected_output_size) {
                free(out);
                return NULL;
            }
            for (int i = 0; i < match_len; i++) {
                out[out_pos] = out[out_pos - (size_t)last_offset];
                out_pos++;
            }

            ctrl = bit_reader_get_bit(&br);
            if (ctrl < 0) {
                free(out);
                return NULL;
            }
            if (ctrl == 0) {
                break;
            }
        }
    }

reached_eof:
    if (out_pos != expected_output_size) {
        free(out);
        return NULL;
    }

    if (format == ULTRA_FORMAT_STRD_80B ||
        (format == ULTRA_FORMAT_COMPAT_82B && eof_msb_read > ZX0_EOF_MSB_VALUE) ||
        (format == ULTRA_FORMAT_UNIV_93B && eof_msb_read == ZX0_EOF_ALT_VALUE)) {
        if (br.index + FILTER_WORD_BYTES > br.size) {
            free(out);
            return NULL;
        }
        uint16_t count = (uint16_t)(br.data[br.index] | (br.data[br.index + 1] << 8));
        br.index += FILTER_WORD_BYTES;
        if (count > 0) {
            if ((size_t)count + (size_t)last_offset > out_pos) {
                free(out);
                return NULL;
            }
            size_t di = out_pos - (size_t)count;
            for (uint16_t c = 0; c < count; c++, di++) {
                out[di] = (uint8_t)(out[di] + out[di - (size_t)last_offset]);
            }
        }
    } else if ((format == ULTRA_FORMAT_X86E8_84B && eof_msb_read > ZX0_EOF_MSB_VALUE) ||
               (format == ULTRA_FORMAT_UNIV_93B && eof_msb_read == ZX0_EOF_E8_VALUE)) {
        if (br.index + FILTER_WORD_BYTES > br.size) {
            free(out);
            return NULL;
        }
        uint16_t count = (uint16_t)(br.data[br.index] | (br.data[br.index + 1] << 8));
        br.index += FILTER_WORD_BYTES;
        if (count == 0 || (size_t)count > out_pos) {
            free(out);
            return NULL;
        }
        size_t di = out_pos - (size_t)count;
        for (uint32_t rem = count; rem > 0; rem--, di++) {
            if (out[di] == X86_OPCODE_CALL_REL16) {
                if (di + 2 >= out_pos) {
                    free(out);
                    return NULL;
                }
                uint16_t rel = (uint16_t)(out[di + 1] | (out[di + 2] << 8));
                rel = (uint16_t)(rel + (uint16_t)rem);
                out[di + 1] = (uint8_t)(rel & 0xFF);
                out[di + 2] = (uint8_t)((rel >> 8) & 0xFF);
            }
        }
    } else if (format == ULTRA_FORMAT_STRD_82B) {
        for (;;) {
            if (br.index + FILTER_WORD_BYTES > br.size) {
                free(out);
                return NULL;
            }
            uint16_t count = (uint16_t)(br.data[br.index] | (br.data[br.index + 1] << 8));
            br.index += FILTER_WORD_BYTES;
            if (count == 0) {
                break;
            }
            if ((size_t)count + (size_t)last_offset > out_pos) {
                free(out);
                return NULL;
            }
            size_t di = out_pos - (size_t)count;
            for (uint16_t c = 0; c < count; c++, di++) {
                out[di] = (uint8_t)(out[di] + out[di - (size_t)last_offset]);
            }
        }
    } else if ((format == ULTRA_FORMAT_HYBRID_108B ||
                format == ULTRA_FORMAT_PRO_111B ||
                format == ULTRA_FORMAT_ULTRA_128B) &&
               eof_msb_read > ZX0_EOF_MSB_VALUE) {
        int e8_stage = 0;
        if (format == ULTRA_FORMAT_HYBRID_108B) {
            e8_stage = (eof_msb_read == ZX0_EOF_E8_VALUE) ? 2 : 0;
        } else {
            if (eof_msb_read == ZX0_EOF_ULTRA_E8E9_VALUE) {
                e8_stage = 2;
            } else if (eof_msb_read == ZX0_EOF_ULTRA_E8_VALUE) {
                e8_stage = 1;
            }
        }
        int e8_total = e8_stage;
        for (;;) {
            if (br.index + FILTER_WORD_BYTES > br.size) {
                free(out);
                return NULL;
            }
            uint16_t count = (uint16_t)(br.data[br.index] | (br.data[br.index + 1] << 8));
            br.index += FILTER_WORD_BYTES;
            if (count == 0) {
                break;
            }
            if (e8_stage > 0) {
                uint8_t target_op = (e8_stage == e8_total) ? X86_OPCODE_CALL_REL16 : X86_OPCODE_JMP_REL16;
                e8_stage--;
                if ((size_t)count > out_pos) {
                    free(out);
                    return NULL;
                }
                size_t di = out_pos - (size_t)count;
                for (uint32_t rem = count; rem > 0; rem--, di++) {
                    if (out[di] == target_op) {
                        if (di + 2 >= out_pos) {
                            free(out);
                            return NULL;
                        }
                        uint16_t rel = (uint16_t)(out[di + 1] | (out[di + 2] << 8));
                        rel = (uint16_t)(rel + (uint16_t)rem);
                        out[di + 1] = (uint8_t)(rel & 0xFF);
                        out[di + 2] = (uint8_t)((rel >> 8) & 0xFF);
                    }
                }
            } else if (format == ULTRA_FORMAT_HYBRID_108B) {
                int neg_b = bit_reader_get_byte(&br);
                if (neg_b < 0) {
                    free(out);
                    return NULL;
                }
                int stride = 256 - neg_b;
                if ((size_t)count + (size_t)stride > out_pos) {
                    free(out);
                    return NULL;
                }
                size_t di = out_pos - (size_t)count;
                for (uint16_t c = 0; c < count; c++, di++) {
                    out[di] = (uint8_t)(out[di] + out[di - (size_t)stride]);
                }
            } else {
                if (br.index + 2 > br.size) {
                    free(out);
                    return NULL;
                }
                uint16_t packed_w = (uint16_t)(br.data[br.index] | (br.data[br.index + 1] << 8));
                br.index += 2;
                int stride = 256 - (int)(packed_w & 0xFFu);
                if ((size_t)count + (size_t)stride > out_pos) {
                    free(out);
                    return NULL;
                }
                size_t di = out_pos - (size_t)count;
                if (packed_w & 0x8000u) {
                    if (count & 1) {
                        free(out);
                        return NULL;
                    }
                    for (uint16_t c = 0; c < count; c += 2, di += 2) {
                        uint16_t w_cur = (uint16_t)(out[di] | (out[di + 1] << 8));
                        uint16_t w_prv = (uint16_t)(out[di - (size_t)stride] | (out[di - (size_t)stride + 1] << 8));
                        uint16_t sum = (uint16_t)(w_cur + w_prv);
                        out[di] = (uint8_t)(sum & 0xFF);
                        out[di + 1] = (uint8_t)((sum >> 8) & 0xFF);
                    }
                } else {
                    for (uint16_t c = 0; c < count; c++, di++) {
                        out[di] = (uint8_t)(out[di] + out[di - (size_t)stride]);
                    }
                }
            }
        }
    }

    if (br.index != br.size) {
        free(out);
        return NULL;
    }
    return out;
}

static void apply_forward_stride_difference(
    const uint8_t *src,
    uint8_t *dst,
    int input_size,
    int stride,
    int tail_count
) {
    memcpy(dst, src, (size_t)input_size);
    int start_idx = input_size - tail_count;
    for (int i = input_size - 1; i >= start_idx; i--) {
        dst[i] = (uint8_t)(dst[i] - dst[i - stride]);
    }
}

static void apply_forward_word16_stride_difference(
    const uint8_t *src,
    uint8_t *dst,
    int input_size,
    int stride,
    int tail_count
) {
    memcpy(dst, src, (size_t)input_size);
    int even_tail = tail_count & ~1;
    int start_idx = input_size - even_tail;
    for (int i = input_size - 2; i >= start_idx; i -= 2) {
        uint16_t w_cur = (uint16_t)(dst[i] | (dst[i + 1] << 8));
        uint16_t w_prv = (uint16_t)(dst[i - stride] | (dst[i - stride + 1] << 8));
        uint16_t diff = (uint16_t)(w_cur - w_prv);
        dst[i] = (uint8_t)(diff & 0xFF);
        dst[i + 1] = (uint8_t)((diff >> 8) & 0xFF);
    }
}

/*
 * O(N) Kadane suffix-gain scan to locate optimal piecewise tail boundaries [input_size - count .. input_size)
 * where Stride-Delta increases local byte predictability without corrupting an earlier code/header prefix.
 * Detects cross-byte constant deltas (d_cur == d_prv), same-channel linear ramps (d_cur == d_stride_prev),
 * and smooth 2nd-order derivatives (dd in {-1, 0, +1}), plus an unbiased Kadane variant for hybrid ROMs.
 */
static int find_optimal_stride_tail_count(
    const uint8_t *src,
    int input_size,
    int stride,
    int *best_gain_out,
    int *unbiased_tail_out
) {
    int max_tail = input_size - stride;
    if (max_tail > 65535) {
        max_tail = 65535;
    }
    if (max_tail <= stride) {
        *best_gain_out = 0;
        if (unbiased_tail_out != NULL) {
            *unbiased_tail_out = 0;
        }
        return 0;
    }

    int min_idx = input_size - max_tail;
    int suffix_gain = 0;
    int best_gain = 0;
    int best_tail = 0;
    int unbiased_gain = 0;
    int best_unbiased_gain = 0;
    int best_unbiased_tail = 0;

    for (int i = input_size - 1; i >= min_idx; i--) {
        int raw_hit = (src[i] == 0 || src[i] == src[i - 1] || src[i] == src[i - stride]) ? 1 : 0;
        uint8_t d_cur = (uint8_t)(src[i] - src[i - stride]);
        int dif_hit = (d_cur == 0) ? 1 : 0;
        if (!dif_hit && i > stride) {
            uint8_t d_prv = (uint8_t)(src[i - 1] - src[i - 1 - stride]);
            if (d_cur == d_prv) {
                dif_hit = 1;
            }
        }
        if (!dif_hit && i >= 2 * stride) {
            uint8_t d_str = (uint8_t)(src[i - stride] - src[i - 2 * stride]);
            uint8_t dd = (uint8_t)(d_cur - d_str);
            if (dd == 0 || dd == 1 || dd == 255) {
                dif_hit = 1;
            }
        }
        int delta_score = 2 * (dif_hit - raw_hit) - ((dif_hit == 0 && raw_hit == 0) ? 1 : 0);
        suffix_gain += delta_score;
        if (suffix_gain > best_gain) {
            best_gain = suffix_gain;
            best_tail = input_size - i;
        }
        unbiased_gain += 2 * (dif_hit - raw_hit);
        if (unbiased_gain > best_unbiased_gain) {
            best_unbiased_gain = unbiased_gain;
            best_unbiased_tail = input_size - i;
        }
    }

    *best_gain_out = best_gain;
    if (unbiased_tail_out != NULL) {
        *unbiased_tail_out = best_unbiased_tail;
    }
    return best_tail;
}

/*
 * Exact backward inverse of unzx0_x86e8_84b.asm / unzx0_univ_93b.asm / unzx0_hybrid_108b.asm
 * forward scasb/add [di],cx loop for 0xE8 (CALL rel16) or 0xE9 (JMP rel16).
 */
static int apply_forward_x86_opcode_filter(
    const uint8_t *src,
    uint8_t *dst,
    int input_size,
    int tail_count,
    uint8_t target_op
) {
    if (input_size < 3 || tail_count < 3 || tail_count > input_size || tail_count > 65535) {
        return 0;
    }
    if (src[input_size - 2] == target_op || src[input_size - 1] == target_op) {
        return 0;
    }

    memcpy(dst, src, (size_t)input_size);
    int start_idx = input_size - tail_count;
    int hits = 0;

    for (int i = input_size - 3; i >= start_idx; i--) {
        if (dst[i] == target_op) {
            uint16_t rem = (uint16_t)(tail_count - (i - start_idx));
            uint16_t rel = (uint16_t)(dst[i + 1] | (dst[i + 2] << 8));
            rel = (uint16_t)(rel - rem);
            dst[i + 1] = (uint8_t)(rel & 0xFF);
            dst[i + 2] = (uint8_t)((rel >> 8) & 0xFF);
            hits++;
        }
    }
    return hits;
}

uint8_t *ultra_compress_buffer(
    const uint8_t *input_data,
    int input_size,
    const uint8_t *wildcard_mask,
    UltraFormat format,
    int max_offset_limit,
    size_t *compressed_size_out,
    UltraMetadata *metadata_out,
    uint8_t *resolved_input_out
) {
    assert(input_data != NULL && input_size > 0 && compressed_size_out != NULL);
    int effective_limit = (max_offset_limit > 0 && max_offset_limit <= ZX0_MAX_OFFSET)
                        ? max_offset_limit
                        : ZX0_MAX_OFFSET;

    uint8_t *working_input = (uint8_t *)malloc((size_t)input_size);
    assert(working_input != NULL);
    memcpy(working_input, input_data, (size_t)input_size);

    /* Two-pass wildcard materialization when flexible gap bytes are present */
    if (wildcard_mask != NULL) {
        int has_any_wildcard = 0;
        for (int i = 0; i < input_size; i++) {
            if (wildcard_mask[i] != 0) {
                has_any_wildcard = 1;
                break;
            }
        }
        if (has_any_wildcard) {
            int init_lsb = (format == ULTRA_FORMAT_ADPK_77B) ? 5 : ZX0_DEFAULT_LSB_BITS;
            ParsePlan wp = {NULL, 0, 0};
            if (solve_exact_parse(working_input, input_size, wildcard_mask, init_lsb, effective_limit, 0, &wp)) {
                materialize_wildcards_from_plan(working_input, wildcard_mask, &wp);
                free(wp.tokens);
            }
        }
    }

    if (resolved_input_out != NULL) {
        memcpy(resolved_input_out, working_input, (size_t)input_size);
    }

    uint8_t *best_stream = NULL;
    size_t best_size = (size_t)-1;
    UltraMetadata best_meta;
    memset(&best_meta, 0, sizeof(best_meta));
    best_meta.format = format;
    best_meta.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;

    if (format == ULTRA_FORMAT_ZX0_V2) {
        ParsePlan plan = {NULL, 0, 0};
        int ok = solve_exact_parse(working_input, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, &plan);
        assert(ok);
        best_meta.compressed_bits = plan.payload_bits;
        best_stream = encode_plan_to_stream(working_input, &plan, format, ZX0_DEFAULT_LSB_BITS, &best_meta, &best_size);
        free(plan.tokens);
    } else if (format == ULTRA_FORMAT_ADPK_77B) {
        #pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
        for (int k = 1; k <= MAX_ADPK_LSB_BITS; k++) {
            if (k > 2 && (1 << (k - 1)) >= input_size * 2) {
                continue;
            }
            ParsePlan plan = {NULL, 0, 0};
            if (solve_exact_parse(working_input, input_size, NULL, k, effective_limit, 0, &plan)) {
                UltraMetadata cand_meta;
                memset(&cand_meta, 0, sizeof(cand_meta));
                cand_meta.format = format;
                cand_meta.offset_lsb_bits = k;
                cand_meta.compressed_bits = plan.payload_bits;
                size_t cand_size = 0;
                uint8_t *cand_stream = encode_plan_to_stream(working_input, &plan, format, k, &cand_meta, &cand_size);
                free(plan.tokens);

                #pragma omp critical
                {
                    if (cand_size < best_size ||
                        (cand_size == best_size && cand_meta.compressed_bits < best_meta.compressed_bits)) {
                        free(best_stream);
                        best_stream = cand_stream;
                        best_size = cand_size;
                        best_meta = cand_meta;
                    } else {
                        free(cand_stream);
                    }
                }
            }
        }
    } else {
        /*
         * Unified Post-Filter Search for:
         * - ULTRA_FORMAT_X86E8_84B  (Zero-overhead ZX0 + E8 Call-Target filter)
         * - ULTRA_FORMAT_STRD_80B   (16-bit Stride-Delta filter)
         * - ULTRA_FORMAT_COMPAT_82B (Zero-overhead ZX0 + 16-bit Stride-Delta filter)
         * - ULTRA_FORMAT_STRD_82B   (Multi-pass Order-1..5 Stride-Delta filter)
         * - ULTRA_FORMAT_UNIV_93B   (Universal Zero-overhead ZX0 + 16-bit Stride-Delta + E8 Call-Target)
         * - ULTRA_FORMAT_HYBRID_108B (Multi-Stage 0B-Ovh ZX0 + E8/E9 + Free-BX Multi-Stride 2D filter)
         * - ULTRA_FORMAT_PRO_111B   (Zero-overhead ZX0 + E8/E9 + 16-bit Word/Byte-ADD 2D filter)
         * - ULTRA_FORMAT_ULTRA_128B (Flagship 128B: Packed Header + 4..8-bit Literals + E8/E9 + 16-bit Word/Byte-ADD 2D filter)
         */
        int is_word_capable = (format == ULTRA_FORMAT_ULTRA_128B || format == ULTRA_FORMAT_PRO_111B);
        int is_multi_stage  = (format == ULTRA_FORMAT_HYBRID_108B || is_word_capable);
        int base_req_off = (format == ULTRA_FORMAT_ULTRA_128B) ? -1 : 0;
        int base_payload_bits = 0;
        {
            ParsePlan base_plan = {NULL, 0, 0};
            int ok = solve_exact_parse(working_input, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, base_req_off, &base_plan);
            assert(ok);
            base_payload_bits = base_plan.payload_bits;
            best_meta.compressed_bits = base_plan.payload_bits;
            best_meta.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                   ? detect_buffer_literal_bits(working_input, input_size, NULL) : 8;
            best_stream = encode_plan_to_stream(working_input, &base_plan, format, ZX0_DEFAULT_LSB_BITS, &best_meta, &best_size);
            free(base_plan.tokens);
        }

        uint8_t *best_e8_buf = NULL;
        int best_e8_tail_chosen = 0;

        /* A. Evaluate position-independent x86 E8 Call-Target transform (X86E8_84B, UNIV_93B, HYBRID_108B, PRO_111B, ULTRA_128B) */
        if ((format == ULTRA_FORMAT_X86E8_84B || format == ULTRA_FORMAT_UNIV_93B || is_multi_stage) &&
            input_size >= 3 &&
            working_input[input_size - 2] != X86_OPCODE_CALL_REL16 &&
            working_input[input_size - 1] != X86_OPCODE_CALL_REL16) {
            int full_tail = input_size <= 65535 ? input_size : 65535;
            int cand_tails[6];
            int num_tails = 0;
            cand_tails[num_tails++] = full_tail;

            uint16_t *target_freq = (uint16_t *)calloc(65536, sizeof(uint16_t));
            assert(target_freq != NULL);
            int min_scan_idx = input_size - full_tail;
            for (int i = min_scan_idx; i <= input_size - 3; i++) {
                if (working_input[i] == X86_OPCODE_CALL_REL16) {
                    uint16_t rem = (uint16_t)(full_tail - (i - min_scan_idx));
                    uint16_t tword = (uint16_t)((working_input[i + 1] | (working_input[i + 2] << 8)) - rem);
                    if (target_freq[tword] < 65535) {
                        target_freq[tword]++;
                    }
                }
            }
            int seen_e8 = 0;
            for (int i = min_scan_idx; i <= input_size - 3 && seen_e8 < 4; i++) {
                if (working_input[i] == X86_OPCODE_CALL_REL16) {
                    uint16_t rem = (uint16_t)(full_tail - (i - min_scan_idx));
                    uint16_t tword = (uint16_t)((working_input[i + 1] | (working_input[i + 2] << 8)) - rem);
                    if (target_freq[tword] >= 2) {
                        int repeated_tail = input_size - i;
                        if (repeated_tail >= 3 && repeated_tail < full_tail - 8 && num_tails < 5) {
                            cand_tails[num_tails++] = repeated_tail;
                        }
                        seen_e8++;
                    }
                }
            }

            /* Kadane suffix-gain scan for E8 call targets in hybrid binaries */
            int suf_gain = 0;
            int best_e8_gain = 0;
            int best_e8_tail = 0;
            for (int i = input_size - 3; i >= min_scan_idx; i--) {
                if (working_input[i] == X86_OPCODE_CALL_REL16) {
                    uint16_t rem = (uint16_t)(full_tail - (i - min_scan_idx));
                    uint16_t tword = (uint16_t)((working_input[i + 1] | (working_input[i + 2] << 8)) - rem);
                    suf_gain += (target_freq[tword] >= 2) ? 6 : -4;
                    if (suf_gain > best_e8_gain) {
                        best_e8_gain = suf_gain;
                        best_e8_tail = input_size - i;
                    }
                }
            }
            if (best_e8_tail >= 3 && best_e8_tail < full_tail - 16) {
                int dup = 0;
                for (int k = 0; k < num_tails; k++) {
                    if (abs(cand_tails[k] - best_e8_tail) <= 16) {
                        dup = 1;
                    }
                }
                if (!dup && num_tails < 6) {
                    cand_tails[num_tails++] = best_e8_tail;
                }
            }
            free(target_freq);

            size_t best_e8_only_size = best_size;
            int nested_parallel = omp_in_parallel();
            #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
            for (int ti = 0; ti < num_tails; ti++) {
                int tail_count = cand_tails[ti];
                uint8_t *e8_buf = (uint8_t *)malloc((size_t)input_size);
                assert(e8_buf != NULL);
                int hits = apply_forward_x86_opcode_filter(working_input, e8_buf, input_size, tail_count, X86_OPCODE_CALL_REL16);
                if (hits >= 2) {
                    ParsePlan plan_e8 = {NULL, 0, 0};
                    if (solve_exact_parse(e8_buf, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, base_req_off, &plan_e8)) {
                        UltraMetadata meta_e8;
                        memset(&meta_e8, 0, sizeof(meta_e8));
                        meta_e8.format = format;
                        meta_e8.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                        meta_e8.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                             ? detect_buffer_literal_bits(e8_buf, input_size, NULL) : 8;
                        meta_e8.filter_passes = 1;
                        meta_e8.filter_stride = 0;
                        meta_e8.filter_strides[0] = 0;
                        meta_e8.filter_counts[0] = tail_count;
                        meta_e8.compressed_bits = plan_e8.payload_bits;

                        size_t size_e8 = 0;
                        uint8_t *stream_e8 = encode_plan_to_stream(
                            e8_buf, &plan_e8, format, ZX0_DEFAULT_LSB_BITS, &meta_e8, &size_e8
                        );
                        free(plan_e8.tokens);

                        #pragma omp critical
                        {
                            if (size_e8 < best_size ||
                                (size_e8 == best_size && meta_e8.compressed_bits < best_meta.compressed_bits)) {
                                free(best_stream);
                                best_stream = stream_e8;
                                best_size = size_e8;
                                best_meta = meta_e8;
                            } else {
                                free(stream_e8);
                            }
                            if (size_e8 < best_e8_only_size) {
                                best_e8_only_size = size_e8;
                                free(best_e8_buf);
                                best_e8_buf = (uint8_t *)malloc((size_t)input_size);
                                memcpy(best_e8_buf, e8_buf, (size_t)input_size);
                                best_e8_tail_chosen = tail_count;
                            }
                        }
                    }

                    /* For HYBRID_108B, PRO_111B, and ULTRA_128B: also evaluate chaining E8 (CALL rel16) + E9 (JMP rel16) */
                    if (is_multi_stage &&
                        e8_buf[input_size - 2] != X86_OPCODE_JMP_REL16 &&
                        e8_buf[input_size - 1] != X86_OPCODE_JMP_REL16) {
                        uint8_t *e8e9_buf = (uint8_t *)malloc((size_t)input_size);
                        assert(e8e9_buf != NULL);
                        int hits_e9 = apply_forward_x86_opcode_filter(e8_buf, e8e9_buf, input_size, tail_count, X86_OPCODE_JMP_REL16);
                        if (hits_e9 >= 2) {
                            ParsePlan plan_e89 = {NULL, 0, 0};
                            if (solve_exact_parse(e8e9_buf, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, base_req_off, &plan_e89)) {
                                UltraMetadata meta_e89;
                                memset(&meta_e89, 0, sizeof(meta_e89));
                                meta_e89.format = format;
                                meta_e89.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                meta_e89.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                      ? detect_buffer_literal_bits(e8e9_buf, input_size, NULL) : 8;
                                meta_e89.filter_passes = 2;
                                meta_e89.filter_stride = 0;
                                meta_e89.filter_strides[0] = 0;
                                meta_e89.filter_strides[1] = 0;
                                meta_e89.filter_counts[0] = tail_count;
                                meta_e89.filter_counts[1] = tail_count;
                                meta_e89.compressed_bits = plan_e89.payload_bits;

                                size_t size_e89 = 0;
                                uint8_t *stream_e89 = encode_plan_to_stream(
                                    e8e9_buf, &plan_e89, format, ZX0_DEFAULT_LSB_BITS, &meta_e89, &size_e89
                                );
                                free(plan_e89.tokens);

                                #pragma omp critical
                                {
                                    if (size_e89 < best_size ||
                                        (size_e89 == best_size && meta_e89.compressed_bits < best_meta.compressed_bits)) {
                                        free(best_stream);
                                        best_stream = stream_e89;
                                        best_size = size_e89;
                                        best_meta = meta_e89;
                                    } else {
                                        free(stream_e89);
                                    }
                                }
                            }
                        }
                        free(e8e9_buf);
                    }
                }
                free(e8_buf);
            }
        }

        /* B. Multi-core search over candidate strides (STRD_80B, COMPAT_82B, STRD_82B, UNIV_93B, HYBRID_108B, PRO_111B, ULTRA_128B) */
        if (format != ULTRA_FORMAT_X86E8_84B) {
            int candidate_strides[112];
            int num_strides = 0;
            for (int s = 1; s <= 64 && s < input_size; s++) {
                candidate_strides[num_strides++] = s;
            }
            static const int extra_strides[] = {
                68, 72, 75, 80, 84, 90, 96, 100, 108, 112, 120, 128,
                144, 150, 160, 180, 192, 200, 216, 224, 240, 250, 256, 288, 300, 320
            };
            for (size_t idx = 0; idx < sizeof(extra_strides) / sizeof(extra_strides[0]); idx++) {
                int es = extra_strides[idx];
                if (es < input_size) {
                    if (is_multi_stage && es > 256) {
                        continue;
                    }
                    candidate_strides[num_strides++] = es;
                }
            }

            int max_passes = (format == ULTRA_FORMAT_STRD_82B || is_multi_stage) ? 5 : 1;
            int nested_parallel = omp_in_parallel();

            #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
            for (int si = 0; si < num_strides; si++) {
                int stride = candidate_strides[si];
                int req_final_off = (format == ULTRA_FORMAT_ULTRA_128B) ? -1
                                  : is_multi_stage ? 0 : stride;
                int max_full_count = input_size - stride;
                int full_count = max_full_count > 65535 ? 65535 : max_full_count;
                if (full_count <= 0) {
                    continue;
                }

                int kadane_gain = 0;
                int unbiased_tail = 0;
                int kadane_tail = find_optimal_stride_tail_count(
                    working_input, input_size, stride, &kadane_gain, &unbiased_tail
                );

                int is_structural_stride = (stride == 1 || stride == 2 || stride == 3 || stride == 4 ||
                                            stride == 6 || stride == 8 || stride == 9 || stride == 10 ||
                                            stride == 12 || stride == 15 || stride == 16 || stride == 18 ||
                                            stride == 20 || stride == 24 || stride == 30 || stride == 32 ||
                                            stride == 36 || stride == 40 || stride == 48 || stride == 60 ||
                                            stride == 64 || stride == 72 || stride == 80 || stride == 96 ||
                                            stride == 108 || stride == 120 || stride == 128 || stride == 144 ||
                                            stride == 160 || stride == 192 || stride == 240 || stride == 256 ||
                                            stride == 320 || (input_size % stride == 0));
                int eq_raw = 0;
                int eq_dif = 0;
                for (int i = stride + 1; i < input_size; i++) {
                    if (working_input[i] == working_input[i - 1] ||
                        working_input[i] == working_input[i - stride]) {
                        eq_raw++;
                    }
                    uint8_t d_cur = (uint8_t)(working_input[i] - working_input[i - stride]);
                    uint8_t d_prv = (uint8_t)(working_input[i - 1] - working_input[i - 1 - stride]);
                    int hit = (d_cur == 0 || d_cur == d_prv);
                    if (!hit && i >= 2 * stride) {
                        uint8_t d_str = (uint8_t)(working_input[i - stride] - working_input[i - 2 * stride]);
                        uint8_t dd = (uint8_t)(d_cur - d_str);
                        hit = (dd == 0 || dd == 1 || dd == 255);
                    }
                    if (hit) {
                        eq_dif++;
                    }
                }
                if (stride > 8 && kadane_gain < 16 && (!is_structural_stride || eq_dif * 4 < eq_raw * 3)) {
                    continue;
                }

                uint8_t *cur_buf = (uint8_t *)malloc((size_t)input_size);
                uint8_t *nxt_buf = (uint8_t *)malloc((size_t)input_size);
                assert(cur_buf != NULL && nxt_buf != NULL);
                memcpy(cur_buf, working_input, (size_t)input_size);

                int prev_pass_bits = COST_INFINITY;

                for (int pass = 1; pass <= max_passes; pass++) {
                    if (input_size <= pass * stride) {
                        break;
                    }
                    apply_forward_stride_difference(cur_buf, nxt_buf, input_size, stride, full_count);
                    memcpy(cur_buf, nxt_buf, (size_t)input_size);

                    ParsePlan plan_p = {NULL, 0, 0};
                    if (solve_exact_parse(cur_buf, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, req_final_off, &plan_p)) {
                        int pass_bits = plan_p.payload_bits;
                        UltraMetadata meta_p;
                        memset(&meta_p, 0, sizeof(meta_p));
                        meta_p.format = format;
                        meta_p.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                        meta_p.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                            ? detect_buffer_literal_bits(cur_buf, input_size, NULL) : 8;
                        meta_p.filter_passes = pass;
                        meta_p.filter_stride = stride;
                        for (int p_idx = 0; p_idx < pass; p_idx++) {
                            meta_p.filter_strides[p_idx] = stride;
                            meta_p.filter_counts[p_idx] = full_count;
                        }
                        meta_p.compressed_bits = pass_bits;

                        size_t size_p = 0;
                        uint8_t *stream_p = encode_plan_to_stream(
                            cur_buf, &plan_p, format, ZX0_DEFAULT_LSB_BITS, &meta_p, &size_p
                        );
                        free(plan_p.tokens);

                        #pragma omp critical
                        {
                            if (size_p < best_size ||
                                (size_p == best_size && meta_p.compressed_bits < best_meta.compressed_bits)) {
                                free(best_stream);
                                best_stream = stream_p;
                                best_size = size_p;
                                best_meta = meta_p;
                            } else {
                                free(stream_p);
                            }
                        }

                        if (pass_bits > base_payload_bits + 2048 || pass_bits > prev_pass_bits + 512) {
                            break;
                        }
                        prev_pass_bits = pass_bits;
                    }
                }

                /* Harmonic multi-pass counts (count_p = input_size - pass * stride) avoiding boundary artifacts */
                if (max_passes > 1 && stride <= 16 && prev_pass_bits < base_payload_bits + 1024) {
                    memcpy(cur_buf, working_input, (size_t)input_size);
                    int harm_counts[ULTRA_MAX_FILTER_PASSES] = {0};
                    int prev_harm_bits = COST_INFINITY;
                    for (int pass = 1; pass <= max_passes; pass++) {
                        int hc = input_size - pass * stride;
                        if (hc <= 0 || hc > 65535) {
                            break;
                        }
                        apply_forward_stride_difference(cur_buf, nxt_buf, input_size, stride, hc);
                        memcpy(cur_buf, nxt_buf, (size_t)input_size);
                        harm_counts[pass - 1] = hc;
                        if (pass >= 2) {
                            ParsePlan plan_hm = {NULL, 0, 0};
                            if (solve_exact_parse(cur_buf, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, req_final_off, &plan_hm)) {
                                int hm_bits = plan_hm.payload_bits;
                                UltraMetadata meta_hm;
                                memset(&meta_hm, 0, sizeof(meta_hm));
                                meta_hm.format = format;
                                meta_hm.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                meta_hm.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                     ? detect_buffer_literal_bits(cur_buf, input_size, NULL) : 8;
                                meta_hm.filter_passes = pass;
                                meta_hm.filter_stride = stride;
                                for (int p_idx = 0; p_idx < pass; p_idx++) {
                                    meta_hm.filter_strides[p_idx] = stride;
                                    meta_hm.filter_counts[p_idx] = harm_counts[pass - 1 - p_idx];
                                }
                                if (!is_multi_stage) {
                                    memcpy(meta_hm.filter_counts, harm_counts, sizeof(harm_counts));
                                }
                                meta_hm.compressed_bits = hm_bits;

                                size_t size_hm = 0;
                                uint8_t *stream_hm = encode_plan_to_stream(
                                    cur_buf, &plan_hm, format, ZX0_DEFAULT_LSB_BITS, &meta_hm, &size_hm
                                );
                                free(plan_hm.tokens);

                                #pragma omp critical
                                {
                                    if (size_hm < best_size ||
                                        (size_hm == best_size && meta_hm.compressed_bits < best_meta.compressed_bits)) {
                                        free(best_stream);
                                        best_stream = stream_hm;
                                        best_size = size_hm;
                                        best_meta = meta_hm;
                                    } else {
                                        free(stream_hm);
                                    }
                                }
                                if (hm_bits > prev_harm_bits + 512) {
                                    break;
                                }
                                prev_harm_bits = hm_bits;
                            }
                        }
                    }
                }

                /* Evaluate piecewise half-tail, Kadane-optimal tail, unbiased Kadane tail, and local refinements */
                int cand_piecewise[10];
                int num_piecewise = 0;
                int half_count = input_size / 2;
                if (!nested_parallel && input_size >= 128 && half_count > stride && half_count < full_count) {
                    cand_piecewise[num_piecewise++] = half_count;
                }
                int k_cands[8];
                int nk = 0;
                if (kadane_tail > stride && kadane_tail < full_count) {
                    k_cands[nk++] = kadane_tail;
                    if (kadane_tail - 3 > stride) k_cands[nk++] = kadane_tail - 3;
                    if (kadane_tail - 1 > stride) k_cands[nk++] = kadane_tail - 1;
                    if (stride > 1) {
                        int aligned = ((kadane_tail + stride / 2) / stride) * stride;
                        if (aligned > stride && aligned < full_count) {
                            k_cands[nk++] = aligned;
                        }
                    }
                }
                if (unbiased_tail > stride && unbiased_tail < full_count) {
                    if (nk < 8) k_cands[nk++] = unbiased_tail;
                    if (stride > 1 && nk < 8) {
                        int aligned_u = ((unbiased_tail + stride / 2) / stride) * stride;
                        if (aligned_u > stride && aligned_u < full_count) {
                            k_cands[nk++] = aligned_u;
                        }
                    }
                }
                for (int ki = 0; ki < nk; ki++) {
                    int tc = k_cands[ki];
                    int dup = 0;
                    for (int m = 0; m < num_piecewise; m++) {
                        if (cand_piecewise[m] == tc) {
                            dup = 1;
                            break;
                        }
                    }
                    if (!dup && num_piecewise < 10) {
                        cand_piecewise[num_piecewise++] = tc;
                    }
                }

                for (int pi = 0; pi < num_piecewise; pi++) {
                    int tail_c = cand_piecewise[pi];
                    apply_forward_stride_difference(working_input, nxt_buf, input_size, stride, tail_c);
                    ParsePlan plan_h = {NULL, 0, 0};
                    if (solve_exact_parse(nxt_buf, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, req_final_off, &plan_h)) {
                        UltraMetadata meta_h;
                        memset(&meta_h, 0, sizeof(meta_h));
                        meta_h.format = format;
                        meta_h.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                        meta_h.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                            ? detect_buffer_literal_bits(nxt_buf, input_size, NULL) : 8;
                        meta_h.filter_passes = 1;
                        meta_h.filter_stride = stride;
                        meta_h.filter_strides[0] = stride;
                        meta_h.filter_counts[0] = tail_c;
                        meta_h.compressed_bits = plan_h.payload_bits;

                        size_t size_h = 0;
                        uint8_t *stream_h = encode_plan_to_stream(
                            nxt_buf, &plan_h, format, ZX0_DEFAULT_LSB_BITS, &meta_h, &size_h
                        );
                        free(plan_h.tokens);

                        #pragma omp critical
                        {
                            if (size_h < best_size ||
                                (size_h == best_size && meta_h.compressed_bits < best_meta.compressed_bits)) {
                                free(best_stream);
                                best_stream = stream_h;
                                best_size = size_h;
                                best_meta = meta_h;
                            } else {
                                free(stream_h);
                            }
                        }
                    }

                    /* Multi-pass piecewise tail integration for STRD_82B, HYBRID_108B, PRO_111B, and ULTRA_128B */
                    if (max_passes > 1 && stride <= 16) {
                        memcpy(cur_buf, nxt_buf, (size_t)input_size);
                        for (int pass = 2; pass <= 3; pass++) {
                            if (tail_c <= pass * stride) {
                                break;
                            }
                            apply_forward_stride_difference(cur_buf, nxt_buf, input_size, stride, tail_c);
                            memcpy(cur_buf, nxt_buf, (size_t)input_size);
                            ParsePlan plan_mp = {NULL, 0, 0};
                            if (solve_exact_parse(cur_buf, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, req_final_off, &plan_mp)) {
                                UltraMetadata meta_mp;
                                memset(&meta_mp, 0, sizeof(meta_mp));
                                meta_mp.format = format;
                                meta_mp.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                meta_mp.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                     ? detect_buffer_literal_bits(cur_buf, input_size, NULL) : 8;
                                meta_mp.filter_passes = pass;
                                meta_mp.filter_stride = stride;
                                for (int p_idx = 0; p_idx < pass; p_idx++) {
                                    meta_mp.filter_strides[p_idx] = stride;
                                    meta_mp.filter_counts[p_idx] = tail_c;
                                }
                                meta_mp.compressed_bits = plan_mp.payload_bits;

                                size_t size_mp = 0;
                                uint8_t *stream_mp = encode_plan_to_stream(
                                    cur_buf, &plan_mp, format, ZX0_DEFAULT_LSB_BITS, &meta_mp, &size_mp
                                );
                                free(plan_mp.tokens);

                                #pragma omp critical
                                {
                                    if (size_mp < best_size ||
                                        (size_mp == best_size && meta_mp.compressed_bits < best_meta.compressed_bits)) {
                                        free(best_stream);
                                        best_stream = stream_mp;
                                        best_size = size_mp;
                                        best_meta = meta_mp;
                                    } else {
                                        free(stream_mp);
                                    }
                                }
                            }
                        }
                    }
                }

                free(cur_buf);
                free(nxt_buf);
            }

            /* C0. PRO_111B & ULTRA_128B Native 16-Bit Word-Stride-Delta Search (eliminates carry-ripple on 16-bit PCM/Coord/Ptr tables) */
            if (is_word_capable && input_size >= 4) {
                static const int word_strides[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
                int n_ws = (int)(sizeof(word_strides) / sizeof(word_strides[0]));
                #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
                for (int wi = 0; wi < n_ws; wi++) {
                    int wstride = word_strides[wi];
                    if (wstride + 2 > input_size) {
                        continue;
                    }
                    int max_wcount = ((input_size - wstride) & ~1);
                    if (max_wcount < 2 || max_wcount > 65534) {
                        continue;
                    }
                    uint8_t *wcur = (uint8_t *)malloc((size_t)input_size);
                    uint8_t *wnxt = (uint8_t *)malloc((size_t)input_size);
                    assert(wcur != NULL && wnxt != NULL);

                    /* 1) Full-buffer 1..4 pass 16-bit Word-Stride-Delta */
                    memcpy(wcur, working_input, (size_t)input_size);
                    int prev_wbits = COST_INFINITY;
                    for (int pass = 1; pass <= 4; pass++) {
                        if (input_size <= pass * wstride + 2) {
                            break;
                        }
                        apply_forward_word16_stride_difference(wcur, wnxt, input_size, wstride, max_wcount);
                        memcpy(wcur, wnxt, (size_t)input_size);
                        ParsePlan plan_w = {NULL, 0, 0};
                        if (solve_exact_parse(wcur, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, base_req_off, &plan_w)) {
                            UltraMetadata meta_w;
                            memset(&meta_w, 0, sizeof(meta_w));
                            meta_w.format = format;
                            meta_w.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                            meta_w.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                ? detect_buffer_literal_bits(wcur, input_size, NULL) : 8;
                            meta_w.filter_passes = pass;
                            meta_w.filter_stride = wstride;
                            for (int pk = 0; pk < pass; pk++) {
                                meta_w.filter_strides[pk] = wstride;
                                meta_w.filter_counts[pk] = max_wcount;
                                meta_w.filter_word_modes[pk] = 1;
                            }
                            meta_w.compressed_bits = plan_w.payload_bits;

                            size_t size_w = 0;
                            uint8_t *stream_w = encode_plan_to_stream(
                                wcur, &plan_w, format, ZX0_DEFAULT_LSB_BITS, &meta_w, &size_w
                            );
                            free(plan_w.tokens);

                            #pragma omp critical
                            {
                                if (size_w < best_size ||
                                    (size_w == best_size && meta_w.compressed_bits < best_meta.compressed_bits)) {
                                    free(best_stream);
                                    best_stream = stream_w;
                                    best_size = size_w;
                                    best_meta = meta_w;
                                } else {
                                    free(stream_w);
                                }
                            }
                            if (plan_w.payload_bits > prev_wbits + 512) {
                                break;
                            }
                            prev_wbits = plan_w.payload_bits;
                        }
                    }

                    /* 2) Harmonic 16-bit Word-Stride-Delta (wcount_p = ((input_size - pass * wstride) & ~1)) */
                    if (wstride <= 8) {
                        memcpy(wcur, working_input, (size_t)input_size);
                        int wharm_counts[4] = {0};
                        for (int pass = 1; pass <= 4; pass++) {
                            int whc = (input_size - pass * wstride) & ~1;
                            if (whc < 2 || whc > 65534) {
                                break;
                            }
                            apply_forward_word16_stride_difference(wcur, wnxt, input_size, wstride, whc);
                            memcpy(wcur, wnxt, (size_t)input_size);
                            wharm_counts[pass - 1] = whc;
                            if (pass >= 2) {
                                ParsePlan plan_wh = {NULL, 0, 0};
                                if (solve_exact_parse(wcur, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, base_req_off, &plan_wh)) {
                                    UltraMetadata meta_wh;
                                    memset(&meta_wh, 0, sizeof(meta_wh));
                                    meta_wh.format = format;
                                    meta_wh.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                    meta_wh.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                         ? detect_buffer_literal_bits(wcur, input_size, NULL) : 8;
                                    meta_wh.filter_passes = pass;
                                    meta_wh.filter_stride = wstride;
                                    for (int pk = 0; pk < pass; pk++) {
                                        meta_wh.filter_strides[pk] = wstride;
                                        meta_wh.filter_counts[pk] = wharm_counts[pass - 1 - pk];
                                        meta_wh.filter_word_modes[pk] = 1;
                                    }
                                    meta_wh.compressed_bits = plan_wh.payload_bits;

                                    size_t size_wh = 0;
                                    uint8_t *stream_wh = encode_plan_to_stream(
                                        wcur, &plan_wh, format, ZX0_DEFAULT_LSB_BITS, &meta_wh, &size_wh
                                    );
                                    free(plan_wh.tokens);

                                    #pragma omp critical
                                    {
                                        if (size_wh < best_size ||
                                            (size_wh == best_size && meta_wh.compressed_bits < best_meta.compressed_bits)) {
                                            free(best_stream);
                                            best_stream = stream_wh;
                                            best_size = size_wh;
                                            best_meta = meta_wh;
                                        } else {
                                            free(stream_wh);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    /* 3) Hybrid 16-bit Word-Stride (Order 1..2) followed by 8-bit Byte-Stride (stride=1 or stride=wstride) */
                    if (wstride <= 8) {
                        memcpy(wcur, working_input, (size_t)input_size);
                        for (int wpass = 1; wpass <= 2; wpass++) {
                            int whc = (input_size - wpass * wstride) & ~1;
                            if (whc < 2) break;
                            apply_forward_word16_stride_difference(wcur, wnxt, input_size, wstride, whc);
                            memcpy(wcur, wnxt, (size_t)input_size);

                            int b_strides[2] = {1, wstride};
                            for (int bi = 0; bi < 2; bi++) {
                                int bs = b_strides[bi];
                                int bc = input_size - bs;
                                if (bc <= 0) continue;
                                apply_forward_stride_difference(wcur, wnxt, input_size, bs, bc);
                                ParsePlan plan_wb = {NULL, 0, 0};
                                if (solve_exact_parse(wnxt, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, base_req_off, &plan_wb)) {
                                    UltraMetadata meta_wb;
                                    memset(&meta_wb, 0, sizeof(meta_wb));
                                    meta_wb.format = format;
                                    meta_wb.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                    meta_wb.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                         ? detect_buffer_literal_bits(wnxt, input_size, NULL) : 8;
                                    meta_wb.filter_passes = wpass + 1;
                                    meta_wb.filter_stride = bs;
                                    /* Decoder runs reverse: first 8-bit byte-add (bs), then wpass 16-bit word-add (wstride) */
                                    meta_wb.filter_strides[0] = bs;
                                    meta_wb.filter_counts[0] = bc;
                                    meta_wb.filter_word_modes[0] = 0;
                                    for (int pk = 0; pk < wpass; pk++) {
                                        meta_wb.filter_strides[1 + pk] = wstride;
                                        meta_wb.filter_counts[1 + pk] = (input_size - (wpass - pk) * wstride) & ~1;
                                        meta_wb.filter_word_modes[1 + pk] = 1;
                                    }
                                    meta_wb.compressed_bits = plan_wb.payload_bits;

                                    size_t size_wb = 0;
                                    uint8_t *stream_wb = encode_plan_to_stream(
                                        wnxt, &plan_wb, format, ZX0_DEFAULT_LSB_BITS, &meta_wb, &size_wb
                                    );
                                    free(plan_wb.tokens);

                                    #pragma omp critical
                                    {
                                        if (size_wb < best_size ||
                                            (size_wb == best_size && meta_wb.compressed_bits < best_meta.compressed_bits)) {
                                            free(best_stream);
                                            best_stream = stream_wb;
                                            best_size = size_wb;
                                            best_meta = meta_wb;
                                        } else {
                                            free(stream_wb);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    free(wcur);
                    free(wnxt);
                }
            }

            /* C. HYBRID_108B, PRO_111B & ULTRA_128B Chained Multi-Stage Search:
             *    1) True 2D/3D Spatial Prediction (stride S1 followed by distinct stride S2, and (S1, S1, S2))
             *    2) Chained x86 E8 + E9 + Multi-Segment 2D Stride-Delta (including 16-bit Word-Stride on PRO_111B/ULTRA_128B)
             */
            if (is_multi_stage) {
                static const int pair_s1[] = {1, 2, 3, 4, 6, 8, 12, 16, 30, 32};
                static const int pair_s2[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 30, 32, 48, 64, 128, 256};
                uint8_t *buf_p1 = (uint8_t *)malloc((size_t)input_size);
                uint8_t *buf_p2 = (uint8_t *)malloc((size_t)input_size);
                uint8_t *buf_p3 = (uint8_t *)malloc((size_t)input_size);
                assert(buf_p1 != NULL && buf_p2 != NULL && buf_p3 != NULL);

                for (size_t i1 = 0; i1 < sizeof(pair_s1) / sizeof(pair_s1[0]); i1++) {
                    int s1 = pair_s1[i1];
                    if (s1 >= input_size) continue;
                    int c1 = (input_size - s1) > 65535 ? 65535 : (input_size - s1);
                    apply_forward_stride_difference(working_input, buf_p1, input_size, s1, c1);

                    for (size_t i2 = 0; i2 < sizeof(pair_s2) / sizeof(pair_s2[0]); i2++) {
                        int s2 = pair_s2[i2];
                        if (s2 == s1 || s2 >= input_size) continue;
                        if (s1 > 4 && s2 != s1 + 2 && s1 != s2 + 2) continue;
                        int c2 = (input_size - s2) > 65535 ? 65535 : (input_size - s2);
                        apply_forward_stride_difference(buf_p1, buf_p2, input_size, s2, c2);
                        ParsePlan plan_2d = {NULL, 0, 0};
                        if (solve_exact_parse(buf_p2, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, base_req_off, &plan_2d)) {
                            UltraMetadata meta_2d;
                            memset(&meta_2d, 0, sizeof(meta_2d));
                            meta_2d.format = format;
                            meta_2d.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                            meta_2d.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                 ? detect_buffer_literal_bits(buf_p2, input_size, NULL) : 8;
                            meta_2d.filter_passes = 2;
                            meta_2d.filter_stride = s1;
                            meta_2d.filter_strides[0] = s2;
                            meta_2d.filter_counts[0] = c2;
                            meta_2d.filter_strides[1] = s1;
                            meta_2d.filter_counts[1] = c1;
                            meta_2d.compressed_bits = plan_2d.payload_bits;

                            size_t size_2d = 0;
                            uint8_t *stream_2d = encode_plan_to_stream(
                                buf_p2, &plan_2d, format, ZX0_DEFAULT_LSB_BITS, &meta_2d, &size_2d
                            );
                            free(plan_2d.tokens);

                            if (size_2d < best_size ||
                                (size_2d == best_size && meta_2d.compressed_bits < best_meta.compressed_bits)) {
                                free(best_stream);
                                best_stream = stream_2d;
                                best_size = size_2d;
                                best_meta = meta_2d;
                            } else {
                                free(stream_2d);
                            }
                        }
                    }

                    /* 3-pass heterogeneous (s1, s1, s2) for s1 in {1, 2} and s2 in {6, 16, 32} */
                    if ((s1 == 1 || s1 == 2) && input_size > 2 * s1 + 32) {
                        int c1b = input_size - 2 * s1;
                        apply_forward_stride_difference(buf_p1, buf_p2, input_size, s1, c1b);
                        static const int s3_cands[] = {6, 8, 16, 32};
                        for (size_t i3 = 0; i3 < sizeof(s3_cands) / sizeof(s3_cands[0]); i3++) {
                            int s2 = s3_cands[i3];
                            int c2 = input_size - 2 * s1 - s2;
                            if (c2 <= 0) continue;
                            apply_forward_stride_difference(buf_p2, buf_p3, input_size, s2, c2);
                            ParsePlan plan_3d = {NULL, 0, 0};
                            if (solve_exact_parse(buf_p3, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, base_req_off, &plan_3d)) {
                                UltraMetadata meta_3d;
                                memset(&meta_3d, 0, sizeof(meta_3d));
                                meta_3d.format = format;
                                meta_3d.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                meta_3d.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                     ? detect_buffer_literal_bits(buf_p3, input_size, NULL) : 8;
                                meta_3d.filter_passes = 3;
                                meta_3d.filter_stride = s1;
                                meta_3d.filter_strides[0] = s2;
                                meta_3d.filter_counts[0] = c2;
                                meta_3d.filter_strides[1] = s1;
                                meta_3d.filter_counts[1] = c1b;
                                meta_3d.filter_strides[2] = s1;
                                meta_3d.filter_counts[2] = c1;
                                meta_3d.compressed_bits = plan_3d.payload_bits;

                                size_t size_3d = 0;
                                uint8_t *stream_3d = encode_plan_to_stream(
                                    buf_p3, &plan_3d, format, ZX0_DEFAULT_LSB_BITS, &meta_3d, &size_3d
                                );
                                free(plan_3d.tokens);

                                if (size_3d < best_size ||
                                    (size_3d == best_size && meta_3d.compressed_bits < best_meta.compressed_bits)) {
                                    free(best_stream);
                                    best_stream = stream_3d;
                                    best_size = size_3d;
                                    best_meta = meta_3d;
                                } else {
                                    free(stream_3d);
                                }
                            }
                        }
                    }
                }
                free(buf_p1);
                free(buf_p2);
                free(buf_p3);

                if (best_e8_buf != NULL && best_e8_tail_chosen >= 3) {
                    static const int hyb_strides[] = {1, 2, 3, 4, 8, 12, 16, 24, 32, 64};
                    uint8_t *st_buf = (uint8_t *)malloc((size_t)input_size);
                    uint8_t *e8_on_st = (uint8_t *)malloc((size_t)input_size);
                    uint8_t *e9_on_st = (uint8_t *)malloc((size_t)input_size);
                    assert(st_buf != NULL && e8_on_st != NULL && e9_on_st != NULL);

                    /* Evaluate E8 + 3-pass Word-Stride=2 (PRO_111B/ULTRA_128B) and E8+E9 + (2,2,6) Byte-Stride (HYBRID_108B) on half-suffix */
                    if (input_size >= 512) {
                        int half_t = input_size / 2;
                        if (is_word_capable) {
                            uint8_t *w_cur = (uint8_t *)malloc((size_t)input_size);
                            memcpy(w_cur, working_input, (size_t)input_size);
                            for (int wp = 1; wp <= 3; wp++) {
                                int wc = (half_t - wp * 2) & ~1;
                                if (wc < 4) break;
                                apply_forward_word16_stride_difference(w_cur, st_buf, input_size, 2, wc);
                                memcpy(w_cur, st_buf, (size_t)input_size);
                                if (w_cur[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                    w_cur[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                    int h8w = apply_forward_x86_opcode_filter(
                                        w_cur, e8_on_st, input_size, best_e8_tail_chosen, X86_OPCODE_CALL_REL16
                                    );
                                    if (h8w >= 2) {
                                        ParsePlan plan_ew = {NULL, 0, 0};
                                        if (solve_exact_parse(e8_on_st, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, base_req_off, &plan_ew)) {
                                            UltraMetadata meta_ew;
                                            memset(&meta_ew, 0, sizeof(meta_ew));
                                            meta_ew.format = format;
                                            meta_ew.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                            meta_ew.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                                 ? detect_buffer_literal_bits(e8_on_st, input_size, NULL) : 8;
                                            meta_ew.filter_passes = 1 + wp;
                                            meta_ew.filter_stride = 0;
                                            meta_ew.filter_strides[0] = 0;
                                            meta_ew.filter_counts[0] = best_e8_tail_chosen;
                                            for (int pk = 0; pk < wp; pk++) {
                                                meta_ew.filter_strides[1 + pk] = 2;
                                                meta_ew.filter_counts[1 + pk] = (half_t - (wp - pk) * 2) & ~1;
                                                meta_ew.filter_word_modes[1 + pk] = 1;
                                            }
                                            meta_ew.compressed_bits = plan_ew.payload_bits;
                                            size_t size_ew = 0;
                                            uint8_t *stream_ew = encode_plan_to_stream(
                                                e8_on_st, &plan_ew, format, ZX0_DEFAULT_LSB_BITS, &meta_ew, &size_ew
                                            );
                                            free(plan_ew.tokens);
                                            if (size_ew < best_size ||
                                                (size_ew == best_size && meta_ew.compressed_bits < best_meta.compressed_bits)) {
                                                free(best_stream);
                                                best_stream = stream_ew;
                                                best_size = size_ew;
                                                best_meta = meta_ew;
                                            } else {
                                                free(stream_ew);
                                            }
                                        }
                                    }
                                }
                            }
                            free(w_cur);
                        } else {
                            /* HYBRID_108B: E8+E9 + (2, 2, 6) on half-suffix */
                            int c1 = half_t - 2, c2 = half_t - 4, c3 = half_t - 10;
                            if (c3 > 6) {
                                apply_forward_stride_difference(working_input, e9_on_st, input_size, 2, c1);
                                apply_forward_stride_difference(e9_on_st, st_buf, input_size, 2, c2);
                                apply_forward_stride_difference(st_buf, e9_on_st, input_size, 6, c3);
                                int h9 = apply_forward_x86_opcode_filter(
                                    e9_on_st, st_buf, input_size, best_e8_tail_chosen, X86_OPCODE_JMP_REL16
                                );
                                (void)h9;
                                int h8 = apply_forward_x86_opcode_filter(
                                    st_buf, e8_on_st, input_size, best_e8_tail_chosen, X86_OPCODE_CALL_REL16
                                );
                                if (h8 >= 2) {
                                    ParsePlan plan_eb = {NULL, 0, 0};
                                    if (solve_exact_parse(e8_on_st, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, base_req_off, &plan_eb)) {
                                        UltraMetadata meta_eb;
                                        memset(&meta_eb, 0, sizeof(meta_eb));
                                        meta_eb.format = format;
                                        meta_eb.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                        meta_eb.literal_bits = 8;
                                        meta_eb.filter_passes = 5;
                                        meta_eb.filter_stride = 0;
                                        meta_eb.filter_strides[0] = 0;
                                        meta_eb.filter_counts[0] = best_e8_tail_chosen;
                                        meta_eb.filter_strides[1] = 0;
                                        meta_eb.filter_counts[1] = best_e8_tail_chosen;
                                        meta_eb.filter_strides[2] = 6;
                                        meta_eb.filter_counts[2] = c3;
                                        meta_eb.filter_strides[3] = 2;
                                        meta_eb.filter_counts[3] = c2;
                                        meta_eb.filter_strides[4] = 2;
                                        meta_eb.filter_counts[4] = c1;
                                        meta_eb.compressed_bits = plan_eb.payload_bits;
                                        size_t size_eb = 0;
                                        uint8_t *stream_eb = encode_plan_to_stream(
                                            e8_on_st, &plan_eb, format, ZX0_DEFAULT_LSB_BITS, &meta_eb, &size_eb
                                        );
                                        free(plan_eb.tokens);
                                        if (size_eb < best_size ||
                                            (size_eb == best_size && meta_eb.compressed_bits < best_meta.compressed_bits)) {
                                            free(best_stream);
                                            best_stream = stream_eb;
                                            best_size = size_eb;
                                            best_meta = meta_eb;
                                        } else {
                                            free(stream_eb);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (size_t idx = 0; idx < sizeof(hyb_strides) / sizeof(hyb_strides[0]); idx++) {
                        int st = hyb_strides[idx];
                        if (st >= input_size) {
                            continue;
                        }
                        int kg = 0, ub = 0;
                        int kt = find_optimal_stride_tail_count(working_input, input_size, st, &kg, &ub);
                        int c_tails[4];
                        int n_ct = 0;
                        if (kt > st && kt < input_size - st) c_tails[n_ct++] = kt;
                        if (ub > st && ub < input_size - st && ub != kt) c_tails[n_ct++] = ub;
                        if (input_size / 2 > st) c_tails[n_ct++] = input_size / 2;
                        if (input_size >= 3072 && 2048 > st) c_tails[n_ct++] = 2048;

                        for (int ti = 0; ti < n_ct; ti++) {
                            int t_st = c_tails[ti];
                            apply_forward_stride_difference(working_input, st_buf, input_size, st, t_st);
                            for (int p = 1; p <= 2; p++) {
                                if (p == 2) {
                                    if (st > 4 || t_st <= 2 * st) break;
                                    uint8_t *tmp = (uint8_t *)malloc((size_t)input_size);
                                    apply_forward_stride_difference(st_buf, tmp, input_size, st, t_st);
                                    memcpy(st_buf, tmp, (size_t)input_size);
                                    free(tmp);
                                }
                                /* For PRO_111B & ULTRA_128B: can use 1-pass E8 (V=1049) directly followed by Stride without dummy E9! */
                                if (is_word_capable &&
                                    st_buf[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                    st_buf[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                    int h8_direct = apply_forward_x86_opcode_filter(
                                        st_buf, e8_on_st, input_size, best_e8_tail_chosen, X86_OPCODE_CALL_REL16
                                    );
                                    if (h8_direct >= 2) {
                                        ParsePlan plan_u1 = {NULL, 0, 0};
                                        if (solve_exact_parse(e8_on_st, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, base_req_off, &plan_u1)) {
                                            UltraMetadata meta_u1;
                                            memset(&meta_u1, 0, sizeof(meta_u1));
                                            meta_u1.format = format;
                                            meta_u1.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                            meta_u1.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                                 ? detect_buffer_literal_bits(e8_on_st, input_size, NULL) : 8;
                                            meta_u1.filter_passes = 1 + p;
                                            meta_u1.filter_stride = 0;
                                            meta_u1.filter_strides[0] = 0;
                                            meta_u1.filter_counts[0] = best_e8_tail_chosen;
                                            for (int pk = 0; pk < p; pk++) {
                                                meta_u1.filter_strides[1 + pk] = st;
                                                meta_u1.filter_counts[1 + pk] = t_st;
                                            }
                                            meta_u1.compressed_bits = plan_u1.payload_bits;

                                            size_t size_u1 = 0;
                                            uint8_t *stream_u1 = encode_plan_to_stream(
                                                e8_on_st, &plan_u1, format, ZX0_DEFAULT_LSB_BITS, &meta_u1, &size_u1
                                            );
                                            free(plan_u1.tokens);

                                            if (size_u1 < best_size ||
                                                (size_u1 == best_size && meta_u1.compressed_bits < best_meta.compressed_bits)) {
                                                free(best_stream);
                                                best_stream = stream_u1;
                                                best_size = size_u1;
                                                best_meta = meta_u1;
                                            } else {
                                                free(stream_u1);
                                            }
                                        }
                                    }
                                }

                                /* Decoder runs E8 -> E9 -> Stride, so encoder applies Stride -> E9 -> E8 */
                                int e9_tail = 3;
                                if (st_buf[input_size - 2] != X86_OPCODE_JMP_REL16 &&
                                    st_buf[input_size - 1] != X86_OPCODE_JMP_REL16 &&
                                    st_buf[input_size - 3] != X86_OPCODE_JMP_REL16) {
                                    memcpy(e9_on_st, st_buf, (size_t)input_size);
                                    if (e9_on_st[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                        e9_on_st[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                        int h8 = apply_forward_x86_opcode_filter(
                                            e9_on_st, e8_on_st, input_size, best_e8_tail_chosen, X86_OPCODE_CALL_REL16
                                        );
                                        if (h8 >= 2) {
                                            ParsePlan plan_hyb = {NULL, 0, 0};
                                            if (solve_exact_parse(e8_on_st, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, base_req_off, &plan_hyb)) {
                                                UltraMetadata meta_hyb;
                                                memset(&meta_hyb, 0, sizeof(meta_hyb));
                                                meta_hyb.format = format;
                                                meta_hyb.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                                meta_hyb.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                                      ? detect_buffer_literal_bits(e8_on_st, input_size, NULL) : 8;
                                                meta_hyb.filter_passes = 2 + p;
                                                meta_hyb.filter_stride = 0; /* Signals V=281/537 (E8+E9 entry) */
                                                meta_hyb.filter_strides[0] = 0;
                                                meta_hyb.filter_counts[0] = best_e8_tail_chosen;
                                                meta_hyb.filter_strides[1] = 0;
                                                meta_hyb.filter_counts[1] = e9_tail;
                                                for (int pk = 0; pk < p; pk++) {
                                                    meta_hyb.filter_strides[2 + pk] = st;
                                                    meta_hyb.filter_counts[2 + pk] = t_st;
                                                }
                                                meta_hyb.compressed_bits = plan_hyb.payload_bits;

                                                size_t size_hyb = 0;
                                                uint8_t *stream_hyb = encode_plan_to_stream(
                                                    e8_on_st, &plan_hyb, format, ZX0_DEFAULT_LSB_BITS, &meta_hyb, &size_hyb
                                                );
                                                free(plan_hyb.tokens);

                                                if (size_hyb < best_size ||
                                                    (size_hyb == best_size && meta_hyb.compressed_bits < best_meta.compressed_bits)) {
                                                    free(best_stream);
                                                    best_stream = stream_hyb;
                                                    best_size = size_hyb;
                                                    best_meta = meta_hyb;
                                                } else {
                                                    free(stream_hyb);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    free(st_buf);
                    free(e8_on_st);
                    free(e9_on_st);
                }
            }
        }
        free(best_e8_buf);
    }

    assert(best_stream != NULL);

    /* Mandatory round-trip self-verification against working_input */
    uint8_t *verified = ultra_decompress_buffer(best_stream, best_size, (size_t)input_size, format);
    assert(verified != NULL);
    assert(memcmp(verified, working_input, (size_t)input_size) == 0);
    free(verified);
    free(working_input);

    *compressed_size_out = best_size;
    if (metadata_out != NULL) {
        *metadata_out = best_meta;
    }
    return best_stream;
}
