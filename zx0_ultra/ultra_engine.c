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

/*
 * Exact O(N * W) Multi-State Automaton DP Solver with Zero-Loss Dominated-Literal
 * Pruning, Global O(1) Literal Frontier (lit_frontiers[0]), Single-Cache-Line Frontiers,
 * and Topological Mark-and-Compact GC.
 * If required_final_offset > 0, only terminal states whose active last_offset equals
 * required_final_offset are accepted (enabling 0-byte stride passing via register BX).
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
        int literal_pos_bits = ZX0_LITERAL_BITS_PER_BYTE * pos;
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
} BitWriter;

static void bit_writer_init(BitWriter *bw, size_t initial_capacity, int initial_backtrack) {
    bw->capacity = initial_capacity > 64 ? initial_capacity : 64;
    bw->data = (uint8_t *)calloc(bw->capacity, 1);
    assert(bw->data != NULL);
    bw->size = 0;
    bw->bit_byte_index = 0;
    bw->bit_mask = 0;
    bw->backtrack = initial_backtrack;
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
        if (bw->bit_mask == 0) {
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
    BitWriter bw;
    bit_writer_init(&bw, (size_t)(plan->payload_bits / 8 + 32), use_zx0_lsb_byte ? 1 : 0);

    if (format == ULTRA_FORMAT_ADPK_77B) {
        bit_writer_put_byte(&bw, (uint8_t)offset_lsb_bits);
    }

    int last_offset = ZX0_INITIAL_OFFSET;
    int input_pos = 0;

    for (int t = 0; t < plan->token_count; t++) {
        int length = plan->tokens[t].length;
        int offset = plan->tokens[t].offset;

        if (offset == 0) {
            if (use_zx0_lsb_byte || t > 0) {
                bit_writer_put_bit(&bw, 0);
            }
            bit_writer_put_elias(&bw, length, 0);
            for (int i = 0; i < length; i++) {
                bit_writer_put_byte(&bw, working_input[input_pos++]);
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

    /* Emit ZX0 end-of-stream marker (COMPAT_82B & X86E8_84B use V=257; UNIV_93B uses V=257 for Stride or V=281 for E8) */
    bit_writer_put_bit(&bw, 1);
    int eof_symbol = ZX0_EOF_MSB_VALUE;
    if ((format == ULTRA_FORMAT_COMPAT_82B || format == ULTRA_FORMAT_X86E8_84B) && meta->filter_passes > 0) {
        eof_symbol = ZX0_EOF_ALT_VALUE;
    } else if (format == ULTRA_FORMAT_UNIV_93B && meta->filter_passes > 0) {
        eof_symbol = (meta->filter_stride == 0) ? ZX0_EOF_E8_VALUE : ZX0_EOF_ALT_VALUE;
    }
    bit_writer_put_elias(&bw, eof_symbol, 1);

    /* Emit post-filter control words for STRD_80B, STRD_82B, COMPAT_82B, X86E8_84B, and UNIV_93B */
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
    return (br->bit_value & br->bit_mask) ? 1 : 0;
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

    int offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
    if (format == ULTRA_FORMAT_ADPK_77B) {
        int hdr = bit_reader_get_byte(&br);
        if (hdr < 1 || hdr > MAX_ADPK_LSB_BITS) {
            return NULL;
        }
        offset_lsb_bits = hdr;
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
        for (int i = 0; i < lit_len; i++) {
            int b = bit_reader_get_byte(&br);
            if (b < 0) {
                free(out);
                return NULL;
            }
            out[out_pos++] = (uint8_t)b;
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

/*
 * O(N) Kadane suffix-gain scan to locate an optimal piecewise tail boundary [input_size - count .. input_size)
 * where Stride-Delta increases local byte predictability without corrupting an earlier code/header prefix.
 * Detects both cross-byte constant deltas (d_cur == d_prv) and same-channel linear ramps (d_cur == d_stride_prev).
 */
static int find_optimal_stride_tail_count(
    const uint8_t *src,
    int input_size,
    int stride,
    int *best_gain_out
) {
    int max_tail = input_size - stride;
    if (max_tail > 65535) {
        max_tail = 65535;
    }
    if (max_tail <= stride) {
        *best_gain_out = 0;
        return 0;
    }

    int min_idx = input_size - max_tail;
    int suffix_gain = 0;
    int best_gain = 0;
    int best_tail = 0;

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
            if (d_cur == d_str) {
                dif_hit = 1;
            }
        }
        int delta_score = 2 * (dif_hit - raw_hit) - ((dif_hit == 0 && raw_hit == 0) ? 1 : 0);
        suffix_gain += delta_score;
        if (suffix_gain > best_gain) {
            best_gain = suffix_gain;
            best_tail = input_size - i;
        }
    }

    *best_gain_out = best_gain;
    return best_tail;
}

/*
 * Exact backward inverse of unzx0_x86e8_84b.asm / unzx0_univ_93b.asm forward scasb/add [di],cx loop.
 * Because each step inspects byte[i] and modifies only byte[i+1..i+2], reversing the order
 * from i = input_size - 3 down to start_idx forms a strict bijection even across adjacent 0xE8 bytes.
 */
static int apply_forward_x86_e8_filter(
    const uint8_t *src,
    uint8_t *dst,
    int input_size,
    int tail_count
) {
    if (input_size < 3 || tail_count < 3 || tail_count > input_size || tail_count > 65535) {
        return 0;
    }
    if (src[input_size - 2] == X86_OPCODE_CALL_REL16 ||
        src[input_size - 1] == X86_OPCODE_CALL_REL16) {
        return 0;
    }

    memcpy(dst, src, (size_t)input_size);
    int start_idx = input_size - tail_count;
    int hits = 0;

    for (int i = input_size - 3; i >= start_idx; i--) {
        if (dst[i] == X86_OPCODE_CALL_REL16) {
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
         * - ULTRA_FORMAT_X86E8_84B (Zero-overhead ZX0 + E8 Call-Target filter)
         * - ULTRA_FORMAT_STRD_80B  (16-bit Stride-Delta filter)
         * - ULTRA_FORMAT_COMPAT_82B (Zero-overhead ZX0 + 16-bit Stride-Delta filter)
         * - ULTRA_FORMAT_STRD_82B  (Multi-pass Order-1..5 Stride-Delta filter)
         * - ULTRA_FORMAT_UNIV_93B  (Universal Zero-overhead ZX0 + 16-bit Stride-Delta + E8 Call-Target)
         */
        int base_payload_bits = 0;
        {
            ParsePlan base_plan = {NULL, 0, 0};
            int ok = solve_exact_parse(working_input, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, &base_plan);
            assert(ok);
            base_payload_bits = base_plan.payload_bits;
            best_meta.compressed_bits = base_plan.payload_bits;
            best_stream = encode_plan_to_stream(working_input, &base_plan, format, ZX0_DEFAULT_LSB_BITS, &best_meta, &best_size);
            free(base_plan.tokens);
        }

        /* A. Evaluate position-independent x86 E8 Call-Target transform (X86E8_84B and UNIV_93B) */
        if ((format == ULTRA_FORMAT_X86E8_84B || format == ULTRA_FORMAT_UNIV_93B) &&
            input_size >= 3 &&
            working_input[input_size - 2] != X86_OPCODE_CALL_REL16 &&
            working_input[input_size - 1] != X86_OPCODE_CALL_REL16) {
            int full_tail = input_size <= 65535 ? input_size : 65535;
            int cand_tails[3];
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
            for (int i = min_scan_idx; i <= input_size - 3; i++) {
                if (working_input[i] == X86_OPCODE_CALL_REL16) {
                    uint16_t rem = (uint16_t)(full_tail - (i - min_scan_idx));
                    uint16_t tword = (uint16_t)((working_input[i + 1] | (working_input[i + 2] << 8)) - rem);
                    if (target_freq[tword] >= 2) {
                        int repeated_tail = input_size - i;
                        if (repeated_tail >= 3 && repeated_tail < full_tail - 32) {
                            cand_tails[num_tails++] = repeated_tail;
                        }
                        break;
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
                if (!dup && num_tails < 3) {
                    cand_tails[num_tails++] = best_e8_tail;
                }
            }
            free(target_freq);

            int nested_parallel = omp_in_parallel();
            #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
            for (int ti = 0; ti < num_tails; ti++) {
                int tail_count = cand_tails[ti];
                uint8_t *e8_buf = (uint8_t *)malloc((size_t)input_size);
                assert(e8_buf != NULL);
                int hits = apply_forward_x86_e8_filter(working_input, e8_buf, input_size, tail_count);
                if (hits >= 2) {
                    ParsePlan plan_e8 = {NULL, 0, 0};
                    if (solve_exact_parse(e8_buf, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, &plan_e8)) {
                        UltraMetadata meta_e8;
                        memset(&meta_e8, 0, sizeof(meta_e8));
                        meta_e8.format = format;
                        meta_e8.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                        meta_e8.filter_passes = 1;
                        meta_e8.filter_stride = 0;
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
                        }
                    }
                }
                free(e8_buf);
            }
        }

        /* B. Multi-core search over candidate strides (STRD_80B, COMPAT_82B, STRD_82B, UNIV_93B) */
        if (format != ULTRA_FORMAT_X86E8_84B) {
            int candidate_strides[80];
            int num_strides = 0;
            for (int s = 1; s <= 64 && s < input_size; s++) {
                candidate_strides[num_strides++] = s;
            }
            static const int extra_strides[] = {72, 80, 96, 128, 160, 256, 320};
            for (size_t idx = 0; idx < sizeof(extra_strides) / sizeof(extra_strides[0]); idx++) {
                if (extra_strides[idx] < input_size) {
                    candidate_strides[num_strides++] = extra_strides[idx];
                }
            }

            int max_passes = (format == ULTRA_FORMAT_STRD_82B) ? ULTRA_MAX_FILTER_PASSES : 1;
            int nested_parallel = omp_in_parallel();

            #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
            for (int si = 0; si < num_strides; si++) {
                int stride = candidate_strides[si];
                int max_full_count = input_size - stride;
                int full_count = max_full_count > 65535 ? 65535 : max_full_count;
                if (full_count <= 0) {
                    continue;
                }

                int kadane_gain = 0;
                int kadane_tail = find_optimal_stride_tail_count(working_input, input_size, stride, &kadane_gain);

                /* Fast O(N) compressibility pre-check including same-channel linear ramps (d_cur == d_str) */
                int is_structural_stride = (stride == 1 || stride == 2 || stride == 3 || stride == 4 ||
                                            stride == 6 || stride == 8 || stride == 12 || stride == 16 ||
                                            stride == 24 || stride == 32 || stride == 48 || stride == 64 ||
                                            stride == 72 || stride == 80 || stride == 96 ||
                                            stride == 128 || stride == 160 || stride == 256 || stride == 320);
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
                        hit = (d_cur == d_str);
                    }
                    if (hit) {
                        eq_dif++;
                    }
                }
                if (kadane_gain < 16 && (!is_structural_stride || eq_dif * 4 < eq_raw * 3)) {
                    continue;
                }

                uint8_t *cur_buf = (uint8_t *)malloc((size_t)input_size);
                uint8_t *nxt_buf = (uint8_t *)malloc((size_t)input_size);
                assert(cur_buf != NULL && nxt_buf != NULL);
                memcpy(cur_buf, working_input, (size_t)input_size);

                int active_counts[ULTRA_MAX_FILTER_PASSES] = {0};
                int prev_pass_bits = COST_INFINITY;

                for (int pass = 1; pass <= max_passes; pass++) {
                    if (input_size <= pass * stride) {
                        break;
                    }
                    apply_forward_stride_difference(cur_buf, nxt_buf, input_size, stride, full_count);
                    memcpy(cur_buf, nxt_buf, (size_t)input_size);
                    active_counts[pass - 1] = full_count;

                    ParsePlan plan_p = {NULL, 0, 0};
                    if (solve_exact_parse(cur_buf, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, stride, &plan_p)) {
                        int pass_bits = plan_p.payload_bits;
                        UltraMetadata meta_p;
                        memset(&meta_p, 0, sizeof(meta_p));
                        meta_p.format = format;
                        meta_p.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                        meta_p.filter_passes = pass;
                        meta_p.filter_stride = stride;
                        memcpy(meta_p.filter_counts, active_counts, sizeof(active_counts));
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

                        if (pass_bits > base_payload_bits + 512 || pass_bits > prev_pass_bits + 64) {
                            break;
                        }
                        prev_pass_bits = pass_bits;
                    }
                }

                /* Harmonic multi-pass counts (count_p = input_size - pass * stride) avoiding boundary artifacts */
                if (max_passes > 1 && stride <= 8 && prev_pass_bits < base_payload_bits) {
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
                            if (solve_exact_parse(cur_buf, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, stride, &plan_hm)) {
                                int hm_bits = plan_hm.payload_bits;
                                UltraMetadata meta_hm;
                                memset(&meta_hm, 0, sizeof(meta_hm));
                                meta_hm.format = format;
                                meta_hm.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                meta_hm.filter_passes = pass;
                                meta_hm.filter_stride = stride;
                                memcpy(meta_hm.filter_counts, harm_counts, sizeof(harm_counts));
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
                                if (hm_bits > prev_harm_bits + 64) {
                                    break;
                                }
                                prev_harm_bits = hm_bits;
                            }
                        }
                    }
                }

                /* Evaluate piecewise half-tail and Kadane-optimal tail filtering for hybrid binaries */
                int cand_piecewise[2];
                int num_piecewise = 0;
                int half_count = input_size / 2;
                if (!nested_parallel && input_size >= 128 && half_count > stride && half_count < full_count &&
                    (input_size <= 8192 || stride <= 2 || kadane_gain >= 16)) {
                    cand_piecewise[num_piecewise++] = half_count;
                }
                if (!nested_parallel && kadane_gain >= 16 && kadane_tail > stride && kadane_tail < full_count &&
                    (num_piecewise == 0 || abs(kadane_tail - cand_piecewise[0]) > 16)) {
                    cand_piecewise[num_piecewise++] = kadane_tail;
                }

                for (int pi = 0; pi < num_piecewise; pi++) {
                    int tail_c = cand_piecewise[pi];
                    apply_forward_stride_difference(working_input, nxt_buf, input_size, stride, tail_c);
                    ParsePlan plan_h = {NULL, 0, 0};
                    if (solve_exact_parse(nxt_buf, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, stride, &plan_h)) {
                        UltraMetadata meta_h;
                        memset(&meta_h, 0, sizeof(meta_h));
                        meta_h.format = format;
                        meta_h.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                        meta_h.filter_passes = 1;
                        meta_h.filter_stride = stride;
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
                }

                free(cur_buf);
                free(nxt_buf);
            }
        }
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
