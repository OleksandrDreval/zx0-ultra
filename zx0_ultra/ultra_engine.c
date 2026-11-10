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
#define EXACT_FRONTIER_CAPACITY     8
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

static void node_pool_ensure_capacity(NodePool *pool, int min_capacity) {
    int target = min_capacity > INITIAL_NODE_POOL_CAPACITY ? min_capacity : INITIAL_NODE_POOL_CAPACITY;
    if (pool->capacity < target) {
        pool->capacity = target;
        pool->nodes = (ParseNode *)realloc(pool->nodes, (size_t)pool->capacity * sizeof(ParseNode));
        assert(pool->nodes != NULL);
    }
    pool->count = 0;
}

static inline int node_pool_push(NodePool *pool, int prev_node, int end_pos, int offset) {
    if (__builtin_expect(pool->count >= pool->capacity, 0)) {
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
    const int *lit_min_base,
    MonotonicFrontier *rep_frontiers,
    const int *rep_valid_pos,
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
        int lc = (off == 0 || lit_min_base[off] < COST_INFINITY) ? lit_frontiers[off].count : 0;
        for (int k = 0; k < lc; k++) {
            int nid = lit_frontiers[off].entries[k].node_idx;
            if (nid >= 0) {
                remap[nid] = 1;
            }
        }
        if (off >= 1 && rep_valid_pos[off] >= 0) {
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
        int lc = (off == 0 || lit_min_base[off] < COST_INFINITY) ? lit_frontiers[off].count : 0;
        for (int k = 0; k < lc; k++) {
            int nid = lit_frontiers[off].entries[k].node_idx;
            if (nid >= 0) {
                lit_frontiers[off].entries[k].node_idx = remap[nid];
            }
        }
        if (off >= 1 && rep_valid_pos[off] >= 0) {
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
    if (c >= 2) {
        int out = 1;
        for (int k = 1; k < c; k++) {
            int pa = frontier->entries[out - 1].pos;
            int pb = frontier->entries[k].pos;
            int da = (pos + 1) - pa;
            int db = (pos + 1) - pb;
            if (db >= (pb - pa) && (frontier->entries[k].base_cost - frontier->entries[out - 1].base_cost) >= 2 &&
                (da < (db << 1))) {
                continue;
            }
            frontier->entries[out++] = frontier->entries[k];
        }
        c = out;
    }
    frontier->count = c;
    if (c > 0) {
        if ((base_cost - frontier->entries[0].base_cost) > 64) {
            return;
        }
        int last_delta = pos - frontier->entries[c - 1].pos;
        if (base_cost - frontier->entries[c - 1].base_cost >= compute_elias_gamma_bits(last_delta + 1) - 1) {
            return;
        }
    }
    if (c < EXACT_FRONTIER_CAPACITY) {
        frontier->entries[c].pos = pos;
        frontier->entries[c].base_cost = base_cost;
        frontier->entries[c].node_idx = node_idx;
        frontier->count = c + 1;
    } else {
        frontier->entries[EXACT_FRONTIER_CAPACITY - 1].pos = pos;
        frontier->entries[EXACT_FRONTIER_CAPACITY - 1].base_cost = base_cost;
        frontier->entries[EXACT_FRONTIER_CAPACITY - 1].node_idx = node_idx;
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

typedef struct {
    uint8_t bit_len[256];
    uint8_t prefix_bit[256];
    uint8_t payload_len[256];
    uint8_t payload_val[256];
    uint8_t min_len;
} DhSpec;

static void build_dh_spec(int dh, DhSpec *spec) {
    memset(spec, 0, sizeof(*spec));
    spec->min_len = 99;
    const int p_order[2] = {1, 0};
    for (int pi = 0; pi < 2; pi++) {
        int p = p_order[pi];
        int v = (p == 0) ? dh : ((dh & 5) + 1);
        int k = 0;
        while ((1 << (k + 1)) <= v) {
            k++;
        }
        int m = 8 - k;
        int base = (v - (1 << k)) << m;
        for (int val = base; val < base + (1 << m) && val < 256; val++) {
            uint8_t out_b = (uint8_t)((val - 1) & 0xFF);
            uint8_t total_bits = (uint8_t)(1 + m);
            if (spec->bit_len[out_b] == 0 || total_bits < spec->bit_len[out_b]) {
                spec->bit_len[out_b] = total_bits;
                spec->prefix_bit[out_b] = (uint8_t)p;
                spec->payload_len[out_b] = (uint8_t)m;
                spec->payload_val[out_b] = (uint8_t)(val - base);
                if (total_bits < spec->min_len) {
                    spec->min_len = total_bits;
                }
            }
        }
    }
}

typedef struct {
    int lit_min_base;
    int lit_max_pos;
    int rep_valid_pos;
    int new_off_overhead;
} OffsetState;

typedef struct {
    int cap_n;
    int cap_w;
    NodePool pool;
    int *lit_cum_bits;
    int *prev_same_byte;
    int *prev_same_pair;
    int *run_start;
    int *best_any_cost;
    int *best_any_node;
    int *best_len_cost;
    int *best_len_start;
    int blk64_cost[1024];
    int blk64_pos[1024];
    int blk1024_cost[64];
    int blk1024_pos[64];
    int lit_blk64_min[1024];
    uint64_t lit_act64[1024];
    int lit_blk8_min[4096];
    OffsetState *off_state;
    MonotonicFrontier *lit_frontiers;
    MonotonicFrontier *rep_frontiers;
    int *lit_min_base;
    int *lit_max_pos;
    int *lit_node0;
    int *rep_pos0;
    int *rep_cost0;
    int *rep_node0;
    int *new_off_overhead;
    int *match_lengths;
    int *last_match_pos;
    int *rep_valid_pos;
    int *final_lit_cost;
    int *final_lit_node;
    int *final_mat_cost;
    int *final_mat_node;
    int pair_pos_initialized;
    int last_pair_pos[65536];
    int last_quad_pos[65536];
} DpWorkspace;

static _Thread_local DpWorkspace tls_dp = {0};
static int g_deep_mode = 1;
static _Thread_local int tls_deep_mode = -1;

static void dp_workspace_ensure(DpWorkspace *ws, int n, int w) {
    if (n + 1 > ws->cap_n) {
        int new_n = (n + 1) > 4096 ? (n + 1) : 4096;
        ws->lit_cum_bits   = (int *)realloc(ws->lit_cum_bits,   (size_t)new_n * sizeof(int));
        ws->prev_same_byte = (int *)realloc(ws->prev_same_byte, (size_t)new_n * sizeof(int));
        ws->prev_same_pair = (int *)realloc(ws->prev_same_pair, (size_t)new_n * sizeof(int));
        ws->run_start      = (int *)realloc(ws->run_start,      (size_t)new_n * sizeof(int));
        ws->best_any_cost  = (int *)realloc(ws->best_any_cost,  (size_t)new_n * sizeof(int));
        ws->best_any_node  = (int *)realloc(ws->best_any_node,  (size_t)new_n * sizeof(int));
        ws->best_len_cost  = (int *)realloc(ws->best_len_cost,  (size_t)new_n * sizeof(int));
        ws->best_len_start = (int *)realloc(ws->best_len_start, (size_t)new_n * sizeof(int));
        assert(ws->lit_cum_bits && ws->prev_same_byte && ws->prev_same_pair && ws->run_start &&
               ws->best_any_cost && ws->best_any_node && ws->best_len_cost && ws->best_len_start);
        ws->cap_n = new_n;
    }
    if (w + 64 > ws->cap_w) {
        int new_w = (w + 64) > 4096 ? (w + 64) : 4096;
        ws->lit_frontiers    = (MonotonicFrontier *)realloc(ws->lit_frontiers,   (size_t)new_w * sizeof(MonotonicFrontier));
        ws->rep_frontiers    = (MonotonicFrontier *)realloc(ws->rep_frontiers,   (size_t)new_w * sizeof(MonotonicFrontier));
        ws->lit_min_base     = (int *)realloc(ws->lit_min_base,     (size_t)new_w * sizeof(int));
        ws->lit_max_pos      = (int *)realloc(ws->lit_max_pos,      (size_t)new_w * sizeof(int));
        ws->lit_node0        = (int *)realloc(ws->lit_node0,        (size_t)new_w * sizeof(int));
        ws->rep_pos0         = (int *)realloc(ws->rep_pos0,         (size_t)new_w * sizeof(int));
        ws->rep_cost0        = (int *)realloc(ws->rep_cost0,        (size_t)new_w * sizeof(int));
        ws->rep_node0        = (int *)realloc(ws->rep_node0,        (size_t)new_w * sizeof(int));
        ws->new_off_overhead = (int *)realloc(ws->new_off_overhead, (size_t)new_w * sizeof(int));
        ws->match_lengths    = (int *)realloc(ws->match_lengths,    (size_t)new_w * sizeof(int));
        ws->last_match_pos   = (int *)realloc(ws->last_match_pos,   (size_t)new_w * sizeof(int));
        ws->rep_valid_pos    = (int *)realloc(ws->rep_valid_pos,    (size_t)new_w * sizeof(int));
        ws->off_state        = (OffsetState *)realloc(ws->off_state, (size_t)new_w * sizeof(OffsetState));
        ws->final_lit_cost   = (int *)realloc(ws->final_lit_cost,   (size_t)new_w * sizeof(int));
        ws->final_lit_node   = (int *)realloc(ws->final_lit_node,   (size_t)new_w * sizeof(int));
        ws->final_mat_cost   = (int *)realloc(ws->final_mat_cost,   (size_t)new_w * sizeof(int));
        ws->final_mat_node   = (int *)realloc(ws->final_mat_node,   (size_t)new_w * sizeof(int));
        assert(ws->lit_frontiers && ws->rep_frontiers && ws->lit_min_base && ws->lit_max_pos &&
               ws->new_off_overhead && ws->match_lengths && ws->last_match_pos &&
               ws->rep_valid_pos && ws->final_lit_cost && ws->final_lit_node &&
               ws->final_mat_cost && ws->final_mat_node);
        ws->cap_w = new_w;
    }
    node_pool_ensure_capacity(&ws->pool, n * 4 + 1024);
}

static inline uint16_t hash_quad_bytes(const uint8_t *p) {
    uint32_t v = ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return (uint16_t)((v * 2654435761u) >> 16);
}

static int find_best_dh_sentinel(
    const uint8_t *input_data,
    int input_size,
    const uint8_t *wildcard_mask
) {
    int best_dh = 4;
    DhSpec spec;

    /*
     * Estimate which positions are likely to be emitted as literals rather than LZ77 matches (>= 2B)
     * in O(N) time with zero heap allocations using both 4-byte (long-range periodic/block match)
     * and 2-byte (short-range match) hash tables, then evaluate all non-dominated DH sentinels (3..127).
     * Note: dh=1 and dh=2 produce 8 bits for 128 bytes and 9 bits for 128 bytes, which is strictly
     * dominated on every input by dh=3 (which encodes all 256 bytes in 8 bits).
     */
    if (!tls_dp.pair_pos_initialized) {
        memset(tls_dp.last_pair_pos, -1, sizeof(tls_dp.last_pair_pos));
        memset(tls_dp.last_quad_pos, -1, sizeof(tls_dp.last_quad_pos));
        tls_dp.pair_pos_initialized = 1;
    } else {
        for (int i = 0; i + 1 < input_size; i++) {
            tls_dp.last_pair_pos[((int)input_data[i] << 8) | (int)input_data[i + 1]] = -1;
            if (i + 3 < input_size) {
                tls_dp.last_quad_pos[hash_quad_bytes(&input_data[i])] = -1;
            }
        }
    }

    int hist_whole[256] = {0};
    int hist_lit[256] = {0};
    int wildcard_count = 0;
    int lit_count = 0;

    for (int i = 0; i < input_size; i++) {
        if (wildcard_mask != NULL && wildcard_mask[i] != 0) {
            wildcard_count++;
        } else {
            hist_whole[input_data[i]]++;
        }
    }

    int *last_pair_pos = tls_dp.last_pair_pos;
    int *last_quad_pos = tls_dp.last_quad_pos;
    int pos = 0;
    while (pos < input_size) {
        if (wildcard_mask != NULL && wildcard_mask[pos] != 0) {
            pos++;
            continue;
        }
        int best_ml = 0;
        if (pos + 3 < input_size && wildcard_mask == NULL) {
            int qh = (int)hash_quad_bytes(&input_data[pos]);
            int qprev = last_quad_pos[qh];
            if (qprev >= 0 && (pos - qprev) <= ZX0_MAX_OFFSET) {
                int qml = 0;
                while (pos + qml < input_size && input_data[qprev + qml] == input_data[pos + qml]) {
                    qml++;
                }
                if (qml > best_ml) {
                    best_ml = qml;
                }
            }
        }
        if (pos + 1 < input_size && (wildcard_mask == NULL || wildcard_mask[pos + 1] == 0)) {
            int h = ((int)input_data[pos] << 8) | (int)input_data[pos + 1];
            int prev = last_pair_pos[h];
            if (prev >= 0 && (pos - prev) <= ZX0_MAX_OFFSET) {
                int pml = 0;
                while (pos + pml < input_size &&
                       (wildcard_mask == NULL || wildcard_mask[pos + pml] == 0) &&
                       input_data[prev + pml] == input_data[pos + pml]) {
                    pml++;
                }
                if (pml > best_ml) {
                    best_ml = pml;
                }
            }
        }
        if (pos > 0 && (wildcard_mask == NULL || wildcard_mask[pos - 1] == 0) &&
            input_data[pos] == input_data[pos - 1]) {
            int rle_ml = 0;
            while (pos + rle_ml < input_size &&
                   (wildcard_mask == NULL || wildcard_mask[pos + rle_ml] == 0) &&
                   input_data[pos + rle_ml] == input_data[pos - 1]) {
                rle_ml++;
            }
            if (rle_ml > best_ml) {
                best_ml = rle_ml;
            }
        }
        if (best_ml >= 2) {
            for (int k = 0; k < best_ml; k++) {
                if (pos + k + 1 < input_size) {
                    last_pair_pos[((int)input_data[pos + k] << 8) | (int)input_data[pos + k + 1]] = pos + k;
                }
                if (pos + k + 3 < input_size && wildcard_mask == NULL) {
                    last_quad_pos[hash_quad_bytes(&input_data[pos + k])] = pos + k;
                }
            }
            pos += best_ml;
        } else {
            hist_lit[input_data[pos]]++;
            lit_count++;
            if (pos + 1 < input_size) {
                last_pair_pos[((int)input_data[pos] << 8) | (int)input_data[pos + 1]] = pos;
            }
            if (pos + 3 < input_size && wildcard_mask == NULL) {
                last_quad_pos[hash_quad_bytes(&input_data[pos])] = pos;
            }
            pos++;
        }
    }

    int use_lit_estimate = (lit_count >= 8);
    int best_bits = use_lit_estimate ? (lit_count * 9) : (input_size * 9);
    int best_whole_bits = input_size * 9;

    for (int dh = 3; dh <= 127; dh++) {
        build_dh_spec(dh, &spec);
        int sum_lit = 0;
        int sum_whole = wildcard_count * (int)spec.min_len;
        int ok = 1;
        for (int b = 0; b < 256; b++) {
            int cw = hist_whole[b];
            if (cw > 0) {
                uint8_t l = spec.bit_len[b];
                if (l == 0) {
                    ok = 0;
                    break;
                }
                sum_whole += cw * (int)l;
                sum_lit += (use_lit_estimate ? hist_lit[b] : cw) * (int)l;
            }
        }
        if (!ok) {
            continue;
        }
        if (sum_lit < best_bits ||
            (sum_lit == best_bits && sum_whole < best_whole_bits) ||
            (sum_lit == best_bits && sum_whole == best_whole_bits && dh == 4)) {
            best_bits = sum_lit;
            best_whole_bits = sum_whole;
            best_dh = dh;
        }
    }

    return best_dh;
}

static int find_best_whole_dh_sentinel(
    const uint8_t *input_data,
    int input_size,
    const uint8_t *wildcard_mask
) {
    int best_dh = 4;
    int best_bits = input_size * 9;
    DhSpec spec;

    int hist_whole[256] = {0};
    int wildcard_count = 0;
    for (int i = 0; i < input_size; i++) {
        if (wildcard_mask != NULL && wildcard_mask[i] != 0) {
            wildcard_count++;
        } else {
            hist_whole[input_data[i]]++;
        }
    }

    for (int dh = 3; dh <= 127; dh++) {
        build_dh_spec(dh, &spec);
        int sum = wildcard_count * (int)spec.min_len;
        int ok = 1;
        for (int b = 0; b < 256; b++) {
            int cw = hist_whole[b];
            if (cw > 0) {
                uint8_t l = spec.bit_len[b];
                if (l == 0) {
                    ok = 0;
                    break;
                }
                sum += cw * (int)l;
            }
        }
        if (!ok) {
            continue;
        }
        if (sum < best_bits || (sum == best_bits && dh == 4)) {
            best_bits = sum;
            best_dh = dh;
        }
    }
    return best_dh;
}

static int rank_top_dh_sentinels(
    const uint8_t *input_data,
    int input_size,
    int exclude_dh,
    int *out_dhs,
    int max_out
) {
    int whole_dh = find_best_whole_dh_sentinel(input_data, input_size, NULL);
    int hist_whole[256] = {0};
    int hist_lit[256] = {0};
    int lit_count = 0;
    for (int i = 0; i < input_size; i++) {
        hist_whole[input_data[i]]++;
    }

    if (!tls_dp.pair_pos_initialized) {
        memset(tls_dp.last_pair_pos, -1, sizeof(tls_dp.last_pair_pos));
        memset(tls_dp.last_quad_pos, -1, sizeof(tls_dp.last_quad_pos));
        tls_dp.pair_pos_initialized = 1;
    } else {
        for (int i = 0; i + 1 < input_size; i++) {
            tls_dp.last_pair_pos[((int)input_data[i] << 8) | (int)input_data[i + 1]] = -1;
            if (i + 3 < input_size) {
                tls_dp.last_quad_pos[hash_quad_bytes(&input_data[i])] = -1;
            }
        }
    }
    int *last_pair_pos = tls_dp.last_pair_pos;
    int *last_quad_pos = tls_dp.last_quad_pos;
    int pos = 0;
    while (pos < input_size) {
        int best_ml = 0;
        int best_off = 0;
        if (pos + 3 < input_size) {
            int qh = (int)hash_quad_bytes(&input_data[pos]);
            int qprev = last_quad_pos[qh];
            if (qprev >= 0 && (pos - qprev) <= ZX0_MAX_OFFSET) {
                int qml = 0;
                while (pos + qml < input_size && input_data[qprev + qml] == input_data[pos + qml]) {
                    qml++;
                }
                if (qml > best_ml) {
                    best_ml = qml;
                    best_off = pos - qprev;
                }
            }
        }
        if (pos + 1 < input_size) {
            int h = ((int)input_data[pos] << 8) | (int)input_data[pos + 1];
            int prev = last_pair_pos[h];
            if (prev >= 0 && (pos - prev) <= ZX0_MAX_OFFSET) {
                int pml = 0;
                while (pos + pml < input_size && input_data[prev + pml] == input_data[pos + pml]) {
                    pml++;
                }
                if (pml > best_ml || (pml == best_ml && (pos - prev) < best_off)) {
                    best_ml = pml;
                    best_off = pos - prev;
                }
            }
        }
        if (pos > 0 && input_data[pos] == input_data[pos - 1]) {
            int rle_ml = 0;
            while (pos + rle_ml < input_size && input_data[pos + rle_ml] == input_data[pos - 1]) {
                rle_ml++;
            }
            if (rle_ml > best_ml || (rle_ml == best_ml && best_off > 1)) {
                best_ml = rle_ml;
                best_off = 1;
            }
        }
        if (best_ml >= 3 || (best_ml == 2 && best_off <= 128)) {
            for (int k = 0; k < best_ml; k++) {
                if (pos + k + 1 < input_size) {
                    last_pair_pos[((int)input_data[pos + k] << 8) | (int)input_data[pos + k + 1]] = pos + k;
                }
                if (pos + k + 3 < input_size) {
                    last_quad_pos[hash_quad_bytes(&input_data[pos + k])] = pos + k;
                }
            }
            pos += best_ml;
        } else {
            hist_lit[input_data[pos]]++;
            lit_count++;
            if (pos + 1 < input_size) {
                last_pair_pos[((int)input_data[pos] << 8) | (int)input_data[pos + 1]] = pos;
            }
            if (pos + 3 < input_size) {
                last_quad_pos[hash_quad_bytes(&input_data[pos])] = pos;
            }
            pos++;
        }
    }

    int cand_dh[128];
    int cand_score_whole[128];
    int cand_dh_lit[128];
    int cand_score_lit[128];
    int n_cand = 0;
    DhSpec spec;
    for (int dh = 3; dh <= 127; dh++) {
        if (dh == exclude_dh) {
            continue;
        }
        build_dh_spec(dh, &spec);
        int sum_w = 0;
        int sum_l = 0;
        int ok = 1;
        for (int b = 0; b < 256; b++) {
            int cw = hist_whole[b];
            if (cw > 0) {
                if (spec.bit_len[b] == 0) {
                    ok = 0;
                    break;
                }
                sum_w += cw * (int)spec.bit_len[b];
                sum_l += (lit_count >= 4 ? hist_lit[b] : cw) * (int)spec.bit_len[b];
            }
        }
        if (ok) {
            
            cand_dh[n_cand] = dh;
            cand_score_whole[n_cand] = sum_w;
            cand_dh_lit[n_cand] = dh;
            cand_score_lit[n_cand] = sum_l;
            n_cand++;
        }
    }

    /* Sort candidates by ascending literal-estimate bit cost and ascending whole-buffer bit cost */
    for (int i = 0; i < n_cand; i++) {
        for (int j = i + 1; j < n_cand; j++) {
            if (cand_score_whole[j] < cand_score_whole[i]) {
                int ts = cand_score_whole[i]; cand_score_whole[i] = cand_score_whole[j]; cand_score_whole[j] = ts;
                int td = cand_dh[i]; cand_dh[i] = cand_dh[j]; cand_dh[j] = td;
            }
            if (cand_score_lit[j] < cand_score_lit[i]) {
                int ts = cand_score_lit[i]; cand_score_lit[i] = cand_score_lit[j]; cand_score_lit[j] = ts;
                int td = cand_dh_lit[i]; cand_dh_lit[i] = cand_dh_lit[j]; cand_dh_lit[j] = td;
            }
        }
    }

    int count = 0;
    for (int i = 0; i < n_cand && i < 6 && count < max_out; i++) {
        int ldh = cand_dh_lit[i];
        int dup = 0;
        for (int k = 0; k < count; k++) {
            if (out_dhs[k] == ldh) { dup = 1; break; }
        }
        if (!dup) out_dhs[count++] = ldh;
    }
    for (int i = 0; i < n_cand && i < 5 && count < max_out; i++) {
        int wdh = cand_dh[i];
        int dup = 0;
        for (int k = 0; k < count; k++) {
            if (out_dhs[k] == wdh) { dup = 1; break; }
        }
        if (!dup) out_dhs[count++] = wdh;
    }

    const int seeds[10] = {whole_dh, 3, 4, 8, 9, 10, 11, 16, 32, 64};
    for (int si = 0; si < 10 && count < max_out; si++) {
        int sdh = seeds[si];
        if (sdh == exclude_dh || sdh < 3 || sdh > 127) {
            continue;
        }
        int valid = 0;
        for (int i = 0; i < n_cand; i++) {
            if (cand_dh[i] == sdh) {
                valid = 1;
                break;
            }
        }
        if (!valid) {
            continue;
        }
        int dup = 0;
        for (int k = 0; k < count; k++) {
            if (out_dhs[k] == sdh) {
                dup = 1;
                break;
            }
        }
        if (!dup) {
            out_dhs[count++] = sdh;
        }
    }
    for (int i = 0; i < n_cand && count < max_out; i++) {
        int cdh = cand_dh_lit[i];
        int dup = 0;
        for (int k = 0; k < count; k++) {
            if (out_dhs[k] == cdh) {
                dup = 1;
                break;
            }
        }
        if (!dup) {
            out_dhs[count++] = cdh;
        }
    }
    return count;
}

/*
 * Exact DP-Plan Literal Histogram Ranking for ULTRA_128B:
 * Extracts the true literal byte distribution from an actual DP ParsePlan and ranks
 * DH sentinels by their exact literal bit cost on those tokens.
 */
static int rank_top_dh_sentinels_from_plan(
    const uint8_t *input_data,
    int input_size,
    const ParsePlan *plan,
    int *out_dhs,
    int max_out
) {
    int hist_whole[256] = {0};
    int hist_plan_lit[256] = {0};
    int plan_lit_count = 0;
    for (int i = 0; i < input_size; i++) {
        hist_whole[input_data[i]]++;
    }
    if (plan != NULL && plan->tokens != NULL) {
        int pos = 0;
        for (int t = 0; t < plan->token_count; t++) {
            int len = plan->tokens[t].length;
            int off = plan->tokens[t].offset;
            if (off == 0) {
                for (int k = 0; k < len && pos + k < input_size; k++) {
                    hist_plan_lit[input_data[pos + k]]++;
                    plan_lit_count++;
                }
            }
            pos += len;
        }
    }
    if (plan_lit_count == 0) {
        return rank_top_dh_sentinels(input_data, input_size, -1, out_dhs, max_out);
    }

    int cand_dh[128];
    int cand_score[128];
    int n_cand = 0;
    DhSpec spec;
    for (int dh = 3; dh <= 127; dh++) {
        build_dh_spec(dh, &spec);
        int sum_l = 0;
        int ok = 1;
        for (int b = 0; b < 256; b++) {
            if (hist_whole[b] > 0 && spec.bit_len[b] == 0) {
                ok = 0;
                break;
            }
            if (hist_plan_lit[b] > 0) {
                sum_l += hist_plan_lit[b] * (int)spec.bit_len[b];
            }
        }
        if (ok) {
            cand_dh[n_cand] = dh;
            cand_score[n_cand] = sum_l;
            n_cand++;
        }
    }
    for (int i = 0; i < n_cand; i++) {
        for (int j = i + 1; j < n_cand; j++) {
            if (cand_score[j] < cand_score[i]) {
                int ts = cand_score[i]; cand_score[i] = cand_score[j]; cand_score[j] = ts;
                int td = cand_dh[i]; cand_dh[i] = cand_dh[j]; cand_dh[j] = td;
            }
        }
    }
    int count = 0;
    for (int i = 0; i < n_cand && count < max_out; i++) {
        out_dhs[count++] = cand_dh[i];
    }
    return count;
}

/*
 * Exact Post-DP Token-Plan DH Sentinel Optimizer for ULTRA_128B:
 * Given a finalized ParsePlan, evaluates all 125 non-dominated DH sentinels (3..127)
 * that cover every literal byte actually emitted by the plan (even if matched bytes
 * outside the literal set lie outside the DH table!), and selects the DH sentinel
 * that minimizes the exact literal bit count of the plan.
 */
static void optimize_plan_dh_sentinel(
    const uint8_t *buf,
    int input_size,
    ParsePlan *plan,
    int *dh_sentinel_io
) {
    if (plan == NULL || plan->tokens == NULL || plan->token_count <= 0) {
        return;
    }
    int hist_plan_lit[256] = {0};
    int plan_lit_count = 0;
    int pos = 0;
    for (int t = 0; t < plan->token_count; t++) {
        int len = plan->tokens[t].length;
        int off = plan->tokens[t].offset;
        if (off == 0) {
            for (int k = 0; k < len && pos + k < input_size; k++) {
                hist_plan_lit[buf[pos + k]]++;
                plan_lit_count++;
            }
        }
        pos += len;
    }
    if (plan_lit_count <= 0) {
        return;
    }

    int cur_dh = (*dh_sentinel_io >= 1 && *dh_sentinel_io <= 127) ? *dh_sentinel_io : 3;
    DhSpec cur_spec;
    build_dh_spec(cur_dh, &cur_spec);
    int cur_lit_bits = 0;
    for (int b = 0; b < 256; b++) {
        if (hist_plan_lit[b] > 0) {
            cur_lit_bits += hist_plan_lit[b] * (cur_spec.bit_len[b] ? (int)cur_spec.bit_len[b] : 8);
        }
    }

    int best_dh = cur_dh;
    int best_lit_bits = cur_lit_bits;
    DhSpec spec;
    for (int dh = 3; dh <= 127; dh++) {
        build_dh_spec(dh, &spec);
        int sum_l = 0;
        int ok = 1;
        for (int b = 0; b < 256; b++) {
            int c = hist_plan_lit[b];
            if (c > 0) {
                if (spec.bit_len[b] == 0) {
                    ok = 0;
                    break;
                }
                sum_l += c * (int)spec.bit_len[b];
            }
        }
        if (ok && sum_l < best_lit_bits) {
            best_lit_bits = sum_l;
            best_dh = dh;
        }
    }
    if (best_lit_bits < cur_lit_bits) {
        plan->payload_bits -= (cur_lit_bits - best_lit_bits);
        *dh_sentinel_io = best_dh;
    }
}

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

static inline int range_min_best_any(
    const DpWorkspace *ws,
    const int *best_any_cost,
    int lo_sp,
    int hi_sp,
    int *out_sp
) {
    int min_c = COST_INFINITY;
    int min_p = lo_sp;
    int sp = lo_sp;
    while (sp <= hi_sp && (sp & 63) != 0) {
        int c = best_any_cost[sp];
        if (c < min_c) {
            min_c = c;
            min_p = sp;
        }
        sp++;
    }
    while (sp + 63 <= hi_sp && (sp & 1023) != 0) {
        int b64 = sp >> 6;
        int c = ws->blk64_cost[b64];
        if (c < min_c) {
            min_c = c;
            min_p = ws->blk64_pos[b64];
        }
        sp += 64;
    }
    while (sp + 1023 <= hi_sp) {
        int b1024 = sp >> 10;
        int c = ws->blk1024_cost[b1024];
        if (c < min_c) {
            min_c = c;
            min_p = ws->blk1024_pos[b1024];
        }
        sp += 1024;
    }
    while (sp + 63 <= hi_sp) {
        int b64 = sp >> 6;
        int c = ws->blk64_cost[b64];
        if (c < min_c) {
            min_c = c;
            min_p = ws->blk64_pos[b64];
        }
        sp += 64;
    }
    while (sp <= hi_sp) {
        int c = best_any_cost[sp];
        if (c < min_c) {
            min_c = c;
            min_p = sp;
        }
        sp++;
    }
    *out_sp = min_p;
    return min_c;
}

static inline int eval_best_new_offset_len(
    const DpWorkspace *ws,
    const int *best_any_cost,
    int *best_len_cost,
    int *best_len_start,
    int *computed_best_len_io,
    int *computed_bucket_idx_io,
    int *bucket_pref_cost,
    int *bucket_pref_start,
    int pos,
    int ml,
    int *out_start_pos
) {
    int computed_best_len = *computed_best_len_io;
    int lim_small = ml < 128 ? ml : 128;
    while (computed_best_len < lim_small) {
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
    *computed_best_len_io = computed_best_len;

    if (ml <= 128) {
        *out_start_pos = best_len_start[ml];
        return best_len_cost[ml];
    }

    int computed_bucket_idx = *computed_bucket_idx_io;
    if (computed_bucket_idx == 6) {
        bucket_pref_cost[6] = best_len_cost[128];
        bucket_pref_start[6] = best_len_start[128];
    }
    int target_b = 7;
    while ((2 << target_b) <= ml && target_b < 15) {
        target_b++;
    }
    while (computed_bucket_idx + 1 < target_b) {
        int b = computed_bucket_idx + 1;
        int lo_len = (1 << b) + 1;
        int hi_len = (2 << b);
        int hi_sp = pos - lo_len;
        int lo_sp = pos - hi_len;
        if (lo_sp < 1) {
            lo_sp = 1;
        }
        int pc = bucket_pref_cost[b - 1];
        int ps = bucket_pref_start[b - 1];
        if (lo_sp <= hi_sp) {
            int b_sp = lo_sp;
            int b_min = range_min_best_any(ws, best_any_cost, lo_sp, hi_sp, &b_sp);
            if (b_min < COST_INFINITY) {
                int cand = b_min + (2 * b + 1);
                if (cand <= pc) {
                    pc = cand;
                    ps = b_sp;
                }
            }
        }
        bucket_pref_cost[b] = pc;
        bucket_pref_start[b] = ps;
        computed_bucket_idx = b;
    }
    *computed_bucket_idx_io = computed_bucket_idx;

    int best_c = bucket_pref_cost[target_b - 1];
    int best_s = bucket_pref_start[target_b - 1];
    int lo_len = (1 << target_b) + 1;
    int hi_len = ml;
    if (lo_len <= hi_len) {
        int hi_sp = pos - lo_len;
        int lo_sp = pos - hi_len;
        if (lo_sp < 1) {
            lo_sp = 1;
        }
        if (lo_sp <= hi_sp) {
            int b_sp = lo_sp;
            int b_min = range_min_best_any(ws, best_any_cost, lo_sp, hi_sp, &b_sp);
            if (b_min < COST_INFINITY) {
                int cand = b_min + (2 * target_b + 1);
                if (cand <= best_c) {
                    best_c = cand;
                    best_s = b_sp;
                }
            }
        }
    }
    *out_start_pos = best_s;
    return best_c;
}

/*
 * Exact Active-Offset Multi-State Automaton DP Solver with Zero-Loss Dominated-Literal
 * Pruning, Global O(1) Literal Frontier (lit_frontiers[0]), Decoupled 2-Pass Active-Offset
 * Traversal (Pass 1: cur_matches == 1, Pass 2: cur_matches == 0 && next_pos_matches == 1),
 * Exact Cross-Run Match-Length Formula, Run-Boundary Jumping, Exact Mathematical Dominance Pruning,
 * Thread-Local Zero-Allocation Workspace, and Topological Mark-and-Compact GC.
 */

static inline int compute_exact_ml_run_eq(
    const uint8_t *input_data,
    const int *run_start,
    const int *last_match_pos,
    const int *match_lengths,
    int pos,
    int off,
    int L0,
    int L1,
    int run0
) {
    if (last_match_pos[off] >= L0) {
        return match_lengths[off] + (pos - last_match_pos[off]);
    }
    int cur_L0 = L0;
    int cur_L1 = L1;
    int acc = run0;
    while (cur_L0 > 0 && cur_L1 > 0 && input_data[cur_L0 - 1] == input_data[cur_L1 - 1]) {
        int prev_L0 = run_start[cur_L0 - 1];
        int prev_L1 = run_start[cur_L1 - 1];
        int r0 = cur_L0 - prev_L0;
        int r1 = cur_L1 - prev_L1;
        if (r0 != r1) {
            acc += (r0 < r1) ? r0 : r1;
            return acc;
        }
        if (last_match_pos[off] > prev_L0) {
            return match_lengths[off] + (pos - last_match_pos[off]);
        }
        acc += r0;
        cur_L0 = prev_L0;
        cur_L1 = prev_L1;
    }
    return acc;
}

static __attribute__((noinline)) void eval_1b_swar_tail(
    DpWorkspace *ws,
    const uint8_t *input_data,
    int input_size,
    int pos,
    int off_start,
    int active_max_off,
    uint8_t cur_byte,
    uint8_t prv_byte,
    uint8_t nxt_byte,
    int max_any_raw_1b,
    int prev_lit_bits,
    int literal_pos_bits,
    int cur_pad_1b,
    int f_pad,
    int need_tokens,
    int *best_any_cost,
    int *best_any_node,
    int *best_mat_cost_at_pos,
    int *best_mat_node_at_pos
) {
    OffsetState *os = ws->off_state;
    MonotonicFrontier *lit_frontiers = ws->lit_frontiers;
    int *lit_node0 = ws->lit_node0;
    int *match_lengths = ws->match_lengths;
    int *last_match_pos = ws->last_match_pos;
    NodePool *pool = &ws->pool;
    uint64_t mul_cur = 0x0101010101010101ULL * (uint64_t)cur_byte;
    uint64_t mul_prv = 0x0101010101010101ULL * (uint64_t)prv_byte;
    uint64_t m7f = 0x7F7F7F7F7F7F7F7FULL;
    int start_b64 = off_start >> 6;
    int max_b64 = active_max_off >> 6;
    int max_b8_all = active_max_off >> 3;
    for (int b64 = start_b64; b64 <= max_b64; b64++) {
        if (ws->lit_blk64_min[b64] > max_any_raw_1b) {
            continue;
        }
        uint64_t act64 = ws->lit_act64[b64];
        if (!act64) continue;
        int b8_lo = (b64 == start_b64) ? (off_start >> 3) : (b64 << 3);
        int b8_hi = (b64 << 3) | 7;
        if (b8_hi > max_b8_all) b8_hi = max_b8_all;
        for (int b8 = b8_lo; b8 <= b8_hi; b8++) {
            if (ws->lit_blk8_min[b8] > max_any_raw_1b) continue;
            int idx_lo = (pos - 1) - ((b8 << 3) | 7);
            if (idx_lo >= 1 && b8 > 0) {
                uint64_t w_cur, w_prv;
                memcpy(&w_cur, input_data + idx_lo, 8);
                memcpy(&w_prv, input_data + idx_lo - 1, 8);
                uint64_t x_cur = w_cur ^ mul_cur;
                uint64_t z_cur = ~((((x_cur & m7f) + m7f) | x_cur) | m7f);
                if (!z_cur) continue;
                uint64_t x_prv = w_prv ^ mul_prv;
                uint64_t z_prv = ~((((x_prv & m7f) + m7f) | x_prv) | m7f);
                uint64_t mask = z_cur & ~z_prv;
                if (b8 == (off_start >> 3)) {
                    mask &= (0xFFFFFFFFFFFFFFFFULL >> ((off_start & 7) << 3));
                }
                while (mask) {
                    int lz = __builtin_clzll(mask) >> 3;
                    mask ^= (0x8000000000000000ULL >> (lz << 3));
                    int off = (b8 << 3) + lz;
                    int min_base = os[off].lit_min_base;
                    if (min_base > max_any_raw_1b) continue;
                    int raw_max_p = os[off].lit_max_pos;
                    int max_p = raw_max_p & 0xFFFFF;
                    if (max_p >= pos - 1) continue;
                    int can_enter_lit = (pos < input_size && input_data[pos - off] != nxt_byte);
                    int max_viable_mat = (best_any_cost[pos] < COST_INFINITY)
                                       ? (can_enter_lit ? (best_any_cost[pos] + os[off].new_off_overhead + cur_pad_1b) : (best_any_cost[pos] - 1))
                                       : COST_INFINITY;
                    if (min_base + 4 + prev_lit_bits > max_viable_mat) continue;
                    int min_elias = compute_elias_gamma_bits((pos - 1) - max_p);
                    if (min_base + 1 + min_elias + prev_lit_bits + 2 > max_viable_mat) continue;
                    int lit_src_node = -1;
                    int raw_eval = !(raw_max_p & (1 << 20))
                                 ? (min_base + 1 + min_elias)
                                 : frontier_evaluate_min(&lit_frontiers[off], pos - 1, &lit_src_node);
                    if (raw_eval < COST_INFINITY) {
                        int lit_cost_m = raw_eval + prev_lit_bits;
                        int rep_cost = lit_cost_m + 2;
                        if (rep_cost <= max_viable_mat) {
                            int mat_base_cost = rep_cost - literal_pos_bits;
                            int enters_lit_frontier = can_enter_lit && ((mat_base_cost - min_base) <= f_pad);
                            if (rep_cost < best_any_cost[pos] || enters_lit_frontier) {
                                if (need_tokens && !(raw_max_p & (1 << 20))) lit_src_node = lit_node0[off];
                                int created_lit_node = need_tokens ? node_pool_push(pool, lit_src_node, pos - 1, 0) : 0;
                                int created_mat_node = need_tokens ? node_pool_push(pool, created_lit_node, pos, off) : 0;
                                match_lengths[off] = 1;
                                last_match_pos[off] = pos;
                                if (rep_cost < best_any_cost[pos]) {
                                    best_any_cost[pos] = rep_cost;
                                    best_any_node[pos] = created_mat_node;
                                }
                                if (rep_cost < *best_mat_cost_at_pos) {
                                    *best_mat_cost_at_pos = rep_cost;
                                    *best_mat_node_at_pos = created_mat_node;
                                }
                                if (enters_lit_frontier) {
                                    if (min_base == COST_INFINITY || mat_base_cost <= min_base) {
                                        os[off].lit_min_base = mat_base_cost;
                                        os[off].lit_max_pos = pos;
                                        lit_node0[off] = created_mat_node;
                                        ws->lit_act64[off >> 6] |= (1ULL << (off & 63));
                                        if (mat_base_cost < ws->lit_blk64_min[off >> 6]) { ws->lit_blk64_min[off >> 6] = mat_base_cost; }
                                        if (mat_base_cost < ws->lit_blk8_min[off >> 3]) { ws->lit_blk8_min[off >> 3] = mat_base_cost; }
                                    } else if ((mat_base_cost - min_base) <= 64) {
                                        int raw_mp = os[off].lit_max_pos;
                                        if (!(raw_mp & (1 << 20))) {
                                            int last_delta = pos - raw_mp;
                                            if (mat_base_cost - min_base < compute_elias_gamma_bits(last_delta + 1) - 1) {
                                                lit_frontiers[off].count = 2;
                                                lit_frontiers[off].entries[0].pos = raw_mp;
                                                lit_frontiers[off].entries[0].base_cost = min_base;
                                                lit_frontiers[off].entries[0].node_idx = lit_node0[off];
                                                lit_frontiers[off].entries[1].pos = pos;
                                                lit_frontiers[off].entries[1].base_cost = mat_base_cost;
                                                lit_frontiers[off].entries[1].node_idx = created_mat_node;
                                                os[off].lit_max_pos = pos | (1 << 20);
                                            }
                                        } else {
                                            frontier_push(&lit_frontiers[off], pos, mat_base_cost, created_mat_node);
                                            int new_mb = lit_frontiers[off].entries[0].base_cost;
                                            os[off].lit_min_base = new_mb;
                                            lit_node0[off] = lit_frontiers[off].entries[0].node_idx;
                                            if (new_mb < ws->lit_blk64_min[off >> 6]) { ws->lit_blk64_min[off >> 6] = new_mb; }
                                            if (new_mb < ws->lit_blk8_min[off >> 3]) { ws->lit_blk8_min[off >> 3] = new_mb; }
                                            int fc = lit_frontiers[off].count;
                                            int fp = lit_frontiers[off].entries[fc - 1].pos;
                                            os[off].lit_max_pos = (fc > 1) ? (fp | (1 << 20)) : fp;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                int off_lo = (b8 << 3) > off_start ? (b8 << 3) : off_start;
                int off_hi = ((b8 << 3) | 7) < active_max_off ? ((b8 << 3) | 7) : active_max_off;
                for (int off = off_lo; off <= off_hi; off++) {
                    int min_base = os[off].lit_min_base;
                    if (min_base > max_any_raw_1b) continue;
                    int cur_idx1b = (pos - 1) - off;
                    if (input_data[cur_idx1b] != cur_byte) continue;
                    int raw_max_p = os[off].lit_max_pos;
                    int max_p = raw_max_p & 0xFFFFF;
                    if (max_p >= pos - 1 || (cur_idx1b > 0 && input_data[cur_idx1b - 1] == prv_byte)) continue;
                    int can_enter_lit = (pos < input_size && input_data[pos - off] != nxt_byte);
                    int max_viable_mat = (best_any_cost[pos] < COST_INFINITY)
                                       ? (can_enter_lit ? (best_any_cost[pos] + os[off].new_off_overhead + cur_pad_1b) : (best_any_cost[pos] - 1))
                                       : COST_INFINITY;
                    if (min_base + 4 + prev_lit_bits > max_viable_mat) continue;
                    int min_elias = compute_elias_gamma_bits((pos - 1) - max_p);
                    if (min_base + 1 + min_elias + prev_lit_bits + 2 > max_viable_mat) continue;
                    int lit_src_node = -1;
                    int raw_eval = !(raw_max_p & (1 << 20))
                                 ? (min_base + 1 + min_elias)
                                 : frontier_evaluate_min(&lit_frontiers[off], pos - 1, &lit_src_node);
                    if (raw_eval < COST_INFINITY) {
                        int lit_cost_m = raw_eval + prev_lit_bits;
                        int rep_cost = lit_cost_m + 2;
                        if (rep_cost <= max_viable_mat) {
                            int mat_base_cost = rep_cost - literal_pos_bits;
                            int enters_lit_frontier = can_enter_lit && ((mat_base_cost - min_base) <= f_pad);
                            if (rep_cost < best_any_cost[pos] || enters_lit_frontier) {
                                if (need_tokens && !(raw_max_p & (1 << 20))) lit_src_node = lit_node0[off];
                                int created_lit_node = need_tokens ? node_pool_push(pool, lit_src_node, pos - 1, 0) : 0;
                                int created_mat_node = need_tokens ? node_pool_push(pool, created_lit_node, pos, off) : 0;
                                match_lengths[off] = 1;
                                last_match_pos[off] = pos;
                                if (rep_cost < best_any_cost[pos]) {
                                    best_any_cost[pos] = rep_cost;
                                    best_any_node[pos] = created_mat_node;
                                }
                                if (rep_cost < *best_mat_cost_at_pos) {
                                    *best_mat_cost_at_pos = rep_cost;
                                    *best_mat_node_at_pos = created_mat_node;
                                }
                                if (enters_lit_frontier) {
                                    if (min_base == COST_INFINITY || mat_base_cost <= min_base) {
                                        os[off].lit_min_base = mat_base_cost;
                                        os[off].lit_max_pos = pos;
                                        lit_node0[off] = created_mat_node;
                                        ws->lit_act64[off >> 6] |= (1ULL << (off & 63));
                                        if (mat_base_cost < ws->lit_blk64_min[off >> 6]) { ws->lit_blk64_min[off >> 6] = mat_base_cost; }
                                        if (mat_base_cost < ws->lit_blk8_min[off >> 3]) { ws->lit_blk8_min[off >> 3] = mat_base_cost; }
                                    } else if ((mat_base_cost - min_base) <= 64) {
                                        int raw_mp = os[off].lit_max_pos;
                                        if (!(raw_mp & (1 << 20))) {
                                            int last_delta = pos - raw_mp;
                                            if (mat_base_cost - min_base < compute_elias_gamma_bits(last_delta + 1) - 1) {
                                                lit_frontiers[off].count = 2;
                                                lit_frontiers[off].entries[0].pos = raw_mp;
                                                lit_frontiers[off].entries[0].base_cost = min_base;
                                                lit_frontiers[off].entries[0].node_idx = lit_node0[off];
                                                lit_frontiers[off].entries[1].pos = pos;
                                                lit_frontiers[off].entries[1].base_cost = mat_base_cost;
                                                lit_frontiers[off].entries[1].node_idx = created_mat_node;
                                                os[off].lit_max_pos = pos | (1 << 20);
                                            }
                                        } else {
                                            frontier_push(&lit_frontiers[off], pos, mat_base_cost, created_mat_node);
                                            int new_mb = lit_frontiers[off].entries[0].base_cost;
                                            os[off].lit_min_base = new_mb;
                                            lit_node0[off] = lit_frontiers[off].entries[0].node_idx;
                                            if (new_mb < ws->lit_blk64_min[off >> 6]) { ws->lit_blk64_min[off >> 6] = new_mb; }
                                            if (new_mb < ws->lit_blk8_min[off >> 3]) { ws->lit_blk8_min[off >> 3] = new_mb; }
                                            int fc = lit_frontiers[off].count;
                                            int fp = lit_frontiers[off].entries[fc - 1].pos;
                                            os[off].lit_max_pos = (fc > 1) ? (fp | (1 << 20)) : fp;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

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
    int req_bh_ff = (required_final_offset <= -300);
    int raw_neg = req_bh_ff ? (required_final_offset + 300) : required_final_offset;
    int dh_mode = (raw_neg < 0)
                ? ((-raw_neg >= 1 && -raw_neg <= 127)
                   ? -raw_neg
                   : find_best_dh_sentinel(input_data, input_size, wildcard_mask))
                : (req_bh_ff ? find_best_dh_sentinel(input_data, input_size, wildcard_mask) : 0);
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

    DpWorkspace *ws = &tls_dp;
    dp_workspace_ensure(ws, input_size, max_window);

    int *lit_cum_bits   = ws->lit_cum_bits;
    int *prev_same_byte = ws->prev_same_byte;
    int *prev_same_pair = ws->prev_same_pair;
    int *run_start      = ws->run_start;
    int *best_any_cost  = ws->best_any_cost;
    int *best_any_node  = ws->best_any_node;
    int *best_len_cost  = ws->best_len_cost;
    int *best_len_start = ws->best_len_start;
    MonotonicFrontier *lit_frontiers = ws->lit_frontiers;
    MonotonicFrontier *rep_frontiers = ws->rep_frontiers;
    int *lit_min_base     = ws->lit_min_base;
    int *lit_max_pos      = ws->lit_max_pos;
    int *lit_node0        = ws->lit_node0;
    int *rep_pos0         = ws->rep_pos0;
    int *rep_cost0        = ws->rep_cost0;
    int *rep_node0        = ws->rep_node0;
    int *new_off_overhead = ws->new_off_overhead;
    int *match_lengths    = ws->match_lengths;
    int *last_match_pos   = ws->last_match_pos;
    int *rep_valid_pos    = ws->rep_valid_pos; (void)lit_min_base; (void)lit_max_pos; (void)new_off_overhead; (void)rep_valid_pos;
    OffsetState *os = ws->off_state;
    int *final_lit_cost   = ws->final_lit_cost;
    int *final_lit_node   = ws->final_lit_node;
    int *final_mat_cost   = ws->final_mat_cost;
    int *final_mat_node   = ws->final_mat_node;
    NodePool *pool        = &ws->pool;

    lit_cum_bits[0] = 0;
    if (dh_mode > 0) {
        DhSpec spec;
        build_dh_spec(dh_mode, &spec);
        for (int i = 0; i < input_size; i++) {
            int b = (wildcard_mask != NULL && wildcard_mask[i] != 0)
                  ? (int)spec.min_len
                  : (spec.bit_len[input_data[i]] ? (int)spec.bit_len[input_data[i]] : 8);
            lit_cum_bits[i + 1] = lit_cum_bits[i] + b;
        }
    } else {
        for (int i = 0; i < input_size; i++) {
            lit_cum_bits[i + 1] = (i + 1) * ZX0_LITERAL_BITS_PER_BYTE;
        }
    }

    if (wildcard_mask == NULL) {
        int head_byte[256];
        memset(head_byte, -1, sizeof(head_byte));
        for (int i = 0; i < input_size; i++) {
            uint8_t b = input_data[i];
            prev_same_byte[i] = head_byte[b];
            head_byte[b] = i;
            run_start[i] = (i > 0 && input_data[i - 1] == b) ? run_start[i - 1] : i;
        }
        int *head_pair = ws->last_pair_pos;
        if (!ws->pair_pos_initialized) {
            memset(head_pair, -1, sizeof(ws->last_pair_pos));
            memset(ws->last_quad_pos, -1, sizeof(ws->last_quad_pos));
            ws->pair_pos_initialized = 1;
        } else {
            for (int i = 1; i < input_size; i++) {
                head_pair[((int)input_data[i - 1] << 8) | (int)input_data[i]] = -1;
            }
        }
        prev_same_pair[0] = -1;
        for (int i = 1; i < input_size; i++) {
            int p = ((int)input_data[i - 1] << 8) | (int)input_data[i];
            prev_same_pair[i] = head_pair[p];
            head_pair[p] = i;
        }
    }

    os[0].new_off_overhead = 0;
    lit_frontiers[0].count = 0;
    rep_frontiers[0].count = 0;
    if (wildcard_mask == NULL) {
        for (int off = 1; off <= max_window; off++) {
            os[off].new_off_overhead = compute_new_offset_overhead_bits(off, offset_lsb_bits);
        }
        for (int off = 0; off <= max_window; off++) {
            os[off].lit_min_base = COST_INFINITY;
            os[off].lit_max_pos = 0;
            match_lengths[off] = 0;
            last_match_pos[off] = 0;
            os[off].rep_valid_pos = -1;
        }
        if (required_final_offset > 0 || req_bh_ff) {
            for (int off = 0; off <= max_window; off++) {
                final_lit_cost[off] = COST_INFINITY;
                final_lit_node[off] = -1;
                final_mat_cost[off] = COST_INFINITY;
                final_mat_node[off] = -1;
            }
        }
    } else {
        for (int off = 0; off <= max_window; off++) {
            if (off >= 1) {
                os[off].new_off_overhead = compute_new_offset_overhead_bits(off, offset_lsb_bits);
            }
            lit_frontiers[off].count = 0;
            rep_frontiers[off].count = 0;
            os[off].lit_min_base = COST_INFINITY;
            os[off].lit_max_pos = 0;
            match_lengths[off] = 0;
            last_match_pos[off] = 0;
            os[off].rep_valid_pos = -1;
            final_lit_cost[off] = COST_INFINITY;
            final_lit_node[off] = -1;
            final_mat_cost[off] = COST_INFINITY;
            final_mat_node[off] = -1;
        }
    }

    int num_blk64 = (max_window >> 6) + 2;
    for (int b = 0; b < num_blk64 && b < 1024; b++) {
        ws->lit_blk64_min[b] = COST_INFINITY;
        ws->lit_act64[b] = 0;
    }
    int deep_init = (tls_deep_mode >= 0) ? tls_deep_mode : g_deep_mode;
    if (deep_init == 1 || deep_init == 3) {
        int num_blk8 = (max_window >> 3) + 2;
        for (int b8 = 0; b8 < num_blk8 && b8 < 4096; b8++) {
            ws->lit_blk8_min[b8] = COST_INFINITY;
        }
    }

    int need_tokens = (plan_out != NULL);
    int gc_threshold = NODE_POOL_GC_THRESHOLD;
    
    /* Seed initial state at pos = 0 with base_cost = -1 so first literal block has 0 control bits */
    if (dh_mode == 0) {
        lit_frontiers[ZX0_INITIAL_OFFSET].count = 0;
        frontier_push(&lit_frontiers[ZX0_INITIAL_OFFSET], 0, -1, -1);
        os[ZX0_INITIAL_OFFSET].lit_min_base = -1;
        os[ZX0_INITIAL_OFFSET].lit_max_pos = 0;
        lit_node0[ZX0_INITIAL_OFFSET] = -1;
    }
    frontier_push(&lit_frontiers[0], 0, -1, -1);
    os[0].lit_min_base = -1;
    os[0].lit_max_pos = 0;
    ws->lit_blk64_min[0] = -1;
    ws->lit_act64[0] = 1ULL;
    ws->lit_blk8_min[0] = -1;
    best_any_cost[0] = COST_INFINITY;
    best_any_node[0] = -1;
    ws->blk64_cost[0] = COST_INFINITY;
    ws->blk64_pos[0] = 0;
    ws->blk1024_cost[0] = COST_INFINITY;
    ws->blk1024_pos[0] = 0;

    for (int pos = 1; pos <= input_size; pos++) {
        if (__builtin_expect(need_tokens && pool->count >= gc_threshold, 0)) {
            if (wildcard_mask == NULL) {
                ws->lit_min_base[0] = os[0].lit_min_base;
                ws->rep_valid_pos[0] = -1;
                for (int o = 1; o <= max_window; o++) {
                    ws->lit_min_base[o] = os[o].lit_min_base;
                    ws->rep_valid_pos[o] = os[o].rep_valid_pos;
                    if (os[o].lit_min_base < COST_INFINITY && !(os[o].lit_max_pos & (1 << 20))) {
                        lit_frontiers[o].count = 1;
                        lit_frontiers[o].entries[0].pos = os[o].lit_max_pos;
                        lit_frontiers[o].entries[0].base_cost = os[o].lit_min_base;
                        lit_frontiers[o].entries[0].node_idx = lit_node0[o];
                    }
                    if (os[o].rep_valid_pos >= 0 && (os[o].rep_valid_pos & (1 << 20))) {
                        rep_frontiers[o].count = 1;
                        rep_frontiers[o].entries[0].pos = rep_pos0[o];
                        rep_frontiers[o].entries[0].base_cost = rep_cost0[o];
                        rep_frontiers[o].entries[0].node_idx = rep_node0[o];
                    } else {
                        rep_frontiers[o].count = 0;
                    }
                }
            } else {
                for (int o = 0; o <= max_window; o++) {
                    ws->lit_min_base[o] = os[o].lit_min_base;
                    ws->rep_valid_pos[o] = os[o].rep_valid_pos;
                }
            }
            node_pool_compact(
                pool, best_any_node, pos - 1, lit_frontiers, lit_min_base, rep_frontiers, rep_valid_pos, max_window
            );
            if (wildcard_mask == NULL) {
                for (int o = 1; o <= max_window; o++) {
                    if (os[o].lit_min_base < COST_INFINITY) {
                        lit_node0[o] = lit_frontiers[o].entries[0].node_idx;
                    }
                    if (os[o].rep_valid_pos >= 0 && (os[o].rep_valid_pos & (1 << 20))) {
                        rep_node0[o] = rep_frontiers[o].entries[0].node_idx;
                    }
                }
            }
            if (pool->count * 2 > gc_threshold) {
                gc_threshold = pool->count * 2;
            }
        }

        {
            int sp = pos - 1;
            int b64 = sp >> 6;
            if ((sp & 63) == 0 || best_any_cost[sp] < ws->blk64_cost[b64]) {
                ws->blk64_cost[b64] = best_any_cost[sp];
                ws->blk64_pos[b64] = sp;
            }
            int b1024 = sp >> 10;
            if ((sp & 1023) == 0 || best_any_cost[sp] < ws->blk1024_cost[b1024]) {
                ws->blk1024_cost[b1024] = best_any_cost[sp];
                ws->blk1024_pos[b1024] = sp;
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
        int computed_bucket_idx = 6;
        int bucket_pref_cost[16];
        int bucket_pref_start[16];
        int literal_pos_bits = lit_cum_bits[pos];

        /* O(1) Global Literal Frontier evaluation covers best_any_cost[pos] across all offsets */
        int glob_src_node = -1;
        int glob_raw = frontier_evaluate_min(&lit_frontiers[0], pos, &glob_src_node);
        if (glob_raw < COST_INFINITY) {
            best_any_cost[pos] = glob_raw + literal_pos_bits;
            best_any_node[pos] = need_tokens ? node_pool_push(pool, glob_src_node, pos, 0) : 0;
        }

        int best_mat_cost_at_pos = COST_INFINITY;
        int best_mat_node_at_pos = -1;

        if (wildcard_mask == NULL && (pos < input_size || (required_final_offset == 0 && !req_bh_ff))) {
            int min_idx1 = (pos - 1) - active_max_off;
            int L0 = run_start[pos - 1];
            int run0 = (pos - 1) - L0 + 1;
            uint8_t cur_byte = input_data[pos - 1];
            uint8_t nxt_byte = (pos < input_size) ? input_data[pos] : 0;
            uint8_t prv_byte = (pos >= 2) ? input_data[pos - 2] : 0;
            int cur_continues = (pos < input_size && nxt_byte == cur_byte);

            int deep = (tls_deep_mode >= 0) ? tls_deep_mode : g_deep_mode;
            int chain_steps = 0;
            int match_steps = 0;
            int max_seen_ml = 1;
            int max_chain_steps = (deep == 3) ? 32640 : ((deep == 2) ? (input_size <= 8192 ? 128 : 64) : (deep ? (input_size < 8192 ? 32640 : 4096) : (input_size <= 1024 ? 128 : 16)));
            int max_match_steps = (deep == 3) ? 8192 : ((deep == 2) ? (input_size <= 8192 ? 32 : 24) : (deep ? (input_size < 8192 ? 8192 : 1024) : (input_size <= 1024 ? 32 : 8)));
            int is_incomp = (dh_mode == 0 && pos >= 64 && best_any_cost[pos] > pos * 3);
            int mat_pad = (deep == 1) ? ((input_size < 8192 || is_incomp) ? 64 : 12) : (deep ? 12 : 4);
            int f_pad = (deep == 1) ? ((input_size < 8192 || is_incomp) ? 128 : 16) : 16;

            int idx1 = (pos >= 2) ? prev_same_pair[pos - 1] : -1;
            int min_pair_idx1 = min_idx1 > 1 ? min_idx1 : 1;
            int switched_to_L0 = 0;
            int dh_win_limit = 0;
            uint8_t b_m3 = (pos >= 3) ? input_data[pos - 3] : 0;
            int max_mb_m2 = (pos >= 2 && best_any_cost[pos - 2] < COST_INFINITY)
                          ? (best_any_cost[pos - 2] - lit_cum_bits[pos - 2] + os[loop_max_off].new_off_overhead - 3)
                          : COST_INFINITY;

            while (idx1 >= min_pair_idx1) {
                if (++chain_steps > max_chain_steps) {
                    break;
                }
                int cur_idx1, off, L1, run1, ml;
                if (!switched_to_L0) {
                    cur_idx1 = idx1;
                    off = (pos - 1) - cur_idx1;
                    idx1 = prev_same_pair[cur_idx1];

                    if (!(cur_continues && run0 >= 2)) {
                        if (pos < 3 || cur_idx1 < 2 || input_data[cur_idx1 - 2] != b_m3) {
                            ml = 2;
                            if (match_steps >= 1 && off != required_final_offset && !(req_bh_ff && off <= 256)) {
                                int can_lit_m2 = (pos < input_size && input_data[cur_idx1 + 1] != nxt_byte);
                                if (!can_lit_m2 || match_steps >= max_match_steps) {
                                    if (!(ws->lit_act64[off >> 6] & (1ULL << (off & 63))) ||
                                        os[off].lit_min_base > max_mb_m2) {
                                        match_steps++;
                                        continue;
                                    }
                                }
                            }
                        } else {
                            if (match_steps >= 1 && off != required_final_offset && !(req_bh_ff && off <= 256) &&
                                (cur_idx1 < max_seen_ml || input_data[cur_idx1 - max_seen_ml] != input_data[(pos - 1) - max_seen_ml])) {
                                int can_lit_m3 = (pos < input_size && input_data[cur_idx1 + 1] != nxt_byte);
                                if (!can_lit_m3) {
                                    if (!(ws->lit_act64[off >> 6] & (1ULL << (off & 63)))) {
                                        match_steps++;
                                        continue;
                                    }
                                    if (os[off].rep_valid_pos == pos - 1) {
                                        os[off].rep_valid_pos = pos;
                                        match_steps++;
                                        continue;
                                    }
                                } else if (match_steps >= max_match_steps && off > dh_win_limit) {
                                    if (!(ws->lit_act64[off >> 6] & (1ULL << (off & 63)))) {
                                        match_steps++;
                                        continue;
                                    }
                                    if (os[off].rep_valid_pos == pos - 1 && (pos - (os[off].lit_max_pos & 0xFFFFF)) > 2048) {
                                        os[off].rep_valid_pos = pos;
                                        match_steps++;
                                        continue;
                                    }
                                }
                            }
                            uint64_t _w0, _w1, _diff = 0;
                            if (cur_idx1 >= 9) {
                                memcpy(&_w0, input_data + pos - 10, 8);
                                memcpy(&_w1, input_data + cur_idx1 - 9, 8);
                                _diff = _w0 ^ _w1;
                            }
                            if (_diff != 0) {
                                ml = 2 + (__builtin_clzll(_diff) >> 3);
                            } else {
                                L1 = run_start[cur_idx1];
                                run1 = cur_idx1 - L1 + 1;
                                if (run1 < run0) {
                                    ml = run1;
                                } else if (run1 > run0) {
                                    ml = run0;
                                } else {
                                    ml = compute_exact_ml_run_eq(input_data, run_start, last_match_pos, match_lengths, pos, off, L0, L1, run0);
                                }
                            }
                        }
                    } else {
                        L1 = run_start[cur_idx1];
                        run1 = cur_idx1 - L1 + 1;
                        if (L1 == L0) {
                            if (cur_idx1 < pos - 2) {
                                idx1 = (L0 + 1 < input_size) ? prev_same_pair[L0 + 1] : -1;
                            }
                        } else if (run1 > 3 && cur_idx1 > L1 + 1) {
                            int target = (run1 > run0) ? (L1 + run0 - 1) : (L1 + 1);
                            if (cur_idx1 > target) {
                                idx1 = target;
                            }
                        }
                        if (run1 < run0) {
                            ml = run1;
                        } else if (run1 > run0) {
                            ml = run0;
                        } else {
                            ml = compute_exact_ml_run_eq(input_data, run_start, last_match_pos, match_lengths, pos, off, L0, L1, run0);
                        }
                    }
                    int next_max_ml = (ml > max_seen_ml) ? ml : max_seen_ml;
                    if ((input_size >= 8192 || deep != 1 || match_steps >= 256) && run0 >= 2 && next_max_ml >= run0 && off > dh_win_limit &&
                        (required_final_offset <= 0 || off > required_final_offset) &&
                        ((cur_continues && (run0 > 8 || match_steps >= 64)) || match_steps >= max_match_steps)) {
                        if (L0 == 0) break;
                        switched_to_L0 = 1;
                        int next_L1 = prev_same_pair[L0];
                        int max_L1 = cur_idx1 - (run0 - 1);
                        while (next_L1 >= 1 && next_L1 > max_L1) {
                            next_L1 = prev_same_pair[next_L1];
                        }
                        idx1 = next_L1;
                        min_pair_idx1 = (L0 - active_max_off) > 1 ? (L0 - active_max_off) : 1;
                    }
                } else {
                    L1 = idx1;
                    idx1 = prev_same_pair[L1];
                    cur_idx1 = L1 + run0 - 1;
                    if (run_start[cur_idx1] != L1) {
                        continue;
                    }
                    off = L0 - L1;
                    if (off > 256 && off != required_final_offset &&
                        (L0 == 0 || L1 == 0 || input_data[L0 - 1] != input_data[L1 - 1] ||
                         cur_idx1 < max_seen_ml || input_data[cur_idx1 - max_seen_ml] != input_data[(pos - 1) - max_seen_ml])) {
                        int can_lit_l0 = (pos < input_size && input_data[cur_idx1 + 1] != nxt_byte);
                        if (!can_lit_l0) {
                            if (!(ws->lit_act64[off >> 6] & (1ULL << (off & 63)))) {
                                match_steps++;
                                continue;
                            }
                            if ((os[off].rep_valid_pos < 0 || !(os[off].rep_valid_pos & (1 << 20))) &&
                                ((os[off].lit_max_pos & 0xFFFFF) >= L0 || (os[off].rep_valid_pos >= 0 && (os[off].rep_valid_pos & 0xFFFFF) >= L0))) {
                                if (os[off].rep_valid_pos == pos - 1) os[off].rep_valid_pos = pos;
                                match_steps++;
                                continue;
                            }
                        } else if (match_steps >= max_match_steps && !(ws->lit_act64[off >> 6] & (1ULL << (off & 63)))) {
                            match_steps++;
                            continue;
                        }
                    }
                    run1 = run0;
                    ml = compute_exact_ml_run_eq(input_data, run_start, last_match_pos, match_lengths, pos, off, L0, L1, run0);
                }

                match_steps++;
                if (ml > 2 || run0 >= 2) {
                    match_lengths[off] = ml;
                    last_match_pos[off] = pos;
                }
                int can_enter_lit = (pos < input_size && input_data[pos - off] != nxt_byte);

                int is_req_off = (off == required_final_offset) || (req_bh_ff && off <= 256);
                int prev_max_seen_ml = max_seen_ml;
                int min_base = os[off].lit_min_base;
                if (min_base == COST_INFINITY && ml <= prev_max_seen_ml && !is_req_off) {
                    if ((!can_enter_lit && off > 256) || (off > dh_win_limit && (!can_enter_lit || match_steps > max_match_steps))) {
                        continue;
                    }
                }

                int max_viable_mat = (!is_req_off && best_any_cost[pos] < COST_INFINITY)
                                   ? (can_enter_lit ? (best_any_cost[pos] + os[off].new_off_overhead + mat_pad) : (best_any_cost[pos] - 1))
                                   : COST_INFINITY;
                int mat_cost = COST_INFINITY;
                int mat_prev_node = -1;
                int is_rep_active = 0;
                int max_p = 0;

                if (min_base < COST_INFINITY) {
                    int start_m = pos - ml;
                    int raw_max_p = os[off].lit_max_pos;
                    max_p = raw_max_p & 0xFFFFF;
                    int rv_raw = os[off].rep_valid_pos;
                    int rv_pos = rv_raw & 0xFFFFF;
                    if (max_p < start_m && (rv_raw < 0 || rv_pos < start_m)) {
                        int rv_new = pos;
                        int max_raw_m = (!is_req_off && best_any_cost[start_m] < COST_INFINITY)
                                      ? (best_any_cost[start_m] - lit_cum_bits[start_m] + os[off].new_off_overhead - 1)
                                      : COST_INFINITY;
                        if (min_base + 2 <= max_raw_m) {
                            int min_elias = compute_elias_gamma_bits(start_m - max_p);
                            if (min_base + 1 + min_elias <= max_raw_m) {
                                int lit_src_node_m = -1;
                                int raw_eval;
                                if (!(raw_max_p & (1 << 20))) {
                                    raw_eval = min_base + 1 + min_elias;
                                    if (need_tokens && raw_eval <= max_raw_m) {
                                        lit_src_node_m = lit_node0[off];
                                    }
                                } else {
                                    raw_eval = frontier_evaluate_min(&lit_frontiers[off], start_m, &lit_src_node_m);
                                }
                                if (raw_eval <= max_raw_m) {
                                    int lit_cost_m = raw_eval + lit_cum_bits[start_m];
                                    int created_lit_node_m = need_tokens ? node_pool_push(pool, lit_src_node_m, start_m, 0) : 0;
                                    rep_pos0[off] = start_m;
                                    rep_cost0[off] = lit_cost_m;
                                    rep_node0[off] = created_lit_node_m;
                                    rv_new = pos | (1 << 20);
                                }
                            }
                        }
                        os[off].rep_valid_pos = rv_new;
                        rv_raw = rv_new;
                    } else if (rv_raw >= 0 && rv_pos >= start_m) {
                        rv_raw = pos | (rv_raw & (1 << 20));
                        os[off].rep_valid_pos = rv_raw;
                    }

                    if (rv_raw == (pos | (1 << 20))) {
                        int r_bc = rep_cost0[off];
                        if (r_bc + 2 <= max_viable_mat) {
                            int rep_cost = r_bc + 1 + compute_elias_gamma_bits(pos - rep_pos0[off]);
                            if (rep_cost <= max_viable_mat) {
                                mat_cost = rep_cost;
                                mat_prev_node = rep_node0[off];
                                is_rep_active = 1;
                            }
                        }
                    }
                }

                if (ml <= prev_max_seen_ml && !is_rep_active && !is_req_off &&
                    ((!can_enter_lit && off > 256) || (off > dh_win_limit && (!can_enter_lit || (match_steps > max_match_steps && !(deep && min_base < COST_INFINITY && ml >= 3 && (pos - max_p) <= 2048)))))) {
                    continue;
                }
                if (ml > max_seen_ml) {
                    max_seen_ml = ml;
                }

                if (ml >= MIN_MATCH_LENGTH_NEW_OFFSET && (can_enter_lit || is_req_off || ml > prev_max_seen_ml)) {
                    int best_s = 0, best_c;
                    if (ml <= computed_best_len) {
                        best_c = best_len_cost[ml];
                        best_s = best_len_start[ml];
                    } else {
                        best_c = eval_best_new_offset_len(
                            ws, best_any_cost, best_len_cost, best_len_start,
                            &computed_best_len, &computed_bucket_idx, bucket_pref_cost, bucket_pref_start,
                            pos, ml, &best_s
                        );
                    }
                    if (best_c < COST_INFINITY) {
                        int new_off_cost = best_c + os[off].new_off_overhead;
                        if (new_off_cost < mat_cost) {
                            mat_cost = new_off_cost;
                            mat_prev_node = best_any_node[best_s];
                        }
                    }
                }

                if (mat_cost <= max_viable_mat && mat_cost < COST_INFINITY) {
                    int mat_base_cost = mat_cost - literal_pos_bits;
                    int enters_lit_frontier = can_enter_lit &&
                                              (is_req_off || (min_base == COST_INFINITY) || ((mat_base_cost - min_base) <= f_pad));
                    if (mat_cost < best_any_cost[pos] || enters_lit_frontier) {
                        int created_mat_node = need_tokens ? node_pool_push(pool, mat_prev_node, pos, off) : 0;
                        if (mat_cost < best_any_cost[pos]) {
                            best_any_cost[pos] = mat_cost;
                            best_any_node[pos] = created_mat_node;
                        }
                        if (mat_cost < best_mat_cost_at_pos) {
                            best_mat_cost_at_pos = mat_cost;
                            best_mat_node_at_pos = created_mat_node;
                        }
                        if (enters_lit_frontier) {
                            if (min_base == COST_INFINITY || mat_base_cost <= min_base) {
                                os[off].lit_min_base = mat_base_cost;
                                os[off].lit_max_pos = pos;
                                lit_node0[off] = created_mat_node;
                                ws->lit_act64[off >> 6] |= (1ULL << (off & 63));
                                if (mat_base_cost < ws->lit_blk64_min[off >> 6]) {
                                    ws->lit_blk64_min[off >> 6] = mat_base_cost;
                                }
                                if (mat_base_cost < ws->lit_blk8_min[off >> 3]) {
                                    ws->lit_blk8_min[off >> 3] = mat_base_cost;
                                }
                            } else if ((mat_base_cost - min_base) <= 64) {
                                int raw_mp = os[off].lit_max_pos;
                                if (!(raw_mp & (1 << 20))) {
                                    int last_delta = pos - raw_mp;
                                    if (mat_base_cost - min_base < compute_elias_gamma_bits(last_delta + 1) - 1) {
                                        lit_frontiers[off].count = 2;
                                        lit_frontiers[off].entries[0].pos = raw_mp;
                                        lit_frontiers[off].entries[0].base_cost = min_base;
                                        lit_frontiers[off].entries[0].node_idx = lit_node0[off];
                                        lit_frontiers[off].entries[1].pos = pos;
                                        lit_frontiers[off].entries[1].base_cost = mat_base_cost;
                                        lit_frontiers[off].entries[1].node_idx = created_mat_node;
                                        os[off].lit_max_pos = pos | (1 << 20);
                                    }
                                } else {
                                    frontier_push(&lit_frontiers[off], pos, mat_base_cost, created_mat_node);
                                    int new_mb = lit_frontiers[off].entries[0].base_cost;
                                    os[off].lit_min_base = new_mb;
                                    lit_node0[off] = lit_frontiers[off].entries[0].node_idx;
                                    if (new_mb < ws->lit_blk64_min[off >> 6]) {
                                        ws->lit_blk64_min[off >> 6] = new_mb;
                                    }
                                    if (new_mb < ws->lit_blk8_min[off >> 3]) {
                                        ws->lit_blk8_min[off >> 3] = new_mb;
                                    }
                                    int fc = lit_frontiers[off].count;
                                    int fp = lit_frontiers[off].entries[fc - 1].pos;
                                    os[off].lit_max_pos = (fc > 1) ? (fp | (1 << 20)) : fp;
                                }
                            }
                        }
                    }
                }
            }

            /* Exact 1-byte LAST_OFFSET matches (cur_matches == 1, ml == 1, not in prev_same_pair) */
            {
                int idx1b = prev_same_byte[pos - 1];
                int p1b_steps = 0;
                int p1b_chain = 0;
                int max_p1b_steps = (deep == 2) ? (input_size <= 8192 ? 32 : 16) : (deep ? (input_size > 32768 ? 2048 : 32640) : 4);
                int max_p1b_chain = (deep == 2) ? (input_size <= 8192 ? 128 : 64) : (deep ? (input_size < 8192 ? 32640 : 7168) : 8);
                int prev_lit_bits = lit_cum_bits[pos - 1];
                int is_high_ent_1b = (deep == 1) && (input_size < 8192 || is_incomp);
                int cur_pad_1b = is_high_ent_1b ? mat_pad : -2;
                int max_any_raw_1b = (best_any_cost[pos] < COST_INFINITY)
                                   ? (best_any_cost[pos] - prev_lit_bits + os[loop_max_off].new_off_overhead + cur_pad_1b - 4)
                                   : COST_INFINITY;
                int use_swar_tail = ((deep == 1 || deep == 3) && !is_incomp && pos >= 2 && active_max_off >= 1024 && required_final_offset <= 0);
                if (pos >= 2 && prv_byte == cur_byte) {
                    if (!cur_continues || run0 <= 3) {
                    if (cur_continues && (dh_mode == 0 || (cur_byte == 0 && dh_mode != 16)) && max_p1b_chain > 512) max_p1b_chain = 512;
                    int max_p1b_steps_1ba = (input_size >= 8192 && dh_mode == 0) ? 256 : max_p1b_steps;
                    int chain_1ba_lim = use_swar_tail ? 96 : max_p1b_chain;
                    /* When cur_byte == prv_byte, only L2 = run_start[idx1b] has input_data[L2 - 1] != prv_byte */
                    while (idx1b >= min_idx1) {
                        if (++p1b_chain > chain_1ba_lim) break;
                        int L2 = run_start[idx1b];
                        if (L2 >= min_idx1) {
                            int off = (pos - 1) - L2;
                            int is_req_off = (off == required_final_offset);
                            int min_base = os[off].lit_min_base;
                            if (!is_req_off && min_base > max_any_raw_1b) {
                                idx1b = prev_same_byte[L2];
                                continue;
                            }
                            int raw_max_p = os[off].lit_max_pos;
                            int max_p = raw_max_p & 0xFFFFF;
                            if (max_p < pos - 1) {
                                int can_enter_lit = (pos < input_size && input_data[pos - off] != nxt_byte);
                                int max_viable_mat = (!is_req_off && best_any_cost[pos] < COST_INFINITY)
                                                   ? (can_enter_lit ? (best_any_cost[pos] + os[off].new_off_overhead + cur_pad_1b) : (best_any_cost[pos] - 1))
                                                   : COST_INFINITY;
                                if (min_base + 4 + prev_lit_bits <= max_viable_mat) {
                                    int min_elias = compute_elias_gamma_bits((pos - 1) - max_p);
                                    if (min_base + 1 + min_elias + prev_lit_bits + 2 <= max_viable_mat) {
                                        if (++p1b_steps > max_p1b_steps_1ba) break;
                                        int lit_src_node = -1;
                                        int raw_eval = !(raw_max_p & (1 << 20))
                                                     ? (min_base + 1 + min_elias)
                                                     : frontier_evaluate_min(&lit_frontiers[off], pos - 1, &lit_src_node);
                                        if (raw_eval < COST_INFINITY) {
                                            int lit_cost_m = raw_eval + prev_lit_bits;
                                            int rep_cost = lit_cost_m + 2;
                                            if (rep_cost <= max_viable_mat) {
                                                int mat_base_cost = rep_cost - literal_pos_bits;
                                                int enters_lit_frontier = can_enter_lit &&
                                                                          (is_req_off || ((mat_base_cost - min_base) <= f_pad));
                                                if (rep_cost < best_any_cost[pos] || enters_lit_frontier) {
                                                    if (need_tokens && !(raw_max_p & (1 << 20))) {
                                                        lit_src_node = lit_node0[off];
                                                    }
                                                    int created_lit_node = need_tokens ? node_pool_push(pool, lit_src_node, pos - 1, 0) : 0;
                                                    int created_mat_node = need_tokens ? node_pool_push(pool, created_lit_node, pos, off) : 0;
                                                    match_lengths[off] = 1;
                                                    last_match_pos[off] = pos;
                                                    if (rep_cost < best_any_cost[pos]) {
                                                        best_any_cost[pos] = rep_cost;
                                                        best_any_node[pos] = created_mat_node;
                                                    }
                                                    if (rep_cost < best_mat_cost_at_pos) {
                                                        best_mat_cost_at_pos = rep_cost;
                                                        best_mat_node_at_pos = created_mat_node;
                                                    }
                        if (enters_lit_frontier) {
                            if (min_base == COST_INFINITY || mat_base_cost <= min_base) {
                                os[off].lit_min_base = mat_base_cost;
                                os[off].lit_max_pos = pos;
                                lit_node0[off] = created_mat_node;
                                ws->lit_act64[off >> 6] |= (1ULL << (off & 63));
                                if (mat_base_cost < ws->lit_blk64_min[off >> 6]) {
                                    ws->lit_blk64_min[off >> 6] = mat_base_cost;
                                }
                                if (mat_base_cost < ws->lit_blk8_min[off >> 3]) {
                                    ws->lit_blk8_min[off >> 3] = mat_base_cost;
                                }
                            } else if ((mat_base_cost - min_base) <= 64) {
                                int raw_mp = os[off].lit_max_pos;
                                if (!(raw_mp & (1 << 20))) {
                                    int last_delta = pos - raw_mp;
                                    if (mat_base_cost - min_base < compute_elias_gamma_bits(last_delta + 1) - 1) {
                                        lit_frontiers[off].count = 2;
                                        lit_frontiers[off].entries[0].pos = raw_mp;
                                        lit_frontiers[off].entries[0].base_cost = min_base;
                                        lit_frontiers[off].entries[0].node_idx = lit_node0[off];
                                        lit_frontiers[off].entries[1].pos = pos;
                                        lit_frontiers[off].entries[1].base_cost = mat_base_cost;
                                        lit_frontiers[off].entries[1].node_idx = created_mat_node;
                                        os[off].lit_max_pos = pos | (1 << 20);
                                    }
                                } else {
                                    frontier_push(&lit_frontiers[off], pos, mat_base_cost, created_mat_node);
                                    int new_mb = lit_frontiers[off].entries[0].base_cost;
                                    os[off].lit_min_base = new_mb;
                                    lit_node0[off] = lit_frontiers[off].entries[0].node_idx;
                                    if (new_mb < ws->lit_blk64_min[off >> 6]) {
                                        ws->lit_blk64_min[off >> 6] = new_mb;
                                    }
                                    if (new_mb < ws->lit_blk8_min[off >> 3]) {
                                        ws->lit_blk8_min[off >> 3] = new_mb;
                                    }
                                    int fc = lit_frontiers[off].count;
                                    int fp = lit_frontiers[off].entries[fc - 1].pos;
                                    os[off].lit_max_pos = (fc > 1) ? (fp | (1 << 20)) : fp;
                                }
                            }
                        }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        idx1b = prev_same_byte[L2];
                    }
                    if (use_swar_tail && p1b_chain > chain_1ba_lim && idx1b >= min_idx1 && p1b_steps < max_p1b_steps_1ba) {
                        int L2_next = run_start[idx1b];
                        if (L2_next >= min_idx1) {
                            eval_1b_swar_tail(
                                ws, input_data, input_size, pos, (pos - 1) - L2_next, active_max_off,
                                cur_byte, prv_byte, nxt_byte, max_any_raw_1b, prev_lit_bits, literal_pos_bits,
                                cur_pad_1b, f_pad, need_tokens, best_any_cost, best_any_node,
                                &best_mat_cost_at_pos, &best_mat_node_at_pos
                            );
                        }
                    }
                    }
                } else {
                    int chain_1bb_lim = use_swar_tail ? ((cur_byte == 0 || cur_byte == 0x20) ? 96 : 384) : max_p1b_chain;
                    while (idx1b >= min_idx1) {
                        if (++p1b_chain > chain_1bb_lim) break;
                        int cur_idx1b = idx1b;
                        idx1b = prev_same_byte[cur_idx1b];
                        int L2 = run_start[cur_idx1b];
                        if (cur_idx1b > L2) {
                            int scan = cur_idx1b - 1;
                            int scan_can_lit = (pos < input_size && !cur_continues);
                            int lim_s = (best_any_cost[pos] < COST_INFINITY)
                                      ? (scan_can_lit ? max_any_raw_1b : (best_any_cost[pos] - prev_lit_bits - 5))
                                      : COST_INFINITY;
                            int scan_min = (L2 > min_idx1) ? L2 : min_idx1;
                            while (scan >= scan_min) {
                                int off_s = (pos - 1) - scan;
                                int b64_s = off_s >> 6;
                                int blk_end_s = (pos - 1) - ((b64_s << 6) | 63);
                                if (required_final_offset <= 0 || (required_final_offset >> 6) != b64_s) {
                                    if (ws->lit_blk64_min[b64_s] > lim_s) {
                                        scan = blk_end_s - 1;
                                        continue;
                                    }
                                    uint64_t act_m = ws->lit_act64[b64_s] >> (off_s & 63);
                                    if (!act_m) {
                                        scan = blk_end_s - 1;
                                        continue;
                                    }
                                    int tz = __builtin_ctzll(act_m);
                                    scan -= tz;
                                    if (scan < scan_min) break;
                                    off_s += tz;
                                }
                                if (off_s == required_final_offset || (os[off_s].lit_min_base <= lim_s && (os[off_s].lit_max_pos & 0xFFFFF) < pos - 1)) {
                                    break;
                                }
                                scan--;
                            }
                            idx1b = (scan >= scan_min) ? scan : prev_same_byte[L2];
                        }
                        int off = (pos - 1) - cur_idx1b;
                        int is_req_off = (off == required_final_offset);
                        int min_base = os[off].lit_min_base;
                        if (!is_req_off && min_base > max_any_raw_1b) {
                            continue;
                        }
                        int raw_max_p = os[off].lit_max_pos;
                        int max_p = raw_max_p & 0xFFFFF;
                        if (max_p >= pos - 1 || (cur_idx1b > 0 && input_data[cur_idx1b - 1] == prv_byte)) {
                            continue;
                        }
                        int can_enter_lit = (pos < input_size && input_data[pos - off] != nxt_byte);
                        int max_viable_mat = (!is_req_off && best_any_cost[pos] < COST_INFINITY)
                                           ? (can_enter_lit ? (best_any_cost[pos] + os[off].new_off_overhead + cur_pad_1b) : (best_any_cost[pos] - 1))
                                           : COST_INFINITY;
                        if (min_base + 4 + prev_lit_bits > max_viable_mat) {
                            continue;
                        }
                        int min_elias = compute_elias_gamma_bits((pos - 1) - max_p);
                        if (min_base + 1 + min_elias + prev_lit_bits + 2 > max_viable_mat) {
                            continue;
                        }
                        if (++p1b_steps > max_p1b_steps) break;
                        
                        int lit_src_node = -1;
                        int raw_eval = !(raw_max_p & (1 << 20))
                                     ? (min_base + 1 + min_elias)
                                     : frontier_evaluate_min(&lit_frontiers[off], pos - 1, &lit_src_node);
                        if (raw_eval < COST_INFINITY) {
                            int lit_cost_m = raw_eval + prev_lit_bits;
                            int rep_cost = lit_cost_m + 2;
                            if (rep_cost <= max_viable_mat) {
                                int mat_base_cost = rep_cost - literal_pos_bits;
                                int enters_lit_frontier = can_enter_lit &&
                                                          (is_req_off || ((mat_base_cost - min_base) <= f_pad));
                                if (rep_cost < best_any_cost[pos] || enters_lit_frontier) {
                                    if (need_tokens && !(raw_max_p & (1 << 20))) {
                                        lit_src_node = lit_node0[off];
                                    }
                                    int created_lit_node = need_tokens ? node_pool_push(pool, lit_src_node, pos - 1, 0) : 0;
                                    int created_mat_node = need_tokens ? node_pool_push(pool, created_lit_node, pos, off) : 0;
                                    match_lengths[off] = 1;
                                    last_match_pos[off] = pos;
                                    if (rep_cost < best_any_cost[pos]) {
                                        best_any_cost[pos] = rep_cost;
                                        best_any_node[pos] = created_mat_node;
                                    }
                                    if (rep_cost < best_mat_cost_at_pos) {
                                        best_mat_cost_at_pos = rep_cost;
                                        best_mat_node_at_pos = created_mat_node;
                                    }
                        if (enters_lit_frontier) {
                            if (min_base == COST_INFINITY || mat_base_cost <= min_base) {
                                os[off].lit_min_base = mat_base_cost;
                                os[off].lit_max_pos = pos;
                                lit_node0[off] = created_mat_node;
                                ws->lit_act64[off >> 6] |= (1ULL << (off & 63));
                                if (mat_base_cost < ws->lit_blk64_min[off >> 6]) {
                                    ws->lit_blk64_min[off >> 6] = mat_base_cost;
                                }
                                if (mat_base_cost < ws->lit_blk8_min[off >> 3]) {
                                    ws->lit_blk8_min[off >> 3] = mat_base_cost;
                                }
                            } else if ((mat_base_cost - min_base) <= 64) {
                                int raw_mp = os[off].lit_max_pos;
                                if (!(raw_mp & (1 << 20))) {
                                    int last_delta = pos - raw_mp;
                                    if (mat_base_cost - min_base < compute_elias_gamma_bits(last_delta + 1) - 1) {
                                        lit_frontiers[off].count = 2;
                                        lit_frontiers[off].entries[0].pos = raw_mp;
                                        lit_frontiers[off].entries[0].base_cost = min_base;
                                        lit_frontiers[off].entries[0].node_idx = lit_node0[off];
                                        lit_frontiers[off].entries[1].pos = pos;
                                        lit_frontiers[off].entries[1].base_cost = mat_base_cost;
                                        lit_frontiers[off].entries[1].node_idx = created_mat_node;
                                        os[off].lit_max_pos = pos | (1 << 20);
                                    }
                                } else {
                                    frontier_push(&lit_frontiers[off], pos, mat_base_cost, created_mat_node);
                                    int new_mb = lit_frontiers[off].entries[0].base_cost;
                                    os[off].lit_min_base = new_mb;
                                    lit_node0[off] = lit_frontiers[off].entries[0].node_idx;
                                    if (new_mb < ws->lit_blk64_min[off >> 6]) {
                                        ws->lit_blk64_min[off >> 6] = new_mb;
                                    }
                                    if (new_mb < ws->lit_blk8_min[off >> 3]) {
                                        ws->lit_blk8_min[off >> 3] = new_mb;
                                    }
                                    int fc = lit_frontiers[off].count;
                                    int fp = lit_frontiers[off].entries[fc - 1].pos;
                                    os[off].lit_max_pos = (fc > 1) ? (fp | (1 << 20)) : fp;
                                }
                            }
                        }
                                }
                            }
                        }
                    }
                    if (use_swar_tail && p1b_chain > chain_1bb_lim && idx1b >= min_idx1 && p1b_steps < max_p1b_steps) {
                        eval_1b_swar_tail(
                            ws, input_data, input_size, pos, (pos - 1) - idx1b, active_max_off,
                            cur_byte, prv_byte, nxt_byte, max_any_raw_1b, prev_lit_bits, literal_pos_bits,
                            cur_pad_1b, f_pad, need_tokens, best_any_cost, best_any_node,
                            &best_mat_cost_at_pos, &best_mat_node_at_pos
                        );
                    }
                }
            }
        } else {
            if (wildcard_mask == NULL) {
                for (int o = 1; o <= max_window; o++) {
                    if (os[o].lit_min_base < COST_INFINITY && !(os[o].lit_max_pos & (1 << 20))) {
                        lit_frontiers[o].count = 1;
                        lit_frontiers[o].entries[0].pos = os[o].lit_max_pos;
                        lit_frontiers[o].entries[0].base_cost = os[o].lit_min_base;
                        lit_frontiers[o].entries[0].node_idx = lit_node0[o];
                    }
                    if (os[o].rep_valid_pos >= 0 && (os[o].rep_valid_pos & (1 << 20))) {
                        rep_frontiers[o].count = 1;
                        rep_frontiers[o].entries[0].pos = rep_pos0[o];
                        rep_frontiers[o].entries[0].base_cost = rep_cost0[o];
                        rep_frontiers[o].entries[0].node_idx = rep_node0[o];
                    } else {
                        rep_frontiers[o].count = 0;
                    }
                }
            }
            uint8_t cur_byte = input_data[pos - 1];
            uint8_t nxt_byte = (pos < input_size) ? input_data[pos] : 0;
            int off_lo = 1;
            int off_hi = loop_max_off;
            if (wildcard_mask == NULL && pos == input_size) {
                if (required_final_offset > 0) {
                    off_lo = required_final_offset;
                    off_hi = required_final_offset <= loop_max_off ? required_final_offset : 0;
                } else if (req_bh_ff && off_hi > 256) {
                    off_hi = 256;
                }
            }

            for (int off = off_lo; off <= off_hi; off++) {
                int cur_matches = (wildcard_mask == NULL)
                                ? (off <= active_max_off && input_data[pos - 1 - off] == cur_byte)
                                : (off <= active_max_off && can_match_byte(input_data, wildcard_mask, pos - 1, off));
                int next_pos_matches = (pos == input_size)
                                     ? 1
                                     : ((wildcard_mask == NULL)
                                        ? (off <= pos && input_data[pos - off] == nxt_byte)
                                        : (off <= pos && can_match_byte(input_data, wildcard_mask, pos, off)));
                int ml = 0;
                int is_rep_active = 0;
                if (cur_matches) {
                    if (wildcard_mask == NULL) {
                        int cur_idx1 = (pos - 1) - off;
                        int L0 = run_start[pos - 1];
                        int run0 = (pos - 1) - L0 + 1;
                        int L1 = run_start[cur_idx1];
                        int run1 = cur_idx1 - L1 + 1;
                        if (run1 < run0) {
                            ml = run1;
                        } else if (run1 > run0) {
                            ml = run0;
                        } else {
                            ml = compute_exact_ml_run_eq(input_data, run_start, last_match_pos, match_lengths, pos, off, L0, L1, run0);
                        }
                    } else {
                        ml = (last_match_pos[off] == pos - 1) ? (match_lengths[off] + 1) : 1;
                    }
                    match_lengths[off] = ml;
                    last_match_pos[off] = pos;
                    if (wildcard_mask == NULL) {
                        int start_m = pos - ml;
                        int rv_raw = os[off].rep_valid_pos;
                        int rv_pos = rv_raw & 0xFFFFF;
                        if (os[off].lit_min_base < COST_INFINITY && (os[off].lit_max_pos & 0xFFFFF) < start_m && (rv_raw < 0 || rv_pos < start_m)) {
                            int lit_src_node_m = -1;
                            int raw_eval = frontier_evaluate_min(&lit_frontiers[off], start_m, &lit_src_node_m);
                            if (raw_eval < COST_INFINITY) {
                                int lit_cost_m = raw_eval + lit_cum_bits[start_m];
                                int created_lit_node_m = need_tokens ? node_pool_push(pool, lit_src_node_m, start_m, 0) : 0;
                                rep_frontiers[off].count = 1;
                                rep_frontiers[off].entries[0].pos = start_m;
                                rep_frontiers[off].entries[0].base_cost = lit_cost_m;
                                rep_frontiers[off].entries[0].node_idx = created_lit_node_m;
                                os[off].rep_valid_pos = pos | (1 << 20);
                                is_rep_active = 1;
                            } else {
                                os[off].rep_valid_pos = pos;
                            }
                        } else if (rv_raw >= 0 && rv_pos >= start_m) {
                            is_rep_active = (rv_raw & (1 << 20)) != 0;
                            os[off].rep_valid_pos = pos | (rv_raw & (1 << 20));
                        }
                    } else if (os[off].rep_valid_pos == pos - 1) {
                        os[off].rep_valid_pos = pos;
                        is_rep_active = 1;
                    }
                } else {
                    if (!next_pos_matches) {
                        continue;
                    }
                }

                int min_base = os[off].lit_min_base;

                /* 1. Evaluate Literal run ending at pos only when needed for LAST_OFFSET or EOF */
                int lit_src_node = -1;
                int lit_cost = COST_INFINITY;
                int created_lit_node = -1;
                if (min_base < COST_INFINITY && (pos == input_size || next_pos_matches)) {
                    int raw_eval = frontier_evaluate_min(&lit_frontiers[off], pos, &lit_src_node);
                    if (raw_eval < COST_INFINITY) {
                        lit_cost = raw_eval + literal_pos_bits;
                        created_lit_node = need_tokens ? node_pool_push(pool, lit_src_node, pos, 0) : 0;
                        if (lit_cost < best_any_cost[pos]) {
                            best_any_cost[pos] = lit_cost;
                            best_any_node[pos] = created_lit_node;
                        }
                    }
                }

                /* 2. Evaluate Match (LAST_OFFSET or NEW_OFFSET) ending at pos with offset = off */
                int mat_cost = COST_INFINITY;
                int mat_prev_node = -1;

                if (ml >= 1 && is_rep_active) {
                    int rep_src_node = -1;
                    int rep_cost = frontier_evaluate_min(&rep_frontiers[off], pos, &rep_src_node);
                    if (rep_cost < mat_cost) {
                        mat_cost = rep_cost;
                        mat_prev_node = rep_src_node;
                    }
                }

                if (ml >= MIN_MATCH_LENGTH_NEW_OFFSET) {
                    int best_s = 0;
                    int best_c = eval_best_new_offset_len(
                        ws, best_any_cost, best_len_cost, best_len_start,
                        &computed_best_len, &computed_bucket_idx, bucket_pref_cost, bucket_pref_start,
                        pos, ml, &best_s
                    );
                    if (best_c < COST_INFINITY) {
                        int new_off_cost = best_c + os[off].new_off_overhead;
                        if (new_off_cost < mat_cost) {
                            mat_cost = new_off_cost;
                            mat_prev_node = best_any_node[best_s];
                        }
                    }
                }

                int created_mat_node = -1;
                if (mat_cost < COST_INFINITY) {
                    int mat_base_cost = mat_cost - literal_pos_bits;
                    int enters_lit_frontier = (min_base == COST_INFINITY) ||
                                              ((mat_base_cost - min_base) <= 31);
                    if (pos == input_size || mat_cost < best_any_cost[pos] || enters_lit_frontier) {
                        created_mat_node = need_tokens ? node_pool_push(pool, mat_prev_node, pos, off) : 0;
                        if (mat_cost < best_any_cost[pos]) {
                            best_any_cost[pos] = mat_cost;
                            best_any_node[pos] = created_mat_node;
                        }
                        if (mat_cost < best_mat_cost_at_pos) {
                            best_mat_cost_at_pos = mat_cost;
                            best_mat_node_at_pos = created_mat_node;
                        }
                        if (enters_lit_frontier) {
                            if (min_base == COST_INFINITY) lit_frontiers[off].count = 0;
                            frontier_push(&lit_frontiers[off], pos, mat_base_cost, created_mat_node);
                            int new_mb = lit_frontiers[off].entries[0].base_cost;
                            os[off].lit_min_base = new_mb;
                            if (new_mb < ws->lit_blk64_min[off >> 6]) {
                                ws->lit_blk64_min[off >> 6] = new_mb;
                            }
                            { int fc = lit_frontiers[off].count; int fp = lit_frontiers[off].entries[fc - 1].pos; os[off].lit_max_pos = (fc > 1) ? (fp | (1 << 20)) : fp; }
                        }
                    }
                }

                /* Push literal state after match evaluation so LAST_OFFSET matches have length >= 1 */
                if (created_lit_node >= 0 && next_pos_matches) {
                    if (wildcard_mask == NULL) {
                        if (os[off].rep_valid_pos != (pos | (1 << 20))) {
                            rep_frontiers[off].count = 0;
                        }
                        frontier_push(&rep_frontiers[off], pos, lit_cost, created_lit_node);
                        os[off].rep_valid_pos = (rep_frontiers[off].count > 0) ? (pos | (1 << 20)) : pos;
                    } else {
                        if (os[off].rep_valid_pos != pos) {
                            rep_frontiers[off].count = 0;
                        }
                        frontier_push(&rep_frontiers[off], pos, lit_cost, created_lit_node);
                        if (rep_frontiers[off].count > 0) {
                            os[off].rep_valid_pos = pos;
                        }
                    }
                }

                if (pos == input_size) {
                    final_lit_cost[off] = lit_cost;
                    final_lit_node[off] = created_lit_node;
                    final_mat_cost[off] = mat_cost;
                    final_mat_node[off] = created_mat_node;
                }
            }
        }

        if (best_mat_cost_at_pos < COST_INFINITY) {
            frontier_push(&lit_frontiers[0], pos, best_mat_cost_at_pos - literal_pos_bits, best_mat_node_at_pos);
            os[0].lit_min_base = lit_frontiers[0].entries[0].base_cost;
            os[0].lit_max_pos = pos;
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
    } else if (req_bh_ff) {
        int max_term_off = max_window < 256 ? max_window : 256;
        for (int off = 1; off <= max_term_off; off++) {
            if (final_lit_cost[off] < terminal_cost) {
                terminal_cost = final_lit_cost[off];
                terminal_node = final_lit_node[off];
            }
            if (final_mat_cost[off] < terminal_cost) {
                terminal_cost = final_mat_cost[off];
                terminal_node = final_mat_node[off];
            }
        }
    } else {
        terminal_cost = best_any_cost[input_size];
        terminal_node = best_any_node[input_size];
    }

    int success = (terminal_cost < COST_INFINITY && (!need_tokens || terminal_node >= 0));
    int total_payload_bits = success ? (terminal_cost + EOF_CONTROL_AND_ELIAS_BITS) : 0;
    if (success && need_tokens) {
        int chain_len = 0;
        for (int cur = terminal_node; cur >= 0; cur = pool->nodes[cur].prev_node) {
            chain_len++;
        }
        Token *tokens = (Token *)malloc((size_t)chain_len * sizeof(Token));
        assert(tokens != NULL);

        int idx = chain_len - 1;
        for (int cur = terminal_node; cur >= 0; cur = pool->nodes[cur].prev_node) {
            int prev = pool->nodes[cur].prev_node;
            int start_pos = (prev >= 0) ? pool->nodes[prev].end_pos : 0;
            tokens[idx].end_pos = pool->nodes[cur].end_pos;
            tokens[idx].length = pool->nodes[cur].end_pos - start_pos;
            tokens[idx].offset = pool->nodes[cur].offset;
            idx--;
        }
        plan_out->tokens = tokens;
        plan_out->token_count = chain_len;
        plan_out->payload_bits = total_payload_bits;
    }

    return total_payload_bits;
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

static size_t compute_exact_stream_size(
    int payload_bits,
    UltraFormat format,
    const UltraMetadata *meta
) {
    int eof_symbol = ZX0_EOF_MSB_VALUE;
    if ((format == ULTRA_FORMAT_COMPAT_82B || format == ULTRA_FORMAT_X86E8_84B) && meta->filter_passes > 0) {
        eof_symbol = ZX0_EOF_ALT_VALUE;
    } else if (format == ULTRA_FORMAT_UNIV_93B && meta->filter_passes > 0) {
        eof_symbol = (meta->filter_stride == 0) ? ZX0_EOF_E8_VALUE
                   : (meta->filter_stride == -1 ? ZX0_EOF_E9_VALUE : ZX0_EOF_ALT_VALUE);
    } else if ((format == ULTRA_FORMAT_HYBRID_108B || format == ULTRA_FORMAT_PRO_111B) && meta->filter_passes > 0) {
        if (meta->filter_stride <= 0) {
            int is_two_opcodes = (meta->filter_passes >= 2 && meta->filter_strides[1] <= 0);
            eof_symbol = is_two_opcodes ? ZX0_EOF_ULTRA_E8E9_VALUE
                       : (meta->filter_stride == -1 ? ZX0_EOF_ULTRA_E9_VALUE : ZX0_EOF_ULTRA_E8_VALUE);
        } else {
            eof_symbol = ZX0_EOF_ALT_VALUE;
        }
    } else if (format == ULTRA_FORMAT_ULTRA_128B && meta->filter_passes > 0) {
        if (meta->filter_stride == 0) {
            eof_symbol = ZX0_EOF_ULTRA128_E8_VALUE;
        } else if (meta->filter_stride == -1) {
            eof_symbol = ZX0_EOF_ULTRA128_E9_VALUE;
        } else {
            eof_symbol = ZX0_EOF_ALT_VALUE;
        }
    }
    int extra_eof_bits = (1 + compute_elias_gamma_bits(eof_symbol)) - EOF_CONTROL_AND_ELIAS_BITS;
    int total_bits = payload_bits + extra_eof_bits;

    size_t body_bytes;
    if (format == ULTRA_FORMAT_ADPK_77B) {
        body_bytes = 1 + (size_t)((total_bits + 7) >> 3);
    } else if (format == ULTRA_FORMAT_ULTRA_128B) {
        body_bytes = 2 + (size_t)(total_bits >> 3);
    } else {
        body_bytes = (size_t)((total_bits + 7) >> 3);
    }

    size_t trailer_bytes = 0;
    if (format == ULTRA_FORMAT_STRD_80B) {
        trailer_bytes = 2;
    } else if (format == ULTRA_FORMAT_COMPAT_82B ||
               format == ULTRA_FORMAT_X86E8_84B ||
               format == ULTRA_FORMAT_UNIV_93B) {
        trailer_bytes = (meta->filter_passes > 0) ? 2 : 0;
    } else if (format == ULTRA_FORMAT_STRD_82B) {
        trailer_bytes = (size_t)(meta->filter_passes * 2 + 2);
    } else if (format == ULTRA_FORMAT_HYBRID_108B) {
        if (meta->filter_passes > 0) {
            trailer_bytes = 2;
            for (int p = 0; p < meta->filter_passes; p++) {
                trailer_bytes += (meta->filter_strides[p] > 0) ? 3 : 2;
            }
        }
    } else if (format == ULTRA_FORMAT_PRO_111B) {
        if (meta->filter_passes > 0) {
            trailer_bytes = 2;
            for (int p = 0; p < meta->filter_passes; p++) {
                trailer_bytes += (meta->filter_strides[p] > 0) ? 4 : 2;
            }
        }
    } else if (format == ULTRA_FORMAT_ULTRA_128B) {
        trailer_bytes = 2;
        for (int p = 0; p < meta->filter_passes; p++) {
            trailer_bytes += (meta->filter_strides[p] > 0) ? 4 : 2;
        }
    }
    return body_bytes + trailer_bytes;
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
    int dh_sentinel = (format == ULTRA_FORMAT_ULTRA_128B)
                    ? ((meta->dh_sentinel >= 1 && meta->dh_sentinel <= 127)
                       ? meta->dh_sentinel
                       : find_best_dh_sentinel(working_input, total_raw_len, NULL))
                    : 0;
    DhSpec dh_spec;
    if (format == ULTRA_FORMAT_ULTRA_128B) {
        build_dh_spec(dh_sentinel, &dh_spec);
    }

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
        bit_writer_put_byte(&bw, (uint8_t)dh_sentinel); /* Byte 1: DH sentinel (Asymmetric Split-Domain) */
        bw.bit_byte_index = 0;
        bw.bit_mask = 128;
    }

    int last_offset = (format == ULTRA_FORMAT_ULTRA_128B) ? 0 : ZX0_INITIAL_OFFSET;
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
                    bit_writer_put_bit(&bw, dh_spec.prefix_bit[b]);
                    int m = dh_spec.payload_len[b];
                    int pv = dh_spec.payload_val[b];
                    for (int bit_idx = m - 1; bit_idx >= 0; bit_idx--) {
                        bit_writer_put_bit(&bw, (pv >> bit_idx) & 1);
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
    } else if (format == ULTRA_FORMAT_UNIV_93B && meta->filter_passes > 0) {
        eof_symbol = (meta->filter_stride == 0) ? ZX0_EOF_E8_VALUE
                   : (meta->filter_stride == -1 ? ZX0_EOF_E9_VALUE : ZX0_EOF_ALT_VALUE);
    } else if ((format == ULTRA_FORMAT_HYBRID_108B || format == ULTRA_FORMAT_PRO_111B) && meta->filter_passes > 0) {
        if (meta->filter_stride <= 0) {
            int is_two_opcodes = (meta->filter_passes >= 2 && meta->filter_strides[1] <= 0);
            eof_symbol = is_two_opcodes ? ZX0_EOF_ULTRA_E8E9_VALUE
                       : (meta->filter_stride == -1 ? ZX0_EOF_ULTRA_E9_VALUE : ZX0_EOF_ULTRA_E8_VALUE);
        } else {
            eof_symbol = ZX0_EOF_ALT_VALUE;
        }
    } else if (format == ULTRA_FORMAT_ULTRA_128B && meta->filter_passes > 0) {
        if (meta->filter_stride == 0) {
            eof_symbol = ZX0_EOF_ULTRA128_E8_VALUE;
        } else if (meta->filter_stride == -1) {
            eof_symbol = ZX0_EOF_ULTRA128_E9_VALUE;
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
                int st = (meta->filter_strides[p] != 0) ? meta->filter_strides[p] : meta->filter_stride;
                if (st > 0) {
                    bit_writer_put_byte(&bw, (uint8_t)((256 - st) & 0xFF));
                }
            }
            bit_writer_put_u16_le(&bw, 0);
        }
    } else if (format == ULTRA_FORMAT_PRO_111B) {
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
    } else if (format == ULTRA_FORMAT_ULTRA_128B) {
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

    assert(bw.size <= compute_exact_stream_size(plan->payload_bits, format, meta));
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
    int last_offset = (format == ULTRA_FORMAT_ULTRA_128B) ? 0 : ZX0_INITIAL_OFFSET;
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
                int p_bit = bit_reader_get_bit(&br);
                if (p_bit < 0) {
                    free(out);
                    return NULL;
                }
                int al = (p_bit == 0) ? ultra_dh_sentinel : ((ultra_dh_sentinel & 5) + 1);
                while (al < 256) {
                    int bit = bit_reader_get_bit(&br);
                    if (bit < 0) {
                        free(out);
                        return NULL;
                    }
                    al = (al << 1) | bit;
                }
                out[out_pos++] = (uint8_t)((al - 1) & 0xFF);
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
               (format == ULTRA_FORMAT_UNIV_93B &&
                (eof_msb_read == ZX0_EOF_E8_VALUE || eof_msb_read == ZX0_EOF_E9_VALUE))) {
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
        uint8_t target_op = (format == ULTRA_FORMAT_UNIV_93B && eof_msb_read == ZX0_EOF_E9_VALUE)
                          ? X86_OPCODE_JMP_REL16
                          : X86_OPCODE_CALL_REL16;
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
    } else if (((format == ULTRA_FORMAT_HYBRID_108B ||
                 format == ULTRA_FORMAT_PRO_111B) &&
                eof_msb_read > ZX0_EOF_MSB_VALUE) ||
               (format == ULTRA_FORMAT_ULTRA_128B &&
                eof_msb_read >= ZX0_EOF_MSB_VALUE)) {
        int e8_stage = 0;
        uint8_t single_op = X86_OPCODE_CALL_REL16;
        if (format == ULTRA_FORMAT_HYBRID_108B || format == ULTRA_FORMAT_PRO_111B) {
            if (eof_msb_read == ZX0_EOF_ULTRA_E8E9_VALUE) {
                e8_stage = 2;
            } else if (eof_msb_read == ZX0_EOF_ULTRA_E8_VALUE) {
                e8_stage = 1;
            } else if (eof_msb_read == ZX0_EOF_ULTRA_E9_VALUE) {
                e8_stage = 1;
                single_op = X86_OPCODE_JMP_REL16;
            }
        } else {
            if (eof_msb_read == ZX0_EOF_ULTRA128_E8_VALUE) {
                e8_stage = 1;
            } else if (eof_msb_read == ZX0_EOF_ULTRA128_E9_VALUE) {
                e8_stage = 1;
                single_op = X86_OPCODE_JMP_REL16;
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
                uint8_t target_op = (e8_total == 1)
                                  ? single_op
                                  : ((e8_stage == e8_total) ? X86_OPCODE_CALL_REL16 : X86_OPCODE_JMP_REL16);
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
                if (stride < 1 || stride > 128 || (size_t)count + (size_t)stride > out_pos) {
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

int ultra_screen_zx0_v2(
    const uint8_t *input_data,
    int input_size,
    const uint8_t *wildcard_mask,
    int max_offset_limit,
    size_t *compressed_size_out,
    int *compressed_bits_out
) {
    assert(input_data != NULL && input_size > 0);
    int effective_limit = (max_offset_limit > 0 && max_offset_limit <= ZX0_MAX_OFFSET)
                        ? max_offset_limit
                        : ZX0_MAX_OFFSET;

    if (wildcard_mask != NULL) {
        int has_any_wildcard = 0;
        for (int i = 0; i < input_size; i++) {
            if (wildcard_mask[i] != 0) {
                has_any_wildcard = 1;
                break;
            }
        }
        if (has_any_wildcard) {
            uint8_t *working_input = (uint8_t *)malloc((size_t)input_size);
            assert(working_input != NULL);
            memcpy(working_input, input_data, (size_t)input_size);
            ParsePlan wp = {NULL, 0, 0};
            if (solve_exact_parse(working_input, input_size, wildcard_mask, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, &wp)) {
                materialize_wildcards_from_plan(working_input, wildcard_mask, &wp);
                free(wp.tokens);
            }
            int bits = solve_exact_parse(working_input, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, NULL);
            free(working_input);
            if (compressed_bits_out != NULL) {
                *compressed_bits_out = bits;
            }
            if (compressed_size_out != NULL) {
                *compressed_size_out = (size_t)((bits + 6) >> 3);
            }
            return bits > 0;
        }
    }

    int bits = solve_exact_parse(input_data, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, NULL);
    if (compressed_bits_out != NULL) {
        *compressed_bits_out = bits;
    }
    if (compressed_size_out != NULL) {
        *compressed_size_out = (size_t)((bits + 6) >> 3);
    }
    return bits > 0;
}

static void reconstruct_filtered_buffer(
    const uint8_t *working_input,
    uint8_t *f_cur,
    uint8_t *f_nxt,
    int input_size,
    UltraFormat format,
    const UltraMetadata *meta
) {
    memcpy(f_cur, working_input, (size_t)input_size);
    if (meta->filter_passes <= 0) {
        return;
    }
    if (format == ULTRA_FORMAT_STRD_82B) {
        for (int p = 0; p < meta->filter_passes; p++) {
            apply_forward_stride_difference(f_cur, f_nxt, input_size, meta->filter_stride, meta->filter_counts[p]);
            memcpy(f_cur, f_nxt, (size_t)input_size);
        }
    } else {
        for (int p = meta->filter_passes - 1; p >= 0; p--) {
            int st = meta->filter_strides[p];
            int tc = meta->filter_counts[p];
            if (st <= 0) {
                uint8_t op = (st == -1 || (p == 1 && meta->filter_strides[0] <= 0))
                           ? X86_OPCODE_JMP_REL16
                           : X86_OPCODE_CALL_REL16;
                apply_forward_x86_opcode_filter(f_cur, f_nxt, input_size, tc, op);
            } else if (meta->filter_word_modes[p]) {
                apply_forward_word16_stride_difference(f_cur, f_nxt, input_size, st, tc);
            } else {
                apply_forward_stride_difference(f_cur, f_nxt, input_size, st, tc);
            }
            memcpy(f_cur, f_nxt, (size_t)input_size);
        }
    }
}

int g_single_e8_only = 0;

static UltraMetadata g_top_cands[8];
static size_t g_top_sizes[8];
static int g_n_top_cands = 0;

static int same_filter_shape(const UltraMetadata *a, const UltraMetadata *b) {
    if (a->filter_passes != b->filter_passes) return 0;
    for (int i = 0; i < a->filter_passes; i++) {
        if (a->filter_strides[i] != b->filter_strides[i] ||
            a->filter_counts[i] != b->filter_counts[i] ||
            a->filter_word_modes[i] != b->filter_word_modes[i]) return 0;
    }
    return 1;
}

static void record_top_cand(const UltraMetadata *m, size_t csize) {
    for (int i = 0; i < g_n_top_cands; i++) {
        if (same_filter_shape(&g_top_cands[i], m)) {
            if (csize < g_top_sizes[i] || (csize == g_top_sizes[i] && m->compressed_bits < g_top_cands[i].compressed_bits)) {
                g_top_cands[i] = *m;
                g_top_sizes[i] = csize;
                while (i > 0 && (g_top_sizes[i] < g_top_sizes[i-1] ||
                       (g_top_sizes[i] == g_top_sizes[i-1] && g_top_cands[i].compressed_bits < g_top_cands[i-1].compressed_bits))) {
                    UltraMetadata tm = g_top_cands[i]; g_top_cands[i] = g_top_cands[i-1]; g_top_cands[i-1] = tm;
                    size_t ts = g_top_sizes[i]; g_top_sizes[i] = g_top_sizes[i-1]; g_top_sizes[i-1] = ts;
                    i--;
                }
            }
            return;
        }
    }
    int pos = g_n_top_cands;
    if (pos < 8) {
        g_n_top_cands++;
    } else {
        if (csize > g_top_sizes[7] || (csize == g_top_sizes[7] && m->compressed_bits >= g_top_cands[7].compressed_bits)) return;
        pos = 7;
    }
    g_top_cands[pos] = *m;
    g_top_sizes[pos] = csize;
    while (pos > 0 && (g_top_sizes[pos] < g_top_sizes[pos-1] ||
           (g_top_sizes[pos] == g_top_sizes[pos-1] && g_top_cands[pos].compressed_bits < g_top_cands[pos-1].compressed_bits))) {
        UltraMetadata tm = g_top_cands[pos]; g_top_cands[pos] = g_top_cands[pos-1]; g_top_cands[pos-1] = tm;
        size_t ts = g_top_sizes[pos]; g_top_sizes[pos] = g_top_sizes[pos-1]; g_top_sizes[pos-1] = ts;
        pos--;
    }
}

static int evaluate_candidate_meta(
    const uint8_t *buf,
    int input_size,
    UltraFormat format,
    int lsb_bits,
    int effective_limit,
    int req_off,
    UltraMetadata *cand_meta,
    size_t *best_size_io,
    UltraMetadata *best_meta_io,
    size_t *out_cand_size
) {
    if (format == ULTRA_FORMAT_ULTRA_128B) {
        if (cand_meta->dh_sentinel <= 0 || cand_meta->dh_sentinel > 127) {
            cand_meta->dh_sentinel = (req_off < 0 && -req_off >= 1 && -req_off <= 127)
                                   ? -req_off
                                   : find_best_dh_sentinel(buf, input_size, NULL);
        }
        req_off = -cand_meta->dh_sentinel;
    }
    int bits = solve_exact_parse(buf, input_size, NULL, lsb_bits, effective_limit, req_off, NULL);
    if (bits <= 0) {
        return 0;
    }
    cand_meta->compressed_bits = bits;
    size_t csize = compute_exact_stream_size(bits, format, cand_meta);
    if (out_cand_size != NULL) {
        *out_cand_size = csize;
    }
    #pragma omp critical
    {
        if (tls_deep_mode < 0) record_top_cand(cand_meta, csize);
        if (csize < *best_size_io ||
            (csize == *best_size_io && bits < best_meta_io->compressed_bits)) {
            *best_size_io = csize;
            *best_meta_io = *cand_meta;
        }
    }
    return bits;
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

    int top_level_call = !omp_in_parallel();
    if (format == ULTRA_FORMAT_ZX0_V2) {
        if (top_level_call) g_deep_mode = 1; else tls_deep_mode = 1;
        ParsePlan plan = {NULL, 0, 0};
        int ok = solve_exact_parse(working_input, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, &plan);
        assert(ok);
        if (input_size > 32768 && plan.payload_bits > 90000 && (plan.payload_bits & 7) >= 6) {
            if (top_level_call) g_deep_mode = 3; else tls_deep_mode = 3;
            ParsePlan plan3 = {NULL, 0, 0};
            if (solve_exact_parse(working_input, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, &plan3)) {
                if (plan3.payload_bits < plan.payload_bits) {
                    free(plan.tokens);
                    plan = plan3;
                } else {
                    free(plan3.tokens);
                }
            }
            if (top_level_call) g_deep_mode = 1; else tls_deep_mode = 1;
        }
        best_meta.compressed_bits = plan.payload_bits;
        best_stream = encode_plan_to_stream(working_input, &plan, format, ZX0_DEFAULT_LSB_BITS, &best_meta, &best_size);
        free(plan.tokens);
    } else if (format == ULTRA_FORMAT_ADPK_77B) {
        if (top_level_call) { g_n_top_cands = 0; g_deep_mode = 0; } else tls_deep_mode = 0;
        #pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
        for (int k = 1; k <= MAX_ADPK_LSB_BITS; k++) {
            if (k > 2 && (1 << (k - 1)) >= input_size * 2) {
                continue;
            }
            UltraMetadata cand_meta;
            memset(&cand_meta, 0, sizeof(cand_meta));
            cand_meta.format = format;
            cand_meta.offset_lsb_bits = k;
            evaluate_candidate_meta(working_input, input_size, format, k, effective_limit, 0, &cand_meta, &best_size, &best_meta, NULL);
        }
        if (top_level_call) g_deep_mode = 1; else tls_deep_mode = 1;
        ParsePlan dplan = {NULL, 0, 0};
        int ok = solve_exact_parse(working_input, input_size, NULL, best_meta.offset_lsb_bits, effective_limit, 0, &dplan);
        assert(ok);
        best_meta.compressed_bits = dplan.payload_bits;
        best_stream = encode_plan_to_stream(working_input, &dplan, format, best_meta.offset_lsb_bits, &best_meta, &best_size);
        free(dplan.tokens);
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
        if (top_level_call) { g_n_top_cands = 0; g_deep_mode = 0; } else tls_deep_mode = 0;
        int is_word_capable = (format == ULTRA_FORMAT_ULTRA_128B || format == ULTRA_FORMAT_PRO_111B);
        int is_multi_stage  = (format == ULTRA_FORMAT_HYBRID_108B || is_word_capable);
        int base_req_off = (format == ULTRA_FORMAT_ULTRA_128B) ? -200 : 0;
        int defer_uni = (input_size >= 8192 && !omp_in_parallel() && format != ULTRA_FORMAT_X86E8_84B);
        UltraMetadata uni_cands[320];
        int uni_req_offs[320];
        int n_uni_cands = 0;
        int base_payload_bits = input_size * 8;
        size_t base_unfiltered_size = (size_t)-1;
        {
            UltraMetadata base_meta;
            memset(&base_meta, 0, sizeof(base_meta));
            base_meta.format = format;
            base_meta.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
            base_meta.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                   ? detect_buffer_literal_bits(working_input, input_size, NULL) : 8;
            if (defer_uni) {
                best_meta = base_meta;
                best_meta.compressed_bits = base_payload_bits;
                uni_req_offs[n_uni_cands] = base_req_off;
                uni_cands[n_uni_cands++] = base_meta;
                if (format == ULTRA_FORMAT_ULTRA_128B) {
                    int raw_est_dh = find_best_dh_sentinel(working_input, input_size, NULL);
                    int uniq_dhs[8];
                    int n_uniq = rank_top_dh_sentinels(working_input, input_size, raw_est_dh, uniq_dhs, 4);
                    for (int ui = 0; ui < n_uniq && n_uni_cands < 320; ui++) {
                        UltraMetadata alt_meta = base_meta;
                        alt_meta.dh_sentinel = uniq_dhs[ui];
                        uni_req_offs[n_uni_cands] = -uniq_dhs[ui];
                        uni_cands[n_uni_cands++] = alt_meta;
                    }
                }
            } else {
                base_payload_bits = evaluate_candidate_meta(
                    working_input, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                    base_req_off, &base_meta, &best_size, &best_meta, NULL
                );
                assert(base_payload_bits > 0);
                base_unfiltered_size = best_size;
            }
        }

        uint8_t *best_e8_buf = NULL;
        int best_e8_tail_chosen = 0;

        /* A. Evaluate position-independent x86 E8 Call-Target transform (X86E8_84B, UNIV_93B, HYBRID_108B, PRO_111B, ULTRA_128B) */
        if ((format == ULTRA_FORMAT_X86E8_84B || format == ULTRA_FORMAT_UNIV_93B || is_multi_stage) &&
            input_size >= 3 &&
            working_input[input_size - 2] != X86_OPCODE_CALL_REL16 &&
            working_input[input_size - 1] != X86_OPCODE_CALL_REL16) {
            int full_tail = input_size <= 65535 ? input_size : 65535;
            int min_scan_idx = input_size - full_tail;
            int total_e8_ops = 0;
            for (int i = min_scan_idx; i <= input_size - 3; i++) {
                if (working_input[i] == X86_OPCODE_CALL_REL16) {
                    total_e8_ops++;
                }
            }
            int cand_tails[6];
            int num_tails = 0;
            if (total_e8_ops >= 2) {
                uint16_t *target_freq = (uint16_t *)calloc(65536, sizeof(uint16_t));
                assert(target_freq != NULL);
                int repeated_e8_hits = 0;
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
                            repeated_e8_hits++;
                        }
                    }
                }
                if (input_size < 4096 || repeated_e8_hits >= 4) {
                    cand_tails[num_tails++] = full_tail;
                    int max_seen = (input_size >= 8192) ? 0 : 4;
                    int seen_e8 = 0;
                    for (int i = min_scan_idx; i <= input_size - 3 && seen_e8 < max_seen; i++) {
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
                    if (best_e8_tail >= 3 && best_e8_tail < full_tail - 16 &&
                        (input_size < 8192 || best_e8_gain >= 18)) {
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
                }
                free(target_freq);
            }

            size_t best_e8_only_size = best_size;
            int nested_parallel = omp_in_parallel();
            #pragma omp parallel for schedule(dynamic) if (!nested_parallel && !defer_uni)
            for (int ti = 0; ti < num_tails; ti++) {
                int tail_count = cand_tails[ti];
                uint8_t *e8_buf = (uint8_t *)malloc((size_t)input_size);
                assert(e8_buf != NULL);
                int hits = apply_forward_x86_opcode_filter(working_input, e8_buf, input_size, tail_count, X86_OPCODE_CALL_REL16);
                if (hits >= 2) {
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
                    if (defer_uni) {
                        if (n_uni_cands < 320) {
                            uni_req_offs[n_uni_cands] = base_req_off;
                            uni_cands[n_uni_cands++] = meta_e8;
                        }
                        if (best_e8_buf == NULL) {
                            best_e8_buf = (uint8_t *)malloc(1);
                            best_e8_tail_chosen = tail_count;
                        }
                    } else {
                        size_t size_e8 = 0;
                        if (evaluate_candidate_meta(
                                e8_buf, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                base_req_off, &meta_e8, &best_size, &best_meta, &size_e8)) {
                            #pragma omp critical
                            {
                                if (size_e8 < best_e8_only_size) {
                                    best_e8_only_size = size_e8;
                                    free(best_e8_buf);
                                    best_e8_buf = (uint8_t *)malloc((size_t)input_size);
                                    memcpy(best_e8_buf, e8_buf, (size_t)input_size);
                                    best_e8_tail_chosen = tail_count;
                                }
                            }
                        }
                    }

                    /* For HYBRID_108B, PRO_111B: also evaluate chaining E8 (CALL rel16) + E9 (JMP rel16) */
                    if (is_multi_stage && format != ULTRA_FORMAT_ULTRA_128B && !g_single_e8_only &&
                        working_input[input_size - 2] != X86_OPCODE_JMP_REL16 &&
                        working_input[input_size - 1] != X86_OPCODE_JMP_REL16) {
                        uint8_t *e9_tmp = (uint8_t *)malloc((size_t)input_size);
                        uint8_t *e8e9_buf = (uint8_t *)malloc((size_t)input_size);
                        assert(e9_tmp != NULL && e8e9_buf != NULL);
                        int hits_e9 = apply_forward_x86_opcode_filter(working_input, e9_tmp, input_size, tail_count, X86_OPCODE_JMP_REL16);
                        int hits_e8_2 = (hits_e9 >= 2 &&
                                         e9_tmp[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                         e9_tmp[input_size - 1] != X86_OPCODE_CALL_REL16)
                                      ? apply_forward_x86_opcode_filter(e9_tmp, e8e9_buf, input_size, tail_count, X86_OPCODE_CALL_REL16)
                                      : 0;
                        if (hits_e9 >= 2 && hits_e8_2 >= 2) {
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
                            if (defer_uni) {
                                if (n_uni_cands < 320) {
                                    uni_req_offs[n_uni_cands] = base_req_off;
                                    uni_cands[n_uni_cands++] = meta_e89;
                                }
                            } else {
                                evaluate_candidate_meta(
                                    e8e9_buf, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                    base_req_off, &meta_e89, &best_size, &best_meta, NULL
                                );
                            }
                        }
                        free(e9_tmp);
                        free(e8e9_buf);
                    }
                }
                free(e8_buf);
            }
        }

        /* B. Multi-core search over candidate strides (STRD_80B, COMPAT_82B, STRD_82B, UNIV_93B, HYBRID_108B, PRO_111B, ULTRA_128B) */
        if (format != ULTRA_FORMAT_X86E8_84B && (input_size < 4096 || base_payload_bits > 256)) {
            int max_allowed_stride = is_multi_stage ? 128 : 32640;
            int stride_req_off = (format == ULTRA_FORMAT_ULTRA_128B) ? -200 : 0;
            int candidate_strides[160];
            int cand_kg[160];
            int cand_df[160];
            int num_strides = 0;
            int raw_strides[160];
            int n_raw_strides = 0;
            int dense_limit = max_allowed_stride < 128 ? max_allowed_stride : 128;
            for (int s = 1; s <= dense_limit && s < input_size; s++) {
                raw_strides[n_raw_strides++] = s;
            }
            static const int extra_strides[] = {
                132, 136, 140, 144, 150, 160, 168, 176, 180, 192, 200,
                216, 224, 240, 250, 256, 288, 300, 320, 384, 512
            };
            for (size_t idx = 0; idx < sizeof(extra_strides) / sizeof(extra_strides[0]); idx++) {
                int es = extra_strides[idx];
                if (es < input_size && es <= max_allowed_stride && n_raw_strides < 160) {
                    raw_strides[n_raw_strides++] = es;
                }
            }

            int raw_kg_arr[160];
            int raw_df_arr[160];
            #pragma omp parallel for schedule(static) if (!omp_in_parallel() && input_size >= 8192)
            for (int rsi = 0; rsi < n_raw_strides; rsi++) {
                int stride = raw_strides[rsi];
                int max_full_count = input_size - stride;
                int full_count = max_full_count > 65535 ? 65535 : max_full_count;
                if (full_count <= 0) {
                    raw_kg_arr[rsi] = -1;
                    raw_df_arr[rsi] = -1;
                    continue;
                }
                int kg_v = 0;
                find_optimal_stride_tail_count(working_input, input_size, stride, &kg_v, NULL);
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
                raw_kg_arr[rsi] = kg_v;
                raw_df_arr[rsi] = eq_dif - eq_raw;
            }
            for (int rsi = 0; rsi < n_raw_strides; rsi++) {
                int stride = raw_strides[rsi];
                int max_full_count = input_size - stride;
                int full_count = max_full_count > 65535 ? 65535 : max_full_count;
                if (full_count <= 0) {
                    continue;
                }
                int kadane_gain = raw_kg_arr[rsi];
                int df_gain = raw_df_arr[rsi];
                int is_structural_stride = (stride == 1 || stride == 2 || stride == 3 || stride == 4 ||
                                            stride == 6 || stride == 8 || stride == 9 || stride == 10 ||
                                            stride == 12 || stride == 13 || stride == 15 || stride == 16 ||
                                            stride == 18 || stride == 20 || stride == 24 || stride == 26 ||
                                            stride == 30 || stride == 32 || stride == 36 || stride == 40 ||
                                            stride == 48 || stride == 52 || stride == 60 || stride == 64 ||
                                            stride == 72 || stride == 80 || stride == 96 || stride == 108 ||
                                            stride == 120 || stride == 128 || stride == 144 || stride == 160 ||
                                            stride == 192 || stride == 240 || stride == 256 || stride == 320 ||
                                            (input_size % stride == 0));
                if ((stride > 2 && kadane_gain < 16 && df_gain < (input_size - stride) / 16) ||
                    (stride > 8 && kadane_gain < 16 && !is_structural_stride && df_gain < (input_size - stride) / 12) ||
                    (stride > 2 && input_size >= 4096 && kadane_gain < 16 && df_gain < 0)) {
                    continue;
                }
                int dominated_harmonic = 0;
                if ((stride > 16 && stride != 24 && stride != 26 && stride != 32) || (input_size >= 8192 && (stride == 10 || stride == 14))) {
                    for (int prev_i = 0; prev_i < num_strides; prev_i++) {
                        int d = candidate_strides[prev_i];
                        if (d >= 2 && stride > d && (stride % d == 0) &&
                            cand_kg[prev_i] >= 300 &&
                            kadane_gain <= cand_kg[prev_i] + (input_size / 32 + 64) &&
                            df_gain <= cand_df[prev_i] + (input_size / 32 + 64)) {
                            dominated_harmonic = 1;
                            break;
                        }
                    }
                }
                if (dominated_harmonic) {
                    continue;
                }
                candidate_strides[num_strides] = stride;
                cand_kg[num_strides] = kadane_gain;
                cand_df[num_strides] = df_gain;
                num_strides++;
            }

            int cap_s = (input_size >= 24576) ? 6 : 8;
            if (input_size >= 8192 && num_strides > cap_s) {
                for (int i = 0; i < num_strides; i++) {
                    for (int j = i + 1; j < num_strides; j++) {
                        int si_score = cand_df[i] + cand_kg[i] * 2 + (candidate_strides[i] <= 2 ? 100000 : ((candidate_strides[i] <= 4 || candidate_strides[i] == 16) ? 2000 : 0));
                        int sj_score = cand_df[j] + cand_kg[j] * 2 + (candidate_strides[j] <= 2 ? 100000 : ((candidate_strides[j] <= 4 || candidate_strides[j] == 16) ? 2000 : 0));
                        if (sj_score > si_score) {
                            int ts = candidate_strides[i]; candidate_strides[i] = candidate_strides[j]; candidate_strides[j] = ts;
                            int tk = cand_kg[i]; cand_kg[i] = cand_kg[j]; cand_kg[j] = tk;
                            int td = cand_df[i]; cand_df[i] = cand_df[j]; cand_df[j] = td;
                        }
                    }
                }
                if (num_strides > cap_s) {
                    num_strides = cap_s;
                }
            }

            int max_passes = (format == ULTRA_FORMAT_STRD_82B || is_multi_stage)
                           ? (input_size >= 16384 ? 4 : 5) : 1;
            int nested_parallel = omp_in_parallel();


            if (input_size >= 8192 && !nested_parallel) {
                UltraMetadata stage_b_cands[96];
                int stage_b_req_offs[96];
                int n_stage_b = 0;

                uint8_t *cur_buf = (uint8_t *)malloc((size_t)input_size);
                uint8_t *nxt_buf = (uint8_t *)malloc((size_t)input_size);
                assert(cur_buf != NULL && nxt_buf != NULL);

                for (int si = 0; si < num_strides; si++) {
                    int stride = candidate_strides[si];
                    int req_final_off = (format == ULTRA_FORMAT_ULTRA_128B) ? -200
                                      : is_multi_stage ? 0 : stride;
                    int max_full_count = input_size - stride;
                    int full_count = max_full_count > 65535 ? 65535 : max_full_count;

                    int kadane_gain = 0;
                    int unbiased_tail = 0;
                    int kadane_tail = find_optimal_stride_tail_count(
                        working_input, input_size, stride, &kadane_gain, &unbiased_tail
                    );

                    memcpy(cur_buf, working_input, (size_t)input_size);
                    int stride_max_passes = (stride <= 2 || si == 0) ? max_passes : ((si == 1 && stride <= 32) ? 2 : 1);
                    for (int pass = 1; pass <= stride_max_passes; pass++) {
                        if (input_size <= pass * stride) {
                            break;
                        }
                        int step_cnt = (format == ULTRA_FORMAT_STRD_82B || is_multi_stage)
                                     ? (input_size - pass * stride) : full_count;
                        if (step_cnt <= 0 || step_cnt > 65535) step_cnt = full_count;
                        apply_forward_stride_difference(cur_buf, nxt_buf, input_size, stride, step_cnt);
                        if (memcmp(nxt_buf, cur_buf, (size_t)input_size) == 0 &&
                            (pass > 1 || (format != ULTRA_FORMAT_STRD_80B && format != ULTRA_FORMAT_STRD_82B) || stride > 1)) {
                            break;
                        }
                        if (pass >= 2) {
                            int eq_c = 0, eq_n = 0;
                            for (int k = pass * stride + 1; k < input_size; k++) {
                                if (cur_buf[k] == 0 || cur_buf[k] == 1 || cur_buf[k] == 255 || cur_buf[k] == cur_buf[k - 1] || cur_buf[k] == cur_buf[k - stride]) eq_c++;
                                if (nxt_buf[k] == 0 || nxt_buf[k] == 1 || nxt_buf[k] == 255 || nxt_buf[k] == nxt_buf[k - 1] || nxt_buf[k] == nxt_buf[k - stride]) eq_n++;
                            }
                            if (eq_n <= eq_c) {
                                break;
                            }
                        }
                        memcpy(cur_buf, nxt_buf, (size_t)input_size);

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
                            if (format == ULTRA_FORMAT_STRD_82B || is_multi_stage) {
                                int hc_p = is_multi_stage ? (input_size - (pass - p_idx) * stride) : (input_size - (p_idx + 1) * stride);
                                meta_p.filter_counts[p_idx] = (hc_p > 0 && hc_p <= 65535) ? hc_p : full_count;
                            } else {
                                meta_p.filter_counts[p_idx] = full_count;
                            }
                        }
                        if (n_stage_b < 96) {
                            stage_b_req_offs[n_stage_b] = req_final_off;
                            stage_b_cands[n_stage_b++] = meta_p;
                        }
                    }

                    int min_pw_gap = (full_count / 4) > 32 ? (full_count / 4) : 32;
                    int min_pw_tail = (stride + 32) > 64 ? (stride + 32) : 64;
                    int has_q1_stride_corr = 0;
                    if (stride <= 2) {
                        int q1_s = input_size / 4;
                        int q1_e = input_size / 2;
                        int eq_q1 = 0;
                        for (int k = q1_s + 2 * stride; k < q1_e; k++) {
                            uint8_t d = (uint8_t)(working_input[k] - working_input[k - stride]);
                            uint8_t ds = (uint8_t)(working_input[k - stride] - working_input[k - 2 * stride]);
                            if (d == 0 || d == 1 || d == 255 || d == ds) eq_q1++;
                        }
                        if (eq_q1 >= (q1_e - q1_s) / 4) {
                            has_q1_stride_corr = 1;
                        }
                    }
                    int allow_piecewise = ((si < 4 && kadane_gain >= 64) || has_q1_stride_corr) &&
                                          ((kadane_tail >= min_pw_tail && kadane_tail < full_count - min_pw_gap) ||
                                           (unbiased_tail >= min_pw_tail && unbiased_tail < full_count - min_pw_gap) ||
                                           has_q1_stride_corr);
                    int cand_piecewise[12];
                    int num_piecewise = 0;
                    int half_count = input_size / 2;
                    int three_q_count = ((3 * input_size) / 4 - stride);
                    if (three_q_count > 65535) {
                        three_q_count = 65535;
                    }
                    if (allow_piecewise && half_count >= min_pw_tail && half_count < full_count) {
                        cand_piecewise[num_piecewise++] = half_count;
                    }
                    if (allow_piecewise && has_q1_stride_corr && three_q_count >= min_pw_tail && three_q_count < full_count) {
                        cand_piecewise[num_piecewise++] = three_q_count;
                    }
                    int k_cands[8];
                    int nk = 0;
                    if (allow_piecewise && kadane_tail >= min_pw_tail && kadane_tail < full_count - min_pw_gap) {
                        k_cands[nk++] = kadane_tail;
                        if (stride > 1) {
                            int aligned = ((kadane_tail + stride / 2) / stride) * stride;
                            if (aligned >= min_pw_tail && aligned < full_count) {
                                k_cands[nk++] = aligned;
                            }
                        }
                    }
                    if (allow_piecewise && unbiased_tail >= min_pw_tail && unbiased_tail < full_count - min_pw_gap) {
                        if (nk < 8) k_cands[nk++] = unbiased_tail;
                        if (stride > 1 && nk < 8) {
                            int aligned_u = ((unbiased_tail + stride / 2) / stride) * stride;
                            if (aligned_u >= min_pw_tail && aligned_u < full_count) {
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
                        if (!dup && num_piecewise < 12) {
                            cand_piecewise[num_piecewise++] = tc;
                        }
                    }

                    for (int pi = 0; pi < num_piecewise; pi++) {
                        int tail_c = cand_piecewise[pi];
                        apply_forward_stride_difference(working_input, nxt_buf, input_size, stride, tail_c);
                        if (memcmp(nxt_buf, working_input, (size_t)input_size) == 0) {
                            continue;
                        }
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
                        if (n_stage_b < 96) {
                            stage_b_req_offs[n_stage_b] = req_final_off;
                            stage_b_cands[n_stage_b++] = meta_h;
                        }

                        if (max_passes > 1 && stride == 2 && tail_c == three_q_count) {
                            memcpy(cur_buf, nxt_buf, (size_t)input_size);
                            int step_counts[4] = {three_q_count, 0, 0, 0};
                            for (int pass = 2; pass <= 4; pass++) {
                                int hc_sub = half_count - pass * stride;
                                if (hc_sub <= stride || hc_sub > 65535) {
                                    break;
                                }
                                apply_forward_stride_difference(cur_buf, nxt_buf, input_size, stride, hc_sub);
                                if (memcmp(nxt_buf, cur_buf, (size_t)input_size) == 0) {
                                    break;
                                }
                                int eq_c = 0, eq_n = 0;
                                for (int k = input_size - hc_sub + stride; k < input_size; k++) {
                                    if (cur_buf[k] == 0 || cur_buf[k] == 1 || cur_buf[k] == 255 || cur_buf[k] == cur_buf[k - 1] || cur_buf[k] == cur_buf[k - stride]) eq_c++;
                                    if (nxt_buf[k] == 0 || nxt_buf[k] == 1 || nxt_buf[k] == 255 || nxt_buf[k] == nxt_buf[k - 1] || nxt_buf[k] == nxt_buf[k - stride]) eq_n++;
                                }
                                if (eq_n <= eq_c) {
                                    break;
                                }
                                memcpy(cur_buf, nxt_buf, (size_t)input_size);
                                step_counts[pass - 1] = hc_sub;
                                UltraMetadata meta_st;
                                memset(&meta_st, 0, sizeof(meta_st));
                                meta_st.format = format;
                                meta_st.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                meta_st.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                     ? detect_buffer_literal_bits(cur_buf, input_size, NULL) : 8;
                                meta_st.filter_passes = pass;
                                meta_st.filter_stride = stride;
                                for (int p_idx = 0; p_idx < pass; p_idx++) {
                                    meta_st.filter_strides[p_idx] = stride;
                                    meta_st.filter_counts[p_idx] = is_multi_stage
                                                                 ? step_counts[pass - 1 - p_idx]
                                                                 : step_counts[p_idx];
                                }
                                if (n_stage_b < 96) {
                                    stage_b_req_offs[n_stage_b] = req_final_off;
                                    stage_b_cands[n_stage_b++] = meta_st;
                                }
                            }
                            if (tail_c == kadane_tail) {
                                apply_forward_stride_difference(working_input, nxt_buf, input_size, stride, tail_c);
                            }
                        }

                        if (max_passes > 1 && stride <= 16 && tail_c == kadane_tail) {
                            memcpy(cur_buf, nxt_buf, (size_t)input_size);
                            for (int pass = 2; pass <= 3; pass++) {
                                if (tail_c <= pass * stride) {
                                    break;
                                }
                                apply_forward_stride_difference(cur_buf, nxt_buf, input_size, stride, tail_c);
                                if (memcmp(nxt_buf, cur_buf, (size_t)input_size) == 0) {
                                    break;
                                }
                                int eq_c = 0, eq_n = 0;
                                for (int k = input_size - tail_c + stride; k < input_size; k++) {
                                    if (cur_buf[k] == 0 || cur_buf[k] == 1 || cur_buf[k] == 255 || cur_buf[k] == cur_buf[k - 1] || cur_buf[k] == cur_buf[k - stride]) eq_c++;
                                    if (nxt_buf[k] == 0 || nxt_buf[k] == 1 || nxt_buf[k] == 255 || nxt_buf[k] == nxt_buf[k - 1] || nxt_buf[k] == nxt_buf[k - stride]) eq_n++;
                                }
                                if (eq_n <= eq_c) {
                                    break;
                                }
                                memcpy(cur_buf, nxt_buf, (size_t)input_size);
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
                                if (n_stage_b < 96) {
                                    stage_b_req_offs[n_stage_b] = req_final_off;
                                    stage_b_cands[n_stage_b++] = meta_mp;
                                }
                            }
                        }
                    }
                }
                free(cur_buf);
                free(nxt_buf);

                for (int sbi = 0; sbi < n_stage_b && n_uni_cands < 320; sbi++) { uni_req_offs[n_uni_cands] = stage_b_req_offs[sbi]; uni_cands[n_uni_cands++] = stage_b_cands[sbi]; }
            } else {
                #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
                for (int si = 0; si < num_strides; si++) {
                    int stride = candidate_strides[si];
                    int req_final_off = (format == ULTRA_FORMAT_ULTRA_128B) ? -200
                                      : is_multi_stage ? 0 : stride;
                    int max_full_count = input_size - stride;
                    int full_count = max_full_count > 65535 ? 65535 : max_full_count;

                    int kadane_gain = 0;
                    int unbiased_tail = 0;
                    int kadane_tail = find_optimal_stride_tail_count(
                        working_input, input_size, stride, &kadane_gain, &unbiased_tail
                    );

                    uint8_t *cur_buf = (uint8_t *)malloc((size_t)input_size);
                    uint8_t *nxt_buf = (uint8_t *)malloc((size_t)input_size);
                    assert(cur_buf != NULL && nxt_buf != NULL);
                    memcpy(cur_buf, working_input, (size_t)input_size);

                    int prev_pass_bits = COST_INFINITY;
                    int pass1_full_bits = COST_INFINITY;
                    int pass2_full_bits = COST_INFINITY;

                    int stride_max_passes = (stride <= 32) ? max_passes : 1;
                    for (int pass = 1; pass <= stride_max_passes; pass++) {
                        if (input_size <= pass * stride) {
                            break;
                        }
                        int step_cnt = (input_size >= 8192 && (format == ULTRA_FORMAT_STRD_82B || is_multi_stage))
                                     ? (input_size - pass * stride) : full_count;
                        if (step_cnt <= 0 || step_cnt > 65535) step_cnt = full_count;
                        apply_forward_stride_difference(cur_buf, nxt_buf, input_size, stride, step_cnt);
                        if (memcmp(nxt_buf, cur_buf, (size_t)input_size) == 0 &&
                            (pass > 1 || (format != ULTRA_FORMAT_STRD_80B && format != ULTRA_FORMAT_STRD_82B) || stride > 1)) {
                            break;
                        }
                        if (input_size >= 8192 && pass >= 2) {
                            int eq_c = 0, eq_n = 0;
                            for (int k = pass * stride + 1; k < input_size; k++) {
                                if (cur_buf[k] == 0 || cur_buf[k] == 1 || cur_buf[k] == 255 || cur_buf[k] == cur_buf[k - 1] || cur_buf[k] == cur_buf[k - stride]) eq_c++;
                                if (nxt_buf[k] == 0 || nxt_buf[k] == 1 || nxt_buf[k] == 255 || nxt_buf[k] == nxt_buf[k - 1] || nxt_buf[k] == nxt_buf[k - stride]) eq_n++;
                            }
                            if (eq_n <= eq_c) {
                                break;
                            }
                        }
                        memcpy(cur_buf, nxt_buf, (size_t)input_size);

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
                            if (input_size >= 8192 && (format == ULTRA_FORMAT_STRD_82B || is_multi_stage)) {
                                int hc_p = is_multi_stage ? (input_size - (pass - p_idx) * stride) : (input_size - (p_idx + 1) * stride);
                                meta_p.filter_counts[p_idx] = (hc_p > 0 && hc_p <= 65535) ? hc_p : full_count;
                            } else {
                                meta_p.filter_counts[p_idx] = full_count;
                            }
                        }
                        int pass_bits = evaluate_candidate_meta(
                            cur_buf, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                            req_final_off, &meta_p, &best_size, &best_meta, NULL
                        );
                        if (pass_bits > 0) {
                            if (pass == 1) pass1_full_bits = pass_bits;
                            if (pass == 2) pass2_full_bits = pass_bits;
                            if (pass_bits > base_payload_bits + (input_size >= 8192 ? 256 : 1024) ||
                                (pass == 1 && pass_bits > prev_pass_bits) ||
                                (pass >= 2 && pass_bits >= prev_pass_bits)) {
                                break;
                            }
                            prev_pass_bits = pass_bits;
                        } else {
                            break;
                        }
                    }

                    /* Harmonic multi-pass counts (count_p = input_size - pass * stride) avoiding boundary artifacts */
                    if (input_size < 8192 && max_passes > 1 && stride <= 32 && pass2_full_bits < COST_INFINITY && pass2_full_bits <= pass1_full_bits + 64) {
                        memcpy(cur_buf, working_input, (size_t)input_size);
                        int harm_counts[ULTRA_MAX_FILTER_PASSES] = {0};
                        int prev_harm_bits = pass1_full_bits;
                        for (int pass = 1; pass <= max_passes; pass++) {
                            int hc = input_size - pass * stride;
                            if (hc <= 0 || hc > 65535) {
                                break;
                            }
                            apply_forward_stride_difference(cur_buf, nxt_buf, input_size, stride, hc);
                            if (memcmp(nxt_buf, cur_buf, (size_t)input_size) == 0) {
                                break;
                            }
                            memcpy(cur_buf, nxt_buf, (size_t)input_size);
                            harm_counts[pass - 1] = hc;
                            if (pass >= 2) {
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
                                int hm_bits = evaluate_candidate_meta(
                                    cur_buf, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                    req_final_off, &meta_hm, &best_size, &best_meta, NULL
                                );
                                if (hm_bits <= 0 || hm_bits >= prev_harm_bits) {
                                    break;
                                }
                                prev_harm_bits = hm_bits;
                            }
                        }
                    }

                    /* Evaluate piecewise half-tail, three-quarter tail, Kadane-optimal tail, and unbiased Kadane tail */
                    int min_pw_gap = (full_count / 4) > 32 ? (full_count / 4) : 32;
                    int min_pw_tail = (stride + 32) > 64 ? (stride + 32) : 64;
                    int has_q1_stride_corr = 0;
                    if (stride <= 2 && input_size >= 512) {
                        int q1_s = input_size / 4;
                        int q1_e = input_size / 2;
                        int eq_q1 = 0;
                        for (int k = q1_s + 2 * stride; k < q1_e; k++) {
                            uint8_t d = (uint8_t)(working_input[k] - working_input[k - stride]);
                            uint8_t ds = (uint8_t)(working_input[k - stride] - working_input[k - 2 * stride]);
                            if (d == 0 || d == 1 || d == 255 || d == ds) eq_q1++;
                        }
                        if (eq_q1 >= (q1_e - q1_s) / 4) {
                            has_q1_stride_corr = 1;
                        }
                    }
                    int allow_piecewise = (pass1_full_bits * 2 >= base_payload_bits) &&
                                          (input_size < 8192 ? (kadane_gain >= 32 || has_q1_stride_corr)
                                                             : ((si < 4 && kadane_gain >= 64) || has_q1_stride_corr)) &&
                                          ((kadane_tail >= min_pw_tail && kadane_tail < full_count - min_pw_gap) ||
                                           (unbiased_tail >= min_pw_tail && unbiased_tail < full_count - min_pw_gap) ||
                                           has_q1_stride_corr);
                    int cand_piecewise[12];
                    int num_piecewise = 0;
                    int half_count = input_size / 2;
                    int three_q_count = ((3 * input_size) / 4 - stride);
                    if (three_q_count > 65535) {
                        three_q_count = 65535;
                    }
                    if (allow_piecewise && !nested_parallel && input_size >= 128 &&
                        half_count >= min_pw_tail && half_count < full_count) {
                        cand_piecewise[num_piecewise++] = half_count;
                    }
                    if (allow_piecewise && !nested_parallel && has_q1_stride_corr &&
                        three_q_count >= min_pw_tail && three_q_count < full_count) {
                        cand_piecewise[num_piecewise++] = three_q_count;
                    }
                    int k_cands[8];
                    int nk = 0;
                    if (allow_piecewise && kadane_tail >= min_pw_tail && kadane_tail < full_count - min_pw_gap) {
                        k_cands[nk++] = kadane_tail;
                        if (stride > 1) {
                            int aligned = ((kadane_tail + stride / 2) / stride) * stride;
                            if (aligned >= min_pw_tail && aligned < full_count) {
                                k_cands[nk++] = aligned;
                            }
                        }
                    }
                    if (allow_piecewise && unbiased_tail >= min_pw_tail && unbiased_tail < full_count - min_pw_gap) {
                        if (nk < 8) k_cands[nk++] = unbiased_tail;
                        if (stride > 1 && nk < 8) {
                            int aligned_u = ((unbiased_tail + stride / 2) / stride) * stride;
                            if (aligned_u >= min_pw_tail && aligned_u < full_count) {
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
                        if (!dup && num_piecewise < 12) {
                            cand_piecewise[num_piecewise++] = tc;
                        }
                    }

                    for (int pi = 0; pi < num_piecewise; pi++) {
                        int tail_c = cand_piecewise[pi];
                        apply_forward_stride_difference(working_input, nxt_buf, input_size, stride, tail_c);
                        if (memcmp(nxt_buf, working_input, (size_t)input_size) == 0) {
                            continue;
                        }
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
                        int pw_pass1_bits = evaluate_candidate_meta(
                            nxt_buf, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                            req_final_off, &meta_h, &best_size, &best_meta, NULL
                        );
                        if (pw_pass1_bits <= 0) {
                            pw_pass1_bits = COST_INFINITY;
                        }

                        /* Stepped-tail multi-pass byte-stride (1 pass on 3*N/4 + 1..3 passes on N/2) for STRD_82B & multi-stage */
                        if (max_passes > 1 && stride == 2 && tail_c == three_q_count) {
                            memcpy(cur_buf, nxt_buf, (size_t)input_size);
                            int step_counts[4] = {three_q_count, 0, 0, 0};
                            int prev_st_bits = pw_pass1_bits;
                            for (int pass = 2; pass <= 4; pass++) {
                                int hc_sub = half_count - pass * stride;
                                if (hc_sub <= stride || hc_sub > 65535) {
                                    break;
                                }
                                apply_forward_stride_difference(cur_buf, nxt_buf, input_size, stride, hc_sub);
                                if (memcmp(nxt_buf, cur_buf, (size_t)input_size) == 0) {
                                    break;
                                }
                                memcpy(cur_buf, nxt_buf, (size_t)input_size);
                                step_counts[pass - 1] = hc_sub;
                                UltraMetadata meta_st;
                                memset(&meta_st, 0, sizeof(meta_st));
                                meta_st.format = format;
                                meta_st.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                meta_st.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                     ? detect_buffer_literal_bits(cur_buf, input_size, NULL) : 8;
                                meta_st.filter_passes = pass;
                                meta_st.filter_stride = stride;
                                for (int p_idx = 0; p_idx < pass; p_idx++) {
                                    meta_st.filter_strides[p_idx] = stride;
                                    meta_st.filter_counts[p_idx] = is_multi_stage
                                                                 ? step_counts[pass - 1 - p_idx]
                                                                 : step_counts[p_idx];
                                }
                                int st_bits = evaluate_candidate_meta(
                                    cur_buf, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                    req_final_off, &meta_st, &best_size, &best_meta, NULL
                                );
                                if (st_bits <= 0 || (input_size >= 8192 && st_bits >= prev_st_bits)) {
                                    break;
                                }
                                prev_st_bits = st_bits;
                            }
                            /* Restore nxt_buf if needed below */
                            if (tail_c == kadane_tail) {
                                apply_forward_stride_difference(working_input, nxt_buf, input_size, stride, tail_c);
                            }
                        }

                        /* Multi-pass piecewise tail integration for STRD_82B, HYBRID_108B, PRO_111B, and ULTRA_128B */
                        if (max_passes > 1 && stride <= 16 && tail_c == kadane_tail && pw_pass1_bits <= base_payload_bits + 256) {
                            memcpy(cur_buf, nxt_buf, (size_t)input_size);
                            int prev_pw_bits = pw_pass1_bits;
                            for (int pass = 2; pass <= 3; pass++) {
                                if (tail_c <= pass * stride) {
                                    break;
                                }
                                apply_forward_stride_difference(cur_buf, nxt_buf, input_size, stride, tail_c);
                                if (memcmp(nxt_buf, cur_buf, (size_t)input_size) == 0) {
                                    break;
                                }
                                memcpy(cur_buf, nxt_buf, (size_t)input_size);
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
                                int mp_bits = evaluate_candidate_meta(
                                    cur_buf, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                    req_final_off, &meta_mp, &best_size, &best_meta, NULL
                                );
                                if (mp_bits <= 0 || mp_bits >= prev_pw_bits) {
                                    break;
                                }
                                prev_pw_bits = mp_bits;
                            }
                        }
                    }

                    free(cur_buf);
                    free(nxt_buf);
                }
            }

            /* C0. PRO_111B & ULTRA_128B Native 16-Bit Word-Stride-Delta Search (eliminates carry-ripple on 16-bit PCM/Coord/Ptr tables) */
            if (is_word_capable && input_size >= 4 && best_meta.compressed_bits > 64) {
                static const int word_strides[] = {
                    2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                    32, 36, 40, 48, 52, 60, 64, 72, 80, 96, 100, 112, 120, 128
                };
                int n_ws_all = (int)(sizeof(word_strides) / sizeof(word_strides[0]));
                int cand_ws[30];
                int cand_wg[30];
                int n_ws = 0;
                for (int wi = 0; wi < n_ws_all; wi++) {
                    int wstride = word_strides[wi];
                    if (wstride + 2 > input_size) continue;
                    int max_wcount = ((input_size - wstride) & ~1);
                    if (max_wcount < 2 || max_wcount > 65534) continue;
                    int eq_w = 0, eq_w3 = 0, eq_raw_w = 0;
                    for (int k = wstride; k + 1 < input_size; k += 2) {
                        uint16_t wcur_v = (uint16_t)(working_input[k] | (working_input[k + 1] << 8));
                        uint16_t wprv_v = (uint16_t)(working_input[k - wstride] | (working_input[k + 1 - wstride] << 8));
                        uint16_t wdif_v = (uint16_t)(wcur_v - wprv_v);
                        uint8_t d_lo = (uint8_t)(wdif_v & 0xFFu);
                        uint8_t d_hi = (uint8_t)(wdif_v >> 8);
                        if (working_input[k] == 0 || working_input[k] == working_input[k - wstride] ||
                            working_input[k] == working_input[k - 1]) eq_raw_w++;
                        if (working_input[k + 1] == 0 || working_input[k + 1] == 0xFFu ||
                            working_input[k + 1] == working_input[k + 1 - wstride]) eq_raw_w++;
                        if (d_lo == 0) eq_w++;
                        if (d_hi == 0 || d_hi == 0xFFu) eq_w++;
                        if (wstride <= 4 && k >= 3 * wstride) {
                            uint16_t w2 = (uint16_t)(working_input[k - 2 * wstride] | (working_input[k + 1 - 2 * wstride] << 8));
                            uint16_t w3 = (uint16_t)(working_input[k - 3 * wstride] | (working_input[k + 1 - 3 * wstride] << 8));
                            uint16_t wd3 = (uint16_t)(wcur_v - 3u * wprv_v + 3u * w2 - w3);
                            uint8_t d3_lo = (uint8_t)(wd3 & 0xFFu);
                            uint8_t d3_hi = (uint8_t)(wd3 >> 8);
                            if (d3_lo == 0) eq_w3++;
                            if (d3_hi == 0 || d3_hi == 0xFFu) eq_w3++;
                        }
                    }
                    int best_eq_w = (eq_w3 > eq_w) ? eq_w3 : eq_w;
                    if ((wstride <= 4 && best_eq_w <= eq_raw_w + (input_size >= 8192 ? (input_size - wstride) / 24 : 0)) ||
                        (wstride > 4 && best_eq_w <= eq_raw_w + (input_size - wstride) / (input_size >= 8192 ? 16 : 32))) {
                        continue;
                    }
                    cand_ws[n_ws] = wstride;
                    cand_wg[n_ws] = (wstride == 2 && input_size >= 8192) ? (best_eq_w - eq_raw_w + input_size) : (best_eq_w - eq_raw_w);
                    n_ws++;
                }
                if (input_size >= 8192 && n_ws > 4) {
                    for (int a = 0; a < 4; a++) {
                        for (int b = a + 1; b < n_ws; b++) {
                            if (cand_wg[b] > cand_wg[a]) {
                                int ts = cand_ws[a]; cand_ws[a] = cand_ws[b]; cand_ws[b] = ts;
                                int tg = cand_wg[a]; cand_wg[a] = cand_wg[b]; cand_wg[b] = tg;
                            }
                        }
                    }
                    n_ws = 4;
                }
                if (input_size >= 8192 && !nested_parallel && n_ws > 0) {
                    UltraMetadata stage_c0_cands[64];
                    int n_stage_c0 = 0;
                    uint8_t *wcur = (uint8_t *)malloc((size_t)input_size);
                    uint8_t *wnxt = (uint8_t *)malloc((size_t)input_size);
                    assert(wcur != NULL && wnxt != NULL);

                    for (int wi = 0; wi < n_ws; wi++) {
                        int wstride = cand_ws[wi];
                        int max_wcount = ((input_size - wstride) & ~1);
                        int eq_raw_hi = 0;
                        for (int k = wstride; k + 1 < input_size; k += 2) {
                            if (working_input[k + 1] == 0 || working_input[k + 1] == 0xFFu) eq_raw_hi++;
                        }

                        memcpy(wcur, working_input, (size_t)input_size);
                        int max_wpasses = (wi <= 1 || wstride <= 8) ? 4 : 2;
                        int prev_eq_wb = 0;
                        for (int k = wstride; k + 1 < input_size; k += 2) {
                            if (working_input[k] == 0 || working_input[k] == working_input[k - wstride]) prev_eq_wb++;
                            if (working_input[k + 1] == 0 || working_input[k + 1] == 0xFFu || working_input[k + 1] == working_input[k + 1 - wstride]) prev_eq_wb++;
                        }
                        for (int pass = 1; pass <= max_wpasses; pass++) {
                            if (input_size <= pass * wstride + 2) {
                                break;
                            }
                            int cur_wc = (input_size - pass * wstride) & ~1;
                            if (cur_wc < 2 || cur_wc > 65534) cur_wc = max_wcount;
                            apply_forward_word16_stride_difference(wcur, wnxt, input_size, wstride, cur_wc);
                            if (memcmp(wnxt, wcur, (size_t)input_size) == 0) {
                                break;
                            }
                            memcpy(wcur, wnxt, (size_t)input_size);
                            int eq_cur_hi = 0, cur_eq_wb = 0;
                            for (int k = wstride; k + 1 < input_size; k += 2) {
                                if (wcur[k] == 0 || wcur[k] == wcur[k - wstride]) cur_eq_wb++;
                                if (wcur[k + 1] == 0 || wcur[k + 1] == 0xFFu) { eq_cur_hi++; cur_eq_wb++; }
                                else if (wcur[k + 1] == wcur[k + 1 - wstride]) cur_eq_wb++;
                            }
                            if (pass >= 2 && wstride > 2 && cur_eq_wb <= prev_eq_wb) {
                                break;
                            }
                            prev_eq_wb = cur_eq_wb;
                            if (wstride > 2 && eq_cur_hi <= eq_raw_hi + (input_size / 32)) {
                                continue;
                            }
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
                                int wc_p = (input_size - (pass - pk) * wstride) & ~1;
                                meta_w.filter_counts[pk] = (wc_p >= 2 && wc_p <= 65534) ? wc_p : max_wcount;
                                meta_w.filter_word_modes[pk] = 1;
                            }
                            if (n_stage_c0 < 64) {
                                stage_c0_cands[n_stage_c0++] = meta_w;
                            }
                        }

                        if (wstride <= 32) {
                            memcpy(wcur, working_input, (size_t)input_size);
                            int max_hyb_wpass = (wstride <= 2) ? 3 : 2;
                            for (int wpass = 1; wpass <= max_hyb_wpass; wpass++) {
                                int whc = (input_size - wpass * wstride) & ~1;
                                if (whc < 2 || whc > 65534) break;
                                apply_forward_word16_stride_difference(wcur, wnxt, input_size, wstride, whc);
                                if (memcmp(wnxt, wcur, (size_t)input_size) == 0) break;
                                memcpy(wcur, wnxt, (size_t)input_size);
                                int eq_cur_hi = 0;
                                for (int k = wpass * wstride; k + 1 < input_size; k += 2) {
                                    if (wcur[k + 1] == 0 || wcur[k + 1] == 0xFFu) eq_cur_hi++;
                                }
                                if (eq_cur_hi <= eq_raw_hi + (input_size / 32)) {
                                    continue;
                                }

                                int b_strides[3] = {1, 2, wstride};
                                for (int bi = 0; bi < 3; bi++) {
                                    int bs = b_strides[bi];
                                    if (bi == 2 && (bs == 1 || bs == 2)) continue;
                                    int bc = (input_size - bs) > 65535 ? 65535 : (input_size - bs);
                                    if (bc <= 0) continue;
                                    apply_forward_stride_difference(wcur, wnxt, input_size, bs, bc);
                                    if (memcmp(wnxt, wcur, (size_t)input_size) == 0) continue;
                                    int eq_wc = 0, eq_wn = 0;
                                    for (int k = bs + 1; k < input_size; k++) {
                                        if (wcur[k] == 0 || wcur[k] == wcur[k - 1] || wcur[k] == wcur[k - bs]) eq_wc++;
                                        if (wnxt[k] == 0 || wnxt[k] == wnxt[k - 1] || wnxt[k] == wnxt[k - bs]) eq_wn++;
                                    }
                                    if (eq_wn <= eq_wc + (input_size - bs) / 16) continue;
                                    UltraMetadata meta_wb;
                                    memset(&meta_wb, 0, sizeof(meta_wb));
                                    meta_wb.format = format;
                                    meta_wb.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                    meta_wb.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                         ? detect_buffer_literal_bits(wnxt, input_size, NULL) : 8;
                                    meta_wb.filter_passes = wpass + 1;
                                    meta_wb.filter_stride = bs;
                                    meta_wb.filter_strides[0] = bs;
                                    meta_wb.filter_counts[0] = bc;
                                    meta_wb.filter_word_modes[0] = 0;
                                    for (int pk = 0; pk < wpass; pk++) {
                                        meta_wb.filter_strides[1 + pk] = wstride;
                                        int wc_pk = (input_size - (wpass - pk) * wstride) & ~1;
                                        meta_wb.filter_counts[1 + pk] = wc_pk > 65534 ? 65534 : wc_pk;
                                        meta_wb.filter_word_modes[1 + pk] = 1;
                                    }
                                    if (n_stage_c0 < 64) {
                                        stage_c0_cands[n_stage_c0++] = meta_wb;
                                    }
                                }
                            }
                        }
                    }
                    free(wcur);
                    free(wnxt);

                    for (int c0i = 0; c0i < n_stage_c0 && n_uni_cands < 320; c0i++) { uni_req_offs[n_uni_cands] = stride_req_off; uni_cands[n_uni_cands++] = stage_c0_cands[c0i]; }
                } else {
                    #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
                    for (int wi = 0; wi < n_ws; wi++) {
                        int wstride = cand_ws[wi];
                        int max_wcount = ((input_size - wstride) & ~1);
                        uint8_t *wcur = (uint8_t *)malloc((size_t)input_size);
                        uint8_t *wnxt = (uint8_t *)malloc((size_t)input_size);
                        assert(wcur != NULL && wnxt != NULL);

                        int eq_raw_hi = 0;
                        for (int k = wstride; k + 1 < input_size; k += 2) {
                            if (working_input[k + 1] == 0 || working_input[k + 1] == 0xFFu) eq_raw_hi++;
                        }

                        /* 1) Full-buffer 1..4 pass 16-bit Word-Stride-Delta */
                        memcpy(wcur, working_input, (size_t)input_size);
                        int prev_wbits = COST_INFINITY;
                        int wpass1_bits = COST_INFINITY;
                        int wpass2_bits = COST_INFINITY;
                        int max_wpasses = 4;
                        for (int pass = 1; pass <= max_wpasses; pass++) {
                            if (input_size <= pass * wstride + 2) {
                                break;
                            }
                            int cur_wc = (input_size >= 8192) ? ((input_size - pass * wstride) & ~1) : max_wcount;
                            if (cur_wc < 2 || cur_wc > 65534) cur_wc = max_wcount;
                            apply_forward_word16_stride_difference(wcur, wnxt, input_size, wstride, cur_wc);
                            if (memcmp(wnxt, wcur, (size_t)input_size) == 0) {
                                break;
                            }
                            memcpy(wcur, wnxt, (size_t)input_size);
                            if (input_size >= 8192 && pass <= 2) {
                                int eq_cur_hi = 0;
                                for (int k = wstride; k + 1 < input_size; k += 2) {
                                    if (wcur[k + 1] == 0 || wcur[k + 1] == 0xFFu) eq_cur_hi++;
                                }
                                if (eq_cur_hi <= eq_raw_hi + (input_size / 32)) {
                                    continue;
                                }
                            }
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
                                int wc_p = (input_size >= 8192) ? ((input_size - (pass - pk) * wstride) & ~1) : max_wcount;
                                meta_w.filter_counts[pk] = (wc_p >= 2 && wc_p <= 65534) ? wc_p : max_wcount;
                                meta_w.filter_word_modes[pk] = 1;
                            }
                            int wbits = evaluate_candidate_meta(
                                wcur, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                stride_req_off, &meta_w, &best_size, &best_meta, NULL
                            );
                            if (wbits > 0) {
                                if (pass == 1) {
                                    wpass1_bits = wbits;
                                } else if (pass == 2) {
                                    wpass2_bits = wbits;
                                }
                                if ((wstride > 4 &&
                                     (wbits > base_payload_bits + (input_size >= 8192 ? 512 : 1024) ||
                                      (pass >= 2 && wbits >= prev_wbits))) ||
                                    (input_size >= 8192 &&
                                     ((pass == 1 && wbits > base_payload_bits + input_size * 2) ||
                                      (pass >= 2 && wbits > prev_wbits + 512)))) {
                                    break;
                                }
                                if (wbits < prev_wbits) {
                                    prev_wbits = wbits;
                                }
                            } else {
                                break;
                            }
                        }

                        /* 2) Harmonic 16-bit Word-Stride-Delta (wcount_p = ((input_size - pass * wstride) & ~1)) */
                        if (input_size < 8192 && (wstride <= 8 || (wstride <= 32 && wpass2_bits < COST_INFINITY && wpass2_bits <= wpass1_bits + 64))) {
                            memcpy(wcur, working_input, (size_t)input_size);
                            int wharm_counts[4] = {0};
                            int prev_wh_bits = wpass1_bits;
                            for (int pass = 1; pass <= max_wpasses; pass++) {
                                int whc = (input_size - pass * wstride) & ~1;
                                if (whc < 2 || whc > 65534) {
                                    break;
                                }
                                apply_forward_word16_stride_difference(wcur, wnxt, input_size, wstride, whc);
                                if (memcmp(wnxt, wcur, (size_t)input_size) == 0) {
                                    break;
                                }
                                memcpy(wcur, wnxt, (size_t)input_size);
                                wharm_counts[pass - 1] = whc;
                                if (pass >= 2) {
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
                                    int wh_bits = evaluate_candidate_meta(
                                        wcur, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                        stride_req_off, &meta_wh, &best_size, &best_meta, NULL
                                    );
                                    if (wh_bits <= 0) {
                                        break;
                                    }
                                    if (wstride > 4 && wh_bits >= prev_wh_bits) {
                                        break;
                                    }
                                    if (wh_bits < prev_wh_bits) {
                                        prev_wh_bits = wh_bits;
                                    }
                                }
                            }
                        }

                        /* 3) Hybrid 16-bit Word-Stride (Order 1..4) followed by 8-bit Byte-Stride (stride=1, 2, or wstride) */
                        int max_hyb_ws = 32;
                        if (wstride <= 8 || (wstride <= max_hyb_ws && prev_wbits < base_payload_bits + 512)) {
                            UltraMetadata local_wb_cands[16];
                            int n_local_wb = 0;
                            memcpy(wcur, working_input, (size_t)input_size);
                            int max_hyb_wpass = (wstride <= 4) ? 4 : 3;
                            for (int wpass = 1; wpass <= max_hyb_wpass; wpass++) {
                                if (wstride > 4 && wpass >= 2 && wpass2_bits > wpass1_bits + 256) break;
                                int whc = (input_size - wpass * wstride) & ~1;
                                if (whc < 2 || whc > 65534) break;
                                apply_forward_word16_stride_difference(wcur, wnxt, input_size, wstride, whc);
                                if (memcmp(wnxt, wcur, (size_t)input_size) == 0) break;
                                memcpy(wcur, wnxt, (size_t)input_size);

                                int b_strides[3] = {1, 2, wstride};
                                for (int bi = 0; bi < 3; bi++) {
                                    int bs = b_strides[bi];
                                    if (bi == 2 && (bs == 1 || bs == 2)) continue;
                                    int bc = (input_size - bs) > 65535 ? 65535 : (input_size - bs);
                                    if (bc <= 0) continue;
                                    apply_forward_stride_difference(wcur, wnxt, input_size, bs, bc);
                                    if (memcmp(wnxt, wcur, (size_t)input_size) == 0) continue;
                                    int eq_wc = 0, eq_wn = 0;
                                    for (int k = bs + 1; k < input_size; k++) {
                                        if (wcur[k] == 0 || wcur[k] == wcur[k - 1] || wcur[k] == wcur[k - bs]) eq_wc++;
                                        if (wnxt[k] == 0 || wnxt[k] == wnxt[k - 1] || wnxt[k] == wnxt[k - bs]) eq_wn++;
                                    }
                                    if (eq_wn <= eq_wc) continue;
                                    UltraMetadata meta_wb;
                                    memset(&meta_wb, 0, sizeof(meta_wb));
                                    meta_wb.format = format;
                                    meta_wb.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                    meta_wb.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                         ? detect_buffer_literal_bits(wnxt, input_size, NULL) : 8;
                                    meta_wb.filter_passes = wpass + 1;
                                    meta_wb.filter_stride = bs;
                                    meta_wb.filter_strides[0] = bs;
                                    meta_wb.filter_counts[0] = bc;
                                    meta_wb.filter_word_modes[0] = 0;
                                    for (int pk = 0; pk < wpass; pk++) {
                                        meta_wb.filter_strides[1 + pk] = wstride;
                                        int wc_pk = (input_size - (wpass - pk) * wstride) & ~1;
                                        meta_wb.filter_counts[1 + pk] = wc_pk > 65534 ? 65534 : wc_pk;
                                        meta_wb.filter_word_modes[1 + pk] = 1;
                                    }
                                    if (n_local_wb < 16) {
                                        local_wb_cands[n_local_wb++] = meta_wb;
                                    }
                                }
                            }
                            for (int lwi = 0; lwi < n_local_wb; lwi++) {
                                uint8_t *wb_cur = (uint8_t *)malloc((size_t)input_size);
                                uint8_t *wb_nxt = (uint8_t *)malloc((size_t)input_size);
                                assert(wb_cur != NULL && wb_nxt != NULL);
                                UltraMetadata m_wb = local_wb_cands[lwi];
                                reconstruct_filtered_buffer(working_input, wb_cur, wb_nxt, input_size, format, &m_wb);
                                evaluate_candidate_meta(
                                    wb_cur, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                    stride_req_off, &m_wb, &best_size, &best_meta, NULL
                                );
                                free(wb_cur);
                                free(wb_nxt);
                            }
                        }

                        free(wcur);
                        free(wnxt);
                    }
                }
            }

            /* C. HYBRID_108B, PRO_111B & ULTRA_128B Chained Multi-Stage Search:
             *    1) True 2D/3D Spatial Prediction (stride S1 followed by distinct stride S2, and (S1, S1, S2))
             *    2) Chained x86 E8 + E9 + Multi-Segment 2D Stride-Delta (including 16-bit Word-Stride on PRO_111B/ULTRA_128B)
             */
            if (is_multi_stage && best_meta.compressed_bits > 300) {
                static const int base_pair_s1[] = {1, 2, 3, 4, 6, 8, 12, 13, 16, 24, 26, 30, 32};
                static const int base_pair_s2[] = {1, 2, 3, 4, 6, 8, 12, 13, 16, 24, 26, 30, 32, 48, 64, 96, 128, 256};
                int pair_s1[16];
                int pair_s2[24];
                int n_s1 = 0, n_s2 = 0;
                for (size_t i = 0; i < sizeof(base_pair_s1) / sizeof(base_pair_s1[0]); i++) {
                    pair_s1[n_s1++] = base_pair_s1[i];
                }
                for (size_t i = 0; i < sizeof(base_pair_s2) / sizeof(base_pair_s2[0]); i++) {
                    pair_s2[n_s2++] = base_pair_s2[i];
                }
                for (int si = 0; si < num_strides && si < 3; si++) {
                    int cs = candidate_strides[si];
                    if (cs <= max_allowed_stride) {
                        int d1 = 0, d2 = 0;
                        for (int k = 0; k < n_s1; k++) if (pair_s1[k] == cs) d1 = 1;
                        if (!d1 && cs <= 64 && n_s1 < 16) pair_s1[n_s1++] = cs;
                        for (int k = 0; k < n_s2; k++) if (pair_s2[k] == cs) d2 = 1;
                        if (!d2 && n_s2 < 24) pair_s2[n_s2++] = cs;
                    }
                }

                UltraMetadata stage_c_cands[128];
                int n_stage_c = 0;

                for (int i1 = 0; i1 < n_s1; i1++) {
                    int s1 = pair_s1[i1];
                    if (s1 >= input_size || s1 > max_allowed_stride) continue;
                    if (input_size >= 8192 && s1 > 8) {
                        int is_top_cand = 0;
                        for (int sc = 0; sc < num_strides && sc < 3; sc++) {
                            if (candidate_strides[sc] == s1) is_top_cand = 1;
                        }
                        if (!is_top_cand) continue;
                    }
                    int c1 = (input_size - s1) > 65535 ? 65535 : (input_size - s1);
                    uint8_t *buf_p1 = (uint8_t *)malloc((size_t)input_size);
                    uint8_t *buf_p2 = (uint8_t *)malloc((size_t)input_size);
                    uint8_t *buf_p3 = (uint8_t *)malloc((size_t)input_size);
                    assert(buf_p1 != NULL && buf_p2 != NULL && buf_p3 != NULL);
                    apply_forward_stride_difference(working_input, buf_p1, input_size, s1, c1);
                    if (memcmp(buf_p1, working_input, (size_t)input_size) == 0) {
                        free(buf_p1);
                        free(buf_p2);
                        free(buf_p3);
                        continue;
                    }

                    int kept_s2[24];
                    int kept_g2[24];
                    int n_kept_s2 = 0;

                    for (int i2 = 0; i2 < n_s2; i2++) {
                        int s2 = pair_s2[i2];
                        if (s2 == s1 || (input_size >= 8192 && s2 < s1) || s2 >= input_size || s2 > max_allowed_stride) continue;
                        if (s1 > 4 && (s2 % s1 != 0) && (s1 % s2 != 0) && s2 != s1 + 2 && s1 != s2 + 2) continue;
                        int c2 = (input_size - s2) > 65535 ? 65535 : (input_size - s2);
                        apply_forward_stride_difference(buf_p1, buf_p2, input_size, s2, c2);
                        if (memcmp(buf_p2, buf_p1, (size_t)input_size) == 0) continue;
                        int eq_p0 = 0, eq_p1 = 0, eq_p2 = 0;
                        int start_k = (s1 > s2 ? s1 : s2) + 1;
                        for (int k = start_k; k < input_size; k++) {
                            if (working_input[k] == 0 || working_input[k] == working_input[k - 1] ||
                                working_input[k] == working_input[k - s1] || working_input[k] == working_input[k - s2]) {
                                eq_p0++;
                            }
                            if (buf_p1[k] == 0 || buf_p1[k] == buf_p1[k - 1] || buf_p1[k] == buf_p1[k - s1]) {
                                eq_p1++;
                            }
                            if (buf_p2[k] == 0 || buf_p2[k] == buf_p2[k - 1] || buf_p2[k] == buf_p2[k - s2]) {
                                eq_p2++;
                            }
                        }
                        if (eq_p1 <= eq_p0 || eq_p2 <= eq_p1 + (input_size >= 8192 ? (input_size - start_k) / 20 : 0) ||
                            eq_p2 <= eq_p0 + (input_size - start_k) / (input_size >= 8192 ? 12 : 16)) {
                            continue;
                        }
                        if (input_size >= 8192) {
                            int eq1_h2 = 0, eq2_h2 = 0;
                            for (int k = input_size / 2; k < input_size; k++) {
                                if (buf_p1[k] == 0 || buf_p1[k] == buf_p1[k - 1] || buf_p1[k] == buf_p1[k - s1]) eq1_h2++;
                                if (buf_p2[k] == 0 || buf_p2[k] == buf_p2[k - 1] || buf_p2[k] == buf_p2[k - s2]) eq2_h2++;
                            }
                            if (eq2_h2 <= eq1_h2) continue;
                        }
                        int gain2 = eq_p2 - eq_p1;
                        int dom_s2 = 0;
                        if (s2 > 32) {
                            for (int kp = 0; kp < n_kept_s2; kp++) {
                                int d2 = kept_s2[kp];
                                if (d2 >= 2 && s2 > d2 && (s2 % d2 == 0) &&
                                    kept_g2[kp] >= 120 && gain2 <= kept_g2[kp] + 32) {
                                    dom_s2 = 1;
                                    break;
                                }
                            }
                        }
                        if (dom_s2) {
                            continue;
                        }
                        kept_s2[n_kept_s2] = s2;
                        kept_g2[n_kept_s2] = gain2;
                        n_kept_s2++;
                    }

                    int eval_s2_limit = (input_size >= 8192) ? 2 : n_kept_s2;
                    for (int a = 0; a < n_kept_s2 && a < eval_s2_limit; a++) {
                        if (input_size >= 8192) {
                            for (int b = a + 1; b < n_kept_s2; b++) {
                                if (kept_g2[b] > kept_g2[a]) {
                                    int ts = kept_s2[a]; kept_s2[a] = kept_s2[b]; kept_s2[b] = ts;
                                    int tg = kept_g2[a]; kept_g2[a] = kept_g2[b]; kept_g2[b] = tg;
                                }
                            }
                        }
                        int s2 = kept_s2[a];
                        int c2 = (input_size - s2) > 65535 ? 65535 : (input_size - s2);
                        apply_forward_stride_difference(buf_p1, buf_p2, input_size, s2, c2);
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
                        if (n_stage_c < 128) {
                            stage_c_cands[n_stage_c++] = meta_2d;
                        }
                    }

                    /* 3-pass heterogeneous (s1, s1, s2) for s1 in {1, 2} and s2 in {6, 8, 12, 13, 16, 24, 26, 32} */
                    if ((s1 == 1 || s1 == 2) && input_size > 2 * s1 + 32) {
                        int has_h13 = 0;
                        if (s1 == 2 && input_size >= 8192) {
                            int hc13 = input_size / 2;
                            int eq13_h = 0;
                            for (int k = input_size - hc13 + 13; k < input_size; k++) {
                                if (working_input[k] == working_input[k - 13]) eq13_h++;
                            }
                            if (eq13_h >= (hc13 - 13) / 4) {
                                has_h13 = 1;
                                int tc13 = (input_size > 32768) ? (hc13 - 13) : hc13;
                                apply_forward_stride_difference(working_input, buf_p3, input_size, 13, tc13);
                                UltraMetadata meta_h13;
                                memset(&meta_h13, 0, sizeof(meta_h13));
                                meta_h13.format = format;
                                meta_h13.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                meta_h13.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                      ? detect_buffer_literal_bits(buf_p3, input_size, NULL) : 8;
                                meta_h13.filter_passes = 1;
                                meta_h13.filter_stride = 13;
                                meta_h13.filter_strides[0] = 13;
                                meta_h13.filter_counts[0] = tc13;
                                if (n_stage_c < 128) {
                                    stage_c_cands[n_stage_c++] = meta_h13;
                                }
                            }
                        }
                        int c1b = (input_size - 2 * s1) > 65535 ? 65535 : (input_size - 2 * s1);
                        apply_forward_stride_difference(buf_p1, buf_p2, input_size, s1, c1b);
                        int p2_ok = !(has_h13 && input_size > 32768) && (memcmp(buf_p2, buf_p1, (size_t)input_size) != 0);
                        if (p2_ok && input_size >= 8192) {
                            int eq1_h = 0, eq2_h = 0;
                            int h_lim = input_size / 2;
                            for (int k = 2 * s1 + 1; k < h_lim; k++) {
                                if (buf_p1[k] == 0 || buf_p1[k] == 1 || buf_p1[k] == 255 || buf_p1[k] == buf_p1[k - 1] || buf_p1[k] == buf_p1[k - s1]) eq1_h++;
                                if (buf_p2[k] == 0 || buf_p2[k] == 1 || buf_p2[k] == 255 || buf_p2[k] == buf_p2[k - 1] || buf_p2[k] == buf_p2[k - s1]) eq2_h++;
                            }
                            if (eq2_h <= eq1_h) p2_ok = 0;
                        }
                        if (p2_ok) {
                            static const int s3_cands[] = {6, 8, 12, 13, 16, 24, 26, 32};
                            for (size_t i3 = 0; i3 < sizeof(s3_cands) / sizeof(s3_cands[0]); i3++) {
                                int s2 = s3_cands[i3];
                                int c2 = (input_size - 2 * s1 - s2) > 65535 ? 65535 : (input_size - 2 * s1 - s2);
                                if (c2 <= 0) continue;
                                int c2_list[2] = {c2, input_size / 2};
                                int n_c2 = (input_size >= 4096 && s1 == 2 && s2 == 13 &&
                                            (input_size / 2) > s2 && (input_size / 2) < c2) ? 2 : 1;
                                for (int ci = 0; ci < n_c2; ci++) {
                                    int cur_c2 = c2_list[ci];
                                    apply_forward_stride_difference(buf_p2, buf_p3, input_size, s2, cur_c2);
                                    if (memcmp(buf_p3, buf_p2, (size_t)input_size) == 0) continue;
                                    int eq_p2b = 0, eq_p3b = 0;
                                    for (int k = s2 + 1; k < input_size; k++) {
                                        if (buf_p2[k] == 0 || buf_p2[k] == buf_p2[k - 1] || buf_p2[k] == buf_p2[k - s1] || buf_p2[k] == buf_p2[k - s2]) eq_p2b++;
                                        if (buf_p3[k] == 0 || buf_p3[k] == buf_p3[k - 1] || buf_p3[k] == buf_p3[k - s2]) eq_p3b++;
                                    }
                                    if (eq_p2b < (input_size - s2) / 8 || eq_p3b <= eq_p2b + (cur_c2 - s2) / 16) continue;
                                    UltraMetadata meta_3d;
                                    memset(&meta_3d, 0, sizeof(meta_3d));
                                    meta_3d.format = format;
                                    meta_3d.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                    meta_3d.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                         ? detect_buffer_literal_bits(buf_p3, input_size, NULL) : 8;
                                    meta_3d.filter_passes = 3;
                                    meta_3d.filter_stride = s1;
                                    meta_3d.filter_strides[0] = s2;
                                    meta_3d.filter_counts[0] = cur_c2;
                                    meta_3d.filter_strides[1] = s1;
                                    meta_3d.filter_counts[1] = c1b;
                                    meta_3d.filter_strides[2] = s1;
                                    meta_3d.filter_counts[2] = c1;
                                    if (n_stage_c < 128) {
                                        stage_c_cands[n_stage_c++] = meta_3d;
                                    }
                                }
                            }
                        }
                    }

                    free(buf_p1);
                    free(buf_p2);
                    free(buf_p3);
                }

                if (input_size >= 8192 && !nested_parallel) {
                    for (int sci = 0; sci < n_stage_c && n_uni_cands < 320; sci++) { uni_req_offs[n_uni_cands] = stride_req_off; uni_cands[n_uni_cands++] = stage_c_cands[sci]; }
                } else {
                    #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
                    for (int sci = 0; sci < n_stage_c; sci++) {
                        uint8_t *sc_cur = (uint8_t *)malloc((size_t)input_size);
                        uint8_t *sc_nxt = (uint8_t *)malloc((size_t)input_size);
                        assert(sc_cur != NULL && sc_nxt != NULL);
                        UltraMetadata m_sc = stage_c_cands[sci];
                        reconstruct_filtered_buffer(working_input, sc_cur, sc_nxt, input_size, format, &m_sc);
                        evaluate_candidate_meta(sc_cur, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit, stride_req_off, &m_sc, &best_size, &best_meta, NULL);
                        free(sc_cur); free(sc_nxt);
                    }
                }

                if (is_word_capable && input_size > 68) {
                    int wc1 = (input_size - 2) & ~1;
                    if (wc1 > 65534) wc1 = 65534;
                    uint8_t *buf_w1 = (uint8_t *)malloc((size_t)input_size);
                    assert(buf_w1 != NULL);
                    apply_forward_word16_stride_difference(working_input, buf_w1, input_size, 2, wc1);
                    if (memcmp(buf_w1, working_input, (size_t)input_size) != 0) {
                        static const int ws2_list[] = {4, 6, 8, 12, 16, 24, 26, 30, 32, 48, 64, 128};
                        int n_ws2 = (int)(sizeof(ws2_list) / sizeof(ws2_list[0]));
                        #pragma omp parallel for schedule(dynamic) if (!nested_parallel && input_size < 8192)
                        for (int wi2 = 0; wi2 < n_ws2; wi2++) {
                            int s2 = ws2_list[wi2];
                            if (s2 >= input_size - 2 || s2 > max_allowed_stride) continue;
                            int wc2 = (input_size - 2 - s2) & ~1;
                            if (wc2 > 65534) wc2 = 65534;
                            if (wc2 < 2) continue;
                            uint8_t *buf_w2 = (uint8_t *)malloc((size_t)input_size);
                            assert(buf_w2 != NULL);
                            apply_forward_word16_stride_difference(buf_w1, buf_w2, input_size, s2, wc2);
                            if (memcmp(buf_w2, buf_w1, (size_t)input_size) == 0) {
                                free(buf_w2);
                                continue;
                            }
                            int eq_w0 = 0, eq_w1 = 0, eq_w2 = 0;
                            for (int k = s2 + 2; k < input_size; k++) {
                                if (working_input[k] == 0 || working_input[k] == working_input[k - 2] || working_input[k] == working_input[k - s2]) eq_w0++;
                                if (buf_w1[k] == 0 || buf_w1[k] == buf_w1[k - 2] || buf_w1[k] == buf_w1[k - s2]) eq_w1++;
                                if (buf_w2[k] == 0 || buf_w2[k] == buf_w2[k - 2] || buf_w2[k] == buf_w2[k - s2]) eq_w2++;
                            }
                            if (eq_w1 <= eq_w0 || eq_w2 <= eq_w1 + (input_size - s2) / 16) {
                                free(buf_w2);
                                continue;
                            }
                            UltraMetadata meta_w2d;
                            memset(&meta_w2d, 0, sizeof(meta_w2d));
                            meta_w2d.format = format;
                            meta_w2d.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                            meta_w2d.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                  ? detect_buffer_literal_bits(buf_w2, input_size, NULL) : 8;
                            meta_w2d.filter_passes = 2;
                            meta_w2d.filter_stride = 2;
                            meta_w2d.filter_strides[0] = s2;
                            meta_w2d.filter_counts[0] = wc2;
                            meta_w2d.filter_word_modes[0] = 1;
                            meta_w2d.filter_strides[1] = 2;
                            meta_w2d.filter_counts[1] = wc1;
                            meta_w2d.filter_word_modes[1] = 1;
                            if (input_size >= 8192 && !nested_parallel && n_uni_cands < 320) {
                                uni_req_offs[n_uni_cands] = stride_req_off;
                                uni_cands[n_uni_cands++] = meta_w2d;
                            } else {
                                evaluate_candidate_meta(
                                    buf_w2, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                    stride_req_off, &meta_w2d, &best_size, &best_meta, NULL
                                );
                            }
                            free(buf_w2);
                        }
                    }
                    free(buf_w1);
                }

                /* Evaluate Multi-Segment Stepped-Tail Word/Byte Stride-Delta (+ optional E8) on hybrid ROMs */
                int is_hybrid_rom_cand = 0;
                int eq_q2_raw = 0;
                if (input_size >= 512) {
                    int q1_s = input_size / 4;
                    int q2_s = input_size / 2;
                    int q3_s = (3 * input_size) / 4;
                    int eq_q1_s2 = 0, eq_q0_w1 = 0, eq_q0_w3 = 0, eq_q2_w1 = 0, eq_q2_w3 = 0;
                    for (int k = q1_s + 2; k < q2_s; k++) {
                        if (working_input[k] == working_input[k - 2]) eq_q1_s2++;
                    }
                    for (int k = 6; k + 1 < q1_s; k += 2) {
                        uint16_t w0 = (uint16_t)(working_input[k] | (working_input[k + 1] << 8));
                        uint16_t w1 = (uint16_t)(working_input[k - 2] | (working_input[k - 1] << 8));
                        uint16_t w2 = (uint16_t)(working_input[k - 4] | (working_input[k - 3] << 8));
                        uint16_t w3 = (uint16_t)(working_input[k - 6] | (working_input[k - 5] << 8));
                        uint8_t hi1 = (uint8_t)((uint16_t)(w0 - w1) >> 8);
                        uint8_t hi3 = (uint8_t)((uint16_t)(w0 - 3u * w1 + 3u * w2 - w3) >> 8);
                        if (hi1 == 0 || hi1 == 0xFFu) eq_q0_w1++;
                        if (hi3 == 0 || hi3 == 0xFFu) eq_q0_w3++;
                    }
                    for (int k = q2_s + 6; k + 1 < q3_s; k += 2) {
                        uint16_t w0 = (uint16_t)(working_input[k] | (working_input[k + 1] << 8));
                        uint16_t w1 = (uint16_t)(working_input[k - 2] | (working_input[k - 1] << 8));
                        uint16_t w2 = (uint16_t)(working_input[k - 4] | (working_input[k - 3] << 8));
                        uint16_t w3 = (uint16_t)(working_input[k - 6] | (working_input[k - 5] << 8));
                        uint8_t hi0 = (uint8_t)(w0 >> 8);
                        uint8_t hi1 = (uint8_t)((uint16_t)(w0 - w1) >> 8);
                        uint8_t hi3 = (uint8_t)((uint16_t)(w0 - 3u * w1 + 3u * w2 - w3) >> 8);
                        if (hi0 == 0 || hi0 == 0xFFu) eq_q2_raw++;
                        if (hi1 == 0 || hi1 == 0xFFu) eq_q2_w1++;
                        if (hi3 == 0 || hi3 == 0xFFu) eq_q2_w3++;
                    }
                    int best_q0_w = (eq_q0_w3 > eq_q0_w1) ? eq_q0_w3 : eq_q0_w1;
                    int best_q2_w = (eq_q2_w3 > eq_q2_w1) ? eq_q2_w3 : eq_q2_w1;
                    if (best_q2_w > eq_q2_raw + (q3_s - q2_s) / 16 &&
                        best_q0_w * 3 < best_q2_w * 2 &&
                        ((best_e8_buf != NULL && best_e8_tail_chosen >= 3) || eq_q1_s2 >= (q2_s - q1_s) / 4)) {
                        is_hybrid_rom_cand = 1;
                    }
                }
                if (is_hybrid_rom_cand) {
                    int e8_tail_hyb = (input_size <= 65535) ? input_size : 65535;
                    int half_t = input_size / 2;
                    int three_q_t = (3 * input_size) / 4;
                    int quarter_t = input_size / 4;
                    UltraMetadata hyb_rom_cands[32];
                    int n_hyb_rom = 0;
                    uint8_t *st_buf = (uint8_t *)malloc((size_t)input_size);
                    uint8_t *e8_on_st = (uint8_t *)malloc((size_t)input_size);
                    uint8_t *e9_on_st = (uint8_t *)malloc((size_t)input_size);
                    assert(st_buf != NULL && e8_on_st != NULL && e9_on_st != NULL);
                    if (is_word_capable) {
                        uint8_t *w_cur = (uint8_t *)malloc((size_t)input_size);
                        assert(w_cur != NULL);
                        memcpy(w_cur, working_input, (size_t)input_size);
                        for (int wp = 1; wp <= 3; wp++) {
                            int wc = (half_t - wp * 2) & ~1;
                            if (wc < 4 || wc > 65534) break;
                            apply_forward_word16_stride_difference(w_cur, st_buf, input_size, 2, wc);
                            memcpy(w_cur, st_buf, (size_t)input_size);
                            if (input_size >= 8192) {
                                int eq_q2_hi = 0;
                                for (int k = half_t + 6; k + 1 < three_q_t; k += 2) {
                                    if (w_cur[k + 1] == 0 || w_cur[k + 1] == 0xFFu) eq_q2_hi++;
                                }
                                if (eq_q2_hi <= eq_q2_raw + (quarter_t / 16)) continue;
                            }

                            /* Evaluate without E8 */
                            UltraMetadata meta_w_only;
                            memset(&meta_w_only, 0, sizeof(meta_w_only));
                            meta_w_only.format = format;
                            meta_w_only.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                            meta_w_only.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                     ? detect_buffer_literal_bits(w_cur, input_size, NULL) : 8;
                            meta_w_only.filter_passes = wp;
                            meta_w_only.filter_stride = 2;
                            for (int pk = 0; pk < wp; pk++) {
                                meta_w_only.filter_strides[pk] = 2;
                                meta_w_only.filter_counts[pk] = (half_t - (wp - pk) * 2) & ~1;
                                meta_w_only.filter_word_modes[pk] = 1;
                            }
                            if (n_hyb_rom < 32) {
                                hyb_rom_cands[n_hyb_rom++] = meta_w_only;
                            }

                            /* Evaluate with E8 */
                            if (w_cur[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                w_cur[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                int h8w = apply_forward_x86_opcode_filter(
                                    w_cur, e8_on_st, input_size, e8_tail_hyb, X86_OPCODE_CALL_REL16
                                );
                                if (h8w >= 2) {
                                    UltraMetadata meta_ew;
                                    memset(&meta_ew, 0, sizeof(meta_ew));
                                    meta_ew.format = format;
                                    meta_ew.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                    meta_ew.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                         ? detect_buffer_literal_bits(e8_on_st, input_size, NULL) : 8;
                                    meta_ew.filter_passes = 1 + wp;
                                    meta_ew.filter_stride = 0;
                                    meta_ew.filter_strides[0] = 0;
                                    meta_ew.filter_counts[0] = e8_tail_hyb;
                                    for (int pk = 0; pk < wp; pk++) {
                                        meta_ew.filter_strides[1 + pk] = 2;
                                        meta_ew.filter_counts[1 + pk] = (half_t - (wp - pk) * 2) & ~1;
                                        meta_ew.filter_word_modes[1 + pk] = 1;
                                    }
                                    if (n_hyb_rom < 32) {
                                        hyb_rom_cands[n_hyb_rom++] = meta_ew;
                                    }
                                }
                            }
                        }

                        /* Stepped-tail multi-order 16-bit word-delta: 1 pass on 3*N/4 + 1..3 passes on N/2 (+ optional E8) */
                        int wc_tq = (three_q_t - 2) & ~1;
                        if (wc_tq >= 8 && wc_tq <= 65534) {
                            apply_forward_word16_stride_difference(working_input, w_cur, input_size, 2, wc_tq);
                            for (int wp = 1; wp <= 3; wp++) {
                                int wc_h = (half_t - (wp + 1) * 2) & ~1;
                                if (wc_h < 4 || wc_h > 65534) break;
                                apply_forward_word16_stride_difference(w_cur, st_buf, input_size, 2, wc_h);
                                memcpy(w_cur, st_buf, (size_t)input_size);
                                if (input_size >= 8192) {
                                    int eq_q2_hi = 0;
                                    for (int k = half_t + 6; k + 1 < three_q_t; k += 2) {
                                        if (w_cur[k + 1] == 0 || w_cur[k + 1] == 0xFFu) eq_q2_hi++;
                                    }
                                    if (eq_q2_hi <= eq_q2_raw + (quarter_t / 16)) continue;
                                }

                                /* Evaluate stepped-tail word-delta without E8 */
                                UltraMetadata meta_st_no_e8;
                                memset(&meta_st_no_e8, 0, sizeof(meta_st_no_e8));
                                meta_st_no_e8.format = format;
                                meta_st_no_e8.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                meta_st_no_e8.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                           ? detect_buffer_literal_bits(w_cur, input_size, NULL) : 8;
                                meta_st_no_e8.filter_passes = 1 + wp;
                                meta_st_no_e8.filter_stride = 2;
                                for (int pk = 0; pk < wp; pk++) {
                                    meta_st_no_e8.filter_strides[pk] = 2;
                                    meta_st_no_e8.filter_counts[pk] = (half_t - (wp + 1 - pk) * 2) & ~1;
                                    meta_st_no_e8.filter_word_modes[pk] = 1;
                                }
                                meta_st_no_e8.filter_strides[wp] = 2;
                                meta_st_no_e8.filter_counts[wp] = wc_tq;
                                meta_st_no_e8.filter_word_modes[wp] = 1;
                                if (n_hyb_rom < 32) {
                                    hyb_rom_cands[n_hyb_rom++] = meta_st_no_e8;
                                }

                                /* Evaluate stepped-tail word-delta with E8 */
                                if (w_cur[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                    w_cur[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                    int h8w = apply_forward_x86_opcode_filter(
                                        w_cur, e8_on_st, input_size, e8_tail_hyb, X86_OPCODE_CALL_REL16
                                    );
                                    if (h8w >= 2) {
                                        UltraMetadata meta_st;
                                        memset(&meta_st, 0, sizeof(meta_st));
                                        meta_st.format = format;
                                        meta_st.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                        meta_st.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                             ? detect_buffer_literal_bits(e8_on_st, input_size, NULL) : 8;
                                        meta_st.filter_passes = 2 + wp;
                                        meta_st.filter_stride = 0;
                                        meta_st.filter_strides[0] = 0;
                                        meta_st.filter_counts[0] = e8_tail_hyb;
                                        for (int pk = 0; pk < wp; pk++) {
                                            meta_st.filter_strides[1 + pk] = 2;
                                            meta_st.filter_counts[1 + pk] = (half_t - (wp + 1 - pk) * 2) & ~1;
                                            meta_st.filter_word_modes[1 + pk] = 1;
                                        }
                                        meta_st.filter_strides[1 + wp] = 2;
                                        meta_st.filter_counts[1 + wp] = wc_tq;
                                        meta_st.filter_word_modes[1 + wp] = 1;
                                        if (n_hyb_rom < 32) {
                                            hyb_rom_cands[n_hyb_rom++] = meta_st;
                                        }
                                    }
                                }
                            }

                            /* Also check 3-stage hybrid: dominant Q3 byte-stride on N/4 + word-stride=2 on 3*N/4 and N/2 + E8 */
                            int best_q3_s = 0;
                            int best_q3_eq = 0;
                            int q3_start = input_size - quarter_t;
                            for (int s = 3; s <= 64; s++) {
                                if (s >= quarter_t) break;
                                int eq = 0;
                                for (int i = q3_start + s; i < input_size; i++) {
                                    if (working_input[i] == working_input[i - s]) {
                                        eq++;
                                    }
                                }
                                if (eq > best_q3_eq) {
                                    best_q3_eq = eq;
                                    best_q3_s = s;
                                }
                            }
                            /* Prefer fundamental period if a harmonic (e.g. 39 = 3*13) tied or slightly beat it */
                            if (best_q3_s > 6) {
                                for (int d = 3; d < best_q3_s; d++) {
                                    if (best_q3_s % d == 0) {
                                        int eq_d = 0;
                                        for (int i = q3_start + d; i < input_size; i++) {
                                            if (working_input[i] == working_input[i - d]) eq_d++;
                                        }
                                        if (eq_d * 5 >= best_q3_eq * 4) {
                                            best_q3_s = d;
                                            best_q3_eq = eq_d;
                                            break;
                                        }
                                    }
                                }
                            }
                            /* Also detect linear ramp periods in Q3 (e.g. 3-byte RGB LUT ramps where d_cur == d_prv) */
                            {
                                static const int ramp_s_cands[] = {3, 4, 6, 8, 12, 13, 16};
                                for (size_t ri = 0; ri < sizeof(ramp_s_cands) / sizeof(ramp_s_cands[0]); ri++) {
                                    int s = ramp_s_cands[ri];
                                    if (2 * s >= quarter_t) continue;
                                    int req = 0;
                                    for (int i = q3_start + 2 * s; i < input_size; i++) {
                                        uint8_t dc = (uint8_t)(working_input[i] - working_input[i - s]);
                                        uint8_t dp = (uint8_t)(working_input[i - 1] - working_input[i - 1 - s]);
                                        uint8_t ds = (uint8_t)(working_input[i - s] - working_input[i - 2 * s]);
                                        if (dc == 0 || dc == dp || dc == ds) req++;
                                    }
                                    if (req > best_q3_eq * 2 && req >= (quarter_t * 3) / 5) {
                                        best_q3_eq = req;
                                        best_q3_s = s;
                                        break;
                                    }
                                }
                            }
                            if (best_q3_s >= 3 && best_q3_eq >= quarter_t / 4) {
                                if (input_size >= 8192) { n_hyb_rom = 0; n_uni_cands = defer_uni ? 1 : 0; }
                                int c_q3 = quarter_t - best_q3_s;
                                apply_forward_stride_difference(working_input, st_buf, input_size, best_q3_s, c_q3);
                                apply_forward_word16_stride_difference(st_buf, w_cur, input_size, 2, wc_tq);
                                for (int wp = 1; wp <= 3; wp++) {
                                    int wc_h = (half_t - (wp + 1) * 2) & ~1;
                                    if (wc_h < 4 || wc_h > 65534) break;
                                    apply_forward_word16_stride_difference(w_cur, st_buf, input_size, 2, wc_h);
                                    memcpy(w_cur, st_buf, (size_t)input_size);
                                    if (w_cur[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                        w_cur[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                        int h8w = apply_forward_x86_opcode_filter(
                                            w_cur, e8_on_st, input_size, e8_tail_hyb, X86_OPCODE_CALL_REL16
                                        );
                                        if (h8w >= 2) {
                                            UltraMetadata meta_q3;
                                            memset(&meta_q3, 0, sizeof(meta_q3));
                                            meta_q3.format = format;
                                            meta_q3.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                            meta_q3.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                                                 ? detect_buffer_literal_bits(e8_on_st, input_size, NULL) : 8;
                                            meta_q3.filter_passes = 3 + wp;
                                            meta_q3.filter_stride = 0;
                                            meta_q3.filter_strides[0] = 0;
                                            meta_q3.filter_counts[0] = e8_tail_hyb;
                                            for (int pk = 0; pk < wp; pk++) {
                                                meta_q3.filter_strides[1 + pk] = 2;
                                                meta_q3.filter_counts[1 + pk] = (half_t - (wp + 1 - pk) * 2) & ~1;
                                                meta_q3.filter_word_modes[1 + pk] = 1;
                                            }
                                            meta_q3.filter_strides[1 + wp] = 2;
                                            meta_q3.filter_counts[1 + wp] = wc_tq;
                                            meta_q3.filter_word_modes[1 + wp] = 1;
                                            meta_q3.filter_strides[2 + wp] = best_q3_s;
                                            meta_q3.filter_counts[2 + wp] = c_q3;
                                            meta_q3.filter_word_modes[2 + wp] = 0;
                                            if (n_hyb_rom < 32) {
                                                hyb_rom_cands[n_hyb_rom++] = meta_q3;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        free(w_cur);
                    } else {
                        /* HYBRID_108B: Stepped-tail byte-stride (1 pass on 3*N/4 + 1..3 passes on N/2, + optional Q3 period) + E8 */
                        int bc_tq = three_q_t - 2;
                        if (bc_tq >= 8 && bc_tq <= 65535) {
                            uint8_t *b_cur = (uint8_t *)malloc((size_t)input_size);
                            assert(b_cur != NULL);
                            apply_forward_stride_difference(working_input, b_cur, input_size, 2, bc_tq);
                            for (int bp = 1; bp <= 3; bp++) {
                                int bc_h = half_t - (bp + 1) * 2;
                                if (bc_h < 4 || bc_h > 65535) break;
                                apply_forward_stride_difference(b_cur, st_buf, input_size, 2, bc_h);
                                memcpy(b_cur, st_buf, (size_t)input_size);
                                if (b_cur[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                    b_cur[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                    int h8b = apply_forward_x86_opcode_filter(
                                        b_cur, e8_on_st, input_size, e8_tail_hyb, X86_OPCODE_CALL_REL16
                                    );
                                    if (h8b >= 2) {
                                        UltraMetadata meta_hb;
                                        memset(&meta_hb, 0, sizeof(meta_hb));
                                        meta_hb.format = format;
                                        meta_hb.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                        meta_hb.literal_bits = 8;
                                        meta_hb.filter_passes = 2 + bp;
                                        meta_hb.filter_stride = 0;
                                        meta_hb.filter_strides[0] = 0;
                                        meta_hb.filter_counts[0] = e8_tail_hyb;
                                        for (int pk = 0; pk < bp; pk++) {
                                            meta_hb.filter_strides[1 + pk] = 2;
                                            meta_hb.filter_counts[1 + pk] = half_t - (bp + 1 - pk) * 2;
                                        }
                                        meta_hb.filter_strides[1 + bp] = 2;
                                        meta_hb.filter_counts[1 + bp] = bc_tq;
                                        if (n_hyb_rom < 32) {
                                            hyb_rom_cands[n_hyb_rom++] = meta_hb;
                                        }
                                    }
                                }
                            }
                            int q3_start = input_size - quarter_t;
                            int best_q3_s = 0, best_q3_eq = 0;
                            for (int s = 3; s <= 128 && s < quarter_t / 2; s++) {
                                int eq = 0;
                                for (int k = q3_start + s; k < input_size; k++) {
                                    if (working_input[k] == working_input[k - s]) eq++;
                                }
                                if (eq > best_q3_eq) {
                                    best_q3_eq = eq;
                                    best_q3_s = s;
                                }
                            }
                            if (best_q3_s >= 6) {
                                for (int d = 2; d <= 4; d++) {
                                    if (best_q3_s % d == 0 && best_q3_s / d >= 3) {
                                        int sub_s = best_q3_s / d;
                                        int eq_d = 0;
                                        for (int k = q3_start + sub_s; k < input_size; k++) {
                                            if (working_input[k] == working_input[k - sub_s]) eq_d++;
                                        }
                                        if (eq_d * 5 >= best_q3_eq * 4) {
                                            best_q3_s = sub_s;
                                            best_q3_eq = eq_d;
                                        }
                                    }
                                }
                            }
                            for (int s = 3; s <= 32 && s < quarter_t / 4; s++) {
                                int req = 0;
                                for (int k = q3_start + 2 * s; k < input_size; k++) {
                                    if ((uint8_t)(working_input[k] - working_input[k - s]) ==
                                        (uint8_t)(working_input[k - s] - working_input[k - 2 * s])) {
                                        req++;
                                    }
                                }
                                if (req > best_q3_eq * 2 && req >= (quarter_t * 3) / 5) {
                                    best_q3_eq = req;
                                    best_q3_s = s;
                                    break;
                                }
                            }
                            if (best_q3_s >= 3 && best_q3_eq >= quarter_t / 4) {
                                int c_q3 = quarter_t - best_q3_s;
                                apply_forward_stride_difference(working_input, st_buf, input_size, best_q3_s, c_q3);
                                apply_forward_stride_difference(st_buf, b_cur, input_size, 2, bc_tq);
                                for (int bp = 1; bp <= 3; bp++) {
                                    int bc_h = half_t - (bp + 1) * 2;
                                    if (bc_h < 4 || bc_h > 65535) break;
                                    apply_forward_stride_difference(b_cur, st_buf, input_size, 2, bc_h);
                                    memcpy(b_cur, st_buf, (size_t)input_size);
                                    if (b_cur[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                        b_cur[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                        int h8q = apply_forward_x86_opcode_filter(
                                            b_cur, e8_on_st, input_size, e8_tail_hyb, X86_OPCODE_CALL_REL16
                                        );
                                        if (h8q >= 2) {
                                            UltraMetadata meta_hq3;
                                            memset(&meta_hq3, 0, sizeof(meta_hq3));
                                            meta_hq3.format = format;
                                            meta_hq3.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                            meta_hq3.literal_bits = 8;
                                            meta_hq3.filter_passes = 3 + bp;
                                            meta_hq3.filter_stride = 0;
                                            meta_hq3.filter_strides[0] = 0;
                                            meta_hq3.filter_counts[0] = e8_tail_hyb;
                                            for (int pk = 0; pk < bp; pk++) {
                                                meta_hq3.filter_strides[1 + pk] = 2;
                                                meta_hq3.filter_counts[1 + pk] = half_t - (bp + 1 - pk) * 2;
                                            }
                                            meta_hq3.filter_strides[1 + bp] = 2;
                                            meta_hq3.filter_counts[1 + bp] = bc_tq;
                                            meta_hq3.filter_strides[2 + bp] = best_q3_s;
                                            meta_hq3.filter_counts[2 + bp] = c_q3;
                                            if (n_hyb_rom < 32) {
                                                hyb_rom_cands[n_hyb_rom++] = meta_hq3;
                                            }
                                        }
                                    }
                                }
                            }
                            free(b_cur);
                        }

                        /* HYBRID_108B: E8+E9 + (2, 2, 6) on half-suffix */
                        int c1 = half_t - 2, c2 = half_t - 4, c3 = half_t - 10;
                        if (!g_single_e8_only && c3 > 6 && best_e8_buf != NULL && best_e8_tail_chosen >= 3) {
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
                                if (n_hyb_rom < 32) {
                                    hyb_rom_cands[n_hyb_rom++] = meta_eb;
                                }
                            }
                        }
                    }
                    free(st_buf);
                    free(e8_on_st);
                    free(e9_on_st);

                    if (input_size >= 8192 && !nested_parallel) {
                        for (int hri = 0; hri < n_hyb_rom && n_uni_cands < 320; hri++) { uni_req_offs[n_uni_cands] = stride_req_off; uni_cands[n_uni_cands++] = hyb_rom_cands[hri]; }
                    } else {
                        #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
                        for (int hri = 0; hri < n_hyb_rom; hri++) {
                            uint8_t *hr_cur = (uint8_t *)malloc((size_t)input_size);
                            uint8_t *hr_nxt = (uint8_t *)malloc((size_t)input_size);
                            assert(hr_cur != NULL && hr_nxt != NULL);
                            UltraMetadata m_hr = hyb_rom_cands[hri];
                            reconstruct_filtered_buffer(working_input, hr_cur, hr_nxt, input_size, format, &m_hr);
                            evaluate_candidate_meta(hr_cur, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit, stride_req_off, &m_hr, &best_size, &best_meta, NULL);
                            free(hr_cur); free(hr_nxt);
                        }
                    }
                }

                if (best_e8_buf != NULL && best_e8_tail_chosen >= 3 && !(is_hybrid_rom_cand && input_size >= 8192)) {
                    static const int hyb_strides[] = {1, 2, 3, 4, 8, 12, 16, 24, 32, 64};

                    int n_hyb_s = (int)(sizeof(hyb_strides) / sizeof(hyb_strides[0]));
                    #pragma omp parallel for schedule(dynamic) if (!nested_parallel && input_size < 8192)
                    for (int idx = 0; idx < n_hyb_s; idx++) {
                        int st = hyb_strides[idx];
                        if (st >= input_size) {
                            continue;
                        }
                        int kg = 0, ub = 0;
                        int kt = find_optimal_stride_tail_count(working_input, input_size, st, &kg, &ub);
                        if ((st > 2 && kg < 16) || (input_size >= 8192 && kg < 64)) {
                            continue;
                        }
                        int c_tails[4];
                        int n_ct = 0;
                        if (kg >= 8 && kt > st && kt < input_size - st) c_tails[n_ct++] = kt;
                        if (input_size < 8192 && kg >= 8 && ub > st && ub < input_size - st && ub != kt) c_tails[n_ct++] = ub;
                        if (input_size / 2 > st) c_tails[n_ct++] = input_size / 2;
                        if (input_size < 8192 && input_size >= 3072 && 2048 > st) c_tails[n_ct++] = 2048;
                        if (n_ct == 0) {
                            continue;
                        }

                        uint8_t *st_buf = (uint8_t *)malloc((size_t)input_size);
                        uint8_t *e8_on_st = (uint8_t *)malloc((size_t)input_size);
                        uint8_t *e9_on_st = (uint8_t *)malloc((size_t)input_size);
                        assert(st_buf != NULL && e8_on_st != NULL && e9_on_st != NULL);

                        for (int ti = 0; ti < n_ct; ti++) {
                            int t_st = c_tails[ti];
                            apply_forward_stride_difference(working_input, st_buf, input_size, st, t_st);
                            for (int p = 1; p <= 2; p++) {
                                if (p == 2) {
                                    if (st > 4 || t_st <= 2 * st) break;
                                    apply_forward_stride_difference(st_buf, e9_on_st, input_size, st, t_st);
                                    memcpy(st_buf, e9_on_st, (size_t)input_size);
                                }
                                /* For HYBRID_108B, PRO_111B & ULTRA_128B: can use 1-pass E8 (V=1049/536) directly followed by Stride without dummy E9! */
                                if (is_multi_stage &&
                                    st_buf[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                    st_buf[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                    int h8_direct = apply_forward_x86_opcode_filter(
                                        st_buf, e8_on_st, input_size, best_e8_tail_chosen, X86_OPCODE_CALL_REL16
                                    );
                                    if (h8_direct >= 2) {
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
                                        if (input_size >= 8192 && !nested_parallel && n_uni_cands < 320) {
                                            uni_req_offs[n_uni_cands] = stride_req_off;
                                            uni_cands[n_uni_cands++] = meta_u1;
                                        } else {
                                            evaluate_candidate_meta(
                                                e8_on_st, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                                stride_req_off, &meta_u1, &best_size, &best_meta, NULL
                                            );
                                        }
                                    }
                                }

                                /* Decoder runs E8 -> E9 -> Stride, so encoder applies Stride -> E9 -> E8 */
                                if (format != ULTRA_FORMAT_ULTRA_128B && !g_single_e8_only &&
                                    st_buf[input_size - 2] != X86_OPCODE_JMP_REL16 &&
                                    st_buf[input_size - 1] != X86_OPCODE_JMP_REL16) {
                                    int h9 = apply_forward_x86_opcode_filter(
                                        st_buf, e9_on_st, input_size, best_e8_tail_chosen, X86_OPCODE_JMP_REL16
                                    );
                                    if (h9 >= 2 &&
                                        e9_on_st[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                        e9_on_st[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                        int h8 = apply_forward_x86_opcode_filter(
                                            e9_on_st, e8_on_st, input_size, best_e8_tail_chosen, X86_OPCODE_CALL_REL16
                                        );
                                        if (h8 >= 2) {
                                            UltraMetadata meta_hyb;
                                            memset(&meta_hyb, 0, sizeof(meta_hyb));
                                            meta_hyb.format = format;
                                            meta_hyb.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                                            meta_hyb.literal_bits = 8;
                                            meta_hyb.filter_passes = 2 + p;
                                            meta_hyb.filter_stride = 0; /* Signals V=537 (E8+E9 entry) */
                                            meta_hyb.filter_strides[0] = 0;
                                            meta_hyb.filter_counts[0] = best_e8_tail_chosen;
                                            meta_hyb.filter_strides[1] = 0;
                                            meta_hyb.filter_counts[1] = best_e8_tail_chosen;
                                            for (int pk = 0; pk < p; pk++) {
                                                meta_hyb.filter_strides[2 + pk] = st;
                                                meta_hyb.filter_counts[2 + pk] = t_st;
                                            }
                                            if (input_size >= 8192 && !nested_parallel && n_uni_cands < 320) {
                                                uni_req_offs[n_uni_cands] = stride_req_off;
                                                uni_cands[n_uni_cands++] = meta_hyb;
                                            } else {
                                                evaluate_candidate_meta(
                                                    e8_on_st, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                                    stride_req_off, &meta_hyb, &best_size, &best_meta, NULL
                                                );
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
            if (n_uni_cands > 0) {
                #pragma omp parallel for schedule(dynamic)
                for (int uci = 0; uci < n_uni_cands; uci++) {
                    UltraMetadata m_u = uni_cands[uci];
                    if (m_u.filter_passes == 0) {
                        size_t csz = 0;
                        int cbits = evaluate_candidate_meta(working_input, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit, uni_req_offs[uci], &m_u, &best_size, &best_meta, &csz);
                        if (cbits > 0) {
                            #pragma omp critical
                            {
                                if (csz < base_unfiltered_size) {
                                    base_unfiltered_size = csz;
                                    base_payload_bits = cbits;
                                }
                            }
                        }
                    } else {
                        uint8_t *u_cur = (uint8_t *)malloc((size_t)input_size);
                        uint8_t *u_nxt = (uint8_t *)malloc((size_t)input_size);
                        assert(u_cur != NULL && u_nxt != NULL);
                        reconstruct_filtered_buffer(working_input, u_cur, u_nxt, input_size, format, &m_u);
                        evaluate_candidate_meta(u_cur, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit, uni_req_offs[uci], &m_u, &best_size, &best_meta, NULL);
                        free(u_cur);
                        free(u_nxt);
                    }
                }
            }
        }
        free(best_e8_buf);

        /*
         * Final Exact DH Sentinel Refinement for ULTRA_FORMAT_ULTRA_128B:
         * Rank all non-dominated DH sentinels (3..127) by exact byte-histogram cost AND by exact
         * DP-parse literal token distribution on the winning post-filtered buffer (and on working_input
         * if unfiltered remains competitive) and evaluate the top candidates in parallel.
         */
        if (format == ULTRA_FORMAT_ULTRA_128B && (input_size < 4096 || base_payload_bits > 256)) {
            if (top_level_call) g_deep_mode = (input_size <= 32768) ? 2 : 0; else tls_deep_mode = (input_size <= 32768) ? 2 : 0;
            int max_dh_cands = (input_size >= 8192) ? 8 : 12;
            if (!defer_uni && best_meta.filter_passes == 0) {
                int raw_est_dh = find_best_dh_sentinel(working_input, input_size, NULL);
                int uniq_dhs[16];
                int n_uniq = rank_top_dh_sentinels(working_input, input_size, -1, uniq_dhs, max_dh_cands);
                ParsePlan ref_plan8 = {NULL, 0, 0};
                if (0 && solve_exact_parse(working_input, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, &ref_plan8)) {
                    int plan_dhs[6];
                    int np = rank_top_dh_sentinels_from_plan(working_input, input_size, &ref_plan8, plan_dhs, 6);
                    for (int pi = 0; pi < np && n_uniq < 16; pi++) {
                        int pdh = plan_dhs[pi];
                        if (pdh == raw_est_dh) continue;
                        int dup = 0;
                        for (int k = 0; k < n_uniq; k++) if (uniq_dhs[k] == pdh) dup = 1;
                        if (!dup) uniq_dhs[n_uniq++] = pdh;
                    }
                    free(ref_plan8.tokens);
                }
                int lit_bits_cached = detect_buffer_literal_bits(working_input, input_size, NULL);
                #pragma omp parallel for schedule(dynamic) if (!omp_in_parallel() && n_uniq > 1)
                for (int ui = 0; ui < n_uniq; ui++) {
                    int adh = uniq_dhs[ui];
                    UltraMetadata alt_meta;
                    memset(&alt_meta, 0, sizeof(alt_meta));
                    alt_meta.format = format;
                    alt_meta.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                    alt_meta.dh_sentinel = adh;
                    alt_meta.literal_bits = lit_bits_cached;
                    evaluate_candidate_meta(
                        working_input, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                        -adh, &alt_meta, &best_size, &best_meta, NULL
                    );
                }
            }

            if (best_meta.filter_passes > 0) {
                uint8_t *f_cur = (uint8_t *)malloc((size_t)input_size);
                uint8_t *f_nxt = (uint8_t *)malloc((size_t)input_size);
                assert(f_cur != NULL && f_nxt != NULL);
                reconstruct_filtered_buffer(working_input, f_cur, f_nxt, input_size, format, &best_meta);

                int est_dh = find_best_dh_sentinel(f_cur, input_size, NULL);
                if (best_meta.dh_sentinel == 0) {
                    best_meta.dh_sentinel = est_dh;
                }
                int uniq_ref_dhs[16];
                int n_ref_dhs = rank_top_dh_sentinels(f_cur, input_size, -1, uniq_ref_dhs, max_dh_cands);
                ParsePlan f_plan8 = {NULL, 0, 0};
                if (0 && solve_exact_parse(f_cur, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, &f_plan8)) {
                    int plan_dhs[6];
                    int np = rank_top_dh_sentinels_from_plan(f_cur, input_size, &f_plan8, plan_dhs, 6);
                    for (int pi = 0; pi < np && n_ref_dhs < 16; pi++) {
                        int pdh = plan_dhs[pi];
                        if (pdh == est_dh) continue;
                        int dup = 0;
                        for (int k = 0; k < n_ref_dhs; k++) if (uniq_ref_dhs[k] == pdh) dup = 1;
                        if (!dup) uniq_ref_dhs[n_ref_dhs++] = pdh;
                    }
                    free(f_plan8.tokens);
                }
                int f_lit_bits = detect_buffer_literal_bits(f_cur, input_size, NULL);
                UltraMetadata base_win_meta = best_meta;
                #pragma omp parallel for schedule(dynamic) if (!omp_in_parallel() && n_ref_dhs > 1)
                for (int ri = 0; ri < n_ref_dhs; ri++) {
                    int cdh = uniq_ref_dhs[ri];
                    UltraMetadata ref_meta = base_win_meta;
                    ref_meta.dh_sentinel = cdh;
                    ref_meta.literal_bits = f_lit_bits;
                    evaluate_candidate_meta(
                        f_cur, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                        -cdh, &ref_meta, &best_size, &best_meta, NULL
                    );
                }
                free(f_cur);
                free(f_nxt);
            }
        }

        /*
         * Phase 2 Final Deep Exact Parse & Single Stream Emission:
         * Collect all deep candidate metadata configurations (winning filter, complementary s1/s2
         * or Q3 stepped hybrid variant, and unfiltered fallback if close), evaluate them in a single
         * parallel OpenMP pass directly producing ParsePlan tokens (zero duplicate best_meta parse!),
         * refine DH on the finalized token plan for ULTRA_128B, and emit the winning stream.
         */
        if (top_level_call) g_deep_mode = 1; else tls_deep_mode = 1;
        UltraMetadata deep_cands[8];
        int n_deep = 0;
        deep_cands[n_deep++] = best_meta;
        if (top_level_call && input_size < 8192) {
            for (int ti = 0; ti < g_n_top_cands && n_deep < 4; ti++) {
                if (!same_filter_shape(&g_top_cands[ti], &best_meta) && g_top_sizes[ti] <= best_size + 4) {
                    deep_cands[n_deep++] = g_top_cands[ti];
                }
            }
        }



        int need_alt_s12 = (format != ULTRA_FORMAT_ULTRA_128B &&
                            best_meta.filter_passes == 1 &&
                            (best_meta.filter_stride == 1) &&
                            best_meta.filter_word_modes[0] == 0 &&
                            best_meta.filter_counts[0] >= input_size - 2);
        int need_alt_q3 = ((best_meta.filter_passes == 5 || best_meta.filter_passes == 6) &&
                           best_meta.filter_strides[0] == 0 &&
                           best_meta.filter_word_modes[1] == 1 &&
                           best_meta.filter_word_modes[best_meta.filter_passes - 1] == 0);
        int need_alt_unfilt = (best_meta.filter_passes > 0 && base_unfiltered_size <= best_size + 32);

        if (input_size >= 8192 && best_meta.filter_passes > 0) {
            if (need_alt_s12) {
                int alt_s = 3 - best_meta.filter_stride;
                int alt_cnt = input_size - alt_s;
                if (alt_cnt > 0 && alt_cnt <= 65535) {
                    UltraMetadata alt_m = best_meta;
                    alt_m.filter_stride = alt_s;
                    alt_m.filter_strides[0] = alt_s;
                    alt_m.filter_counts[0] = alt_cnt;
                    deep_cands[n_deep++] = alt_m;
                }
            }
            if (need_alt_q3) {
                int cur_wp = best_meta.filter_passes - 3;
                int alt_wp = (cur_wp == 3) ? 2 : 3;
                int half_t = input_size / 2;
                int wc_tq = best_meta.filter_counts[1 + cur_wp];
                int q3_s = best_meta.filter_strides[2 + cur_wp];
                int c_q3 = best_meta.filter_counts[2 + cur_wp];
                int e8_t = best_meta.filter_counts[0];
                UltraMetadata alt_q3;
                memset(&alt_q3, 0, sizeof(alt_q3));
                alt_q3.format = format;
                alt_q3.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                alt_q3.filter_passes = 3 + alt_wp;
                alt_q3.filter_stride = 0;
                alt_q3.filter_strides[0] = 0;
                alt_q3.filter_counts[0] = e8_t;
                for (int pk = 0; pk < alt_wp; pk++) {
                    alt_q3.filter_strides[1 + pk] = 2;
                    alt_q3.filter_counts[1 + pk] = (half_t - (alt_wp + 1 - pk) * 2) & ~1;
                    alt_q3.filter_word_modes[1 + pk] = 1;
                }
                alt_q3.filter_strides[1 + alt_wp] = 2;
                alt_q3.filter_counts[1 + alt_wp] = wc_tq;
                alt_q3.filter_word_modes[1 + alt_wp] = 1;
                alt_q3.filter_strides[2 + alt_wp] = q3_s;
                alt_q3.filter_counts[2 + alt_wp] = c_q3;
                alt_q3.filter_word_modes[2 + alt_wp] = 0;
                uint8_t *d_cur = (uint8_t *)malloc((size_t)input_size);
                uint8_t *d_nxt = (uint8_t *)malloc((size_t)input_size);
                assert(d_cur != NULL && d_nxt != NULL);
                reconstruct_filtered_buffer(working_input, d_cur, d_nxt, input_size, format, &alt_q3);
                if (d_cur[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                    d_cur[input_size - 1] != X86_OPCODE_CALL_REL16) {
                    alt_q3.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                        ? detect_buffer_literal_bits(d_cur, input_size, NULL) : 8;
                    deep_cands[n_deep++] = alt_q3;
                }
                free(d_cur);
                free(d_nxt);
            }
            if (need_alt_unfilt) {
                UltraMetadata umeta;
                memset(&umeta, 0, sizeof(umeta));
                umeta.format = format;
                umeta.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
                umeta.dh_sentinel = (format == ULTRA_FORMAT_ULTRA_128B)
                                  ? find_best_dh_sentinel(working_input, input_size, NULL) : 0;
                umeta.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                   ? detect_buffer_literal_bits(working_input, input_size, NULL) : 8;
                deep_cands[n_deep++] = umeta;
            }
        }

        ParsePlan deep_plans[8];
        size_t deep_sizes[8];
        for (int di = 0; di < n_deep; di++) {
            deep_plans[di].tokens = NULL;
            deep_plans[di].token_count = 0;
            deep_plans[di].payload_bits = 0;
            deep_sizes[di] = (size_t)-1;
        }

        #pragma omp parallel for schedule(dynamic) if (!omp_in_parallel() && n_deep > 1)
        for (int di = 0; di < n_deep; di++) {
            uint8_t *d_cur = NULL;
            uint8_t *d_nxt = NULL;
            const uint8_t *d_buf = working_input;
            if (deep_cands[di].filter_passes > 0) {
                d_cur = (uint8_t *)malloc((size_t)input_size);
                d_nxt = (uint8_t *)malloc((size_t)input_size);
                assert(d_cur != NULL && d_nxt != NULL);
                reconstruct_filtered_buffer(working_input, d_cur, d_nxt, input_size, format, &deep_cands[di]);
                d_buf = d_cur;
            }
            int d_req_off = 0;
            if (format == ULTRA_FORMAT_ULTRA_128B) {
                if (deep_cands[di].dh_sentinel <= 0 || deep_cands[di].dh_sentinel > 127) {
                    deep_cands[di].dh_sentinel = find_best_dh_sentinel(d_buf, input_size, NULL);
                }
                d_req_off = -deep_cands[di].dh_sentinel;
            } else if (deep_cands[di].filter_passes > 0 && !is_multi_stage && deep_cands[di].filter_stride > 0) {
                d_req_off = deep_cands[di].filter_stride;
            }
            int ok = solve_exact_parse(
                d_buf, input_size, NULL, deep_cands[di].offset_lsb_bits, effective_limit, d_req_off, &deep_plans[di]
            );
            if (ok) {
                if (format == ULTRA_FORMAT_ULTRA_128B) {
                    optimize_plan_dh_sentinel(d_buf, input_size, &deep_plans[di], &deep_cands[di].dh_sentinel);
                }
                deep_cands[di].compressed_bits = deep_plans[di].payload_bits;
                deep_sizes[di] = compute_exact_stream_size(deep_plans[di].payload_bits, format, &deep_cands[di]);
            }
            free(d_cur);
            free(d_nxt);
        }

        int win_di = 0;
        for (int di = 1; di < n_deep; di++) {
            if (deep_sizes[di] < deep_sizes[win_di] ||
                (deep_sizes[di] == deep_sizes[win_di] &&
                 deep_cands[di].compressed_bits < deep_cands[win_di].compressed_bits)) {
                win_di = di;
            }
        }
        assert(deep_plans[win_di].tokens != NULL);
        best_meta = deep_cands[win_di];

        uint8_t *win_buf = working_input;
        uint8_t *f_cur = NULL;
        uint8_t *f_nxt = NULL;
        if (best_meta.filter_passes > 0) {
            f_cur = (uint8_t *)malloc((size_t)input_size);
            f_nxt = (uint8_t *)malloc((size_t)input_size);
            assert(f_cur != NULL && f_nxt != NULL);
            reconstruct_filtered_buffer(working_input, f_cur, f_nxt, input_size, format, &best_meta);
            win_buf = f_cur;
        }
        best_stream = encode_plan_to_stream(
            win_buf, &deep_plans[win_di], format, best_meta.offset_lsb_bits, &best_meta, &best_size
        );
        for (int di = 0; di < n_deep; di++) {
            free(deep_plans[di].tokens);
        }
        free(f_cur);
        free(f_nxt);
    }

    assert(best_stream != NULL);

    /* Mandatory round-trip self-verification against working_input */
    uint8_t *verified = ultra_decompress_buffer(best_stream, best_size, (size_t)input_size, format);
    assert(verified != NULL);
    assert(memcmp(verified, working_input, (size_t)input_size) == 0);
    free(verified);
    free(working_input);

    if (!top_level_call) tls_deep_mode = -1;
    *compressed_size_out = best_size;
    if (metadata_out != NULL) {
        *metadata_out = best_meta;
    }
    return best_stream;
}
