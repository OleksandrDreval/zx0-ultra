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
        int lc = lit_frontiers[off].count;
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
        int lc = lit_frontiers[off].count;
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
    frontier->count = c;
    if (c > 0 && (base_cost - frontier->entries[0].base_cost) > 32) {
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
    int cap_n;
    int cap_w;
    NodePool pool;
    int *lit_cum_bits;
    int *prev_same_byte;
    int *run_start;
    int *best_any_cost;
    int *best_any_node;
    int *best_len_cost;
    int *best_len_start;
    int blk64_cost[1024];
    int blk64_pos[1024];
    int blk1024_cost[64];
    int blk1024_pos[64];
    MonotonicFrontier *lit_frontiers;
    MonotonicFrontier *rep_frontiers;
    int *lit_min_base;
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

static void dp_workspace_ensure(DpWorkspace *ws, int n, int w) {
    if (n + 1 > ws->cap_n) {
        int new_n = (n + 1) > 4096 ? (n + 1) : 4096;
        ws->lit_cum_bits   = (int *)realloc(ws->lit_cum_bits,   (size_t)new_n * sizeof(int));
        ws->prev_same_byte = (int *)realloc(ws->prev_same_byte, (size_t)new_n * sizeof(int));
        ws->run_start      = (int *)realloc(ws->run_start,      (size_t)new_n * sizeof(int));
        ws->best_any_cost  = (int *)realloc(ws->best_any_cost,  (size_t)new_n * sizeof(int));
        ws->best_any_node  = (int *)realloc(ws->best_any_node,  (size_t)new_n * sizeof(int));
        ws->best_len_cost  = (int *)realloc(ws->best_len_cost,  (size_t)new_n * sizeof(int));
        ws->best_len_start = (int *)realloc(ws->best_len_start, (size_t)new_n * sizeof(int));
        assert(ws->lit_cum_bits && ws->prev_same_byte && ws->run_start && ws->best_any_cost &&
               ws->best_any_node && ws->best_len_cost && ws->best_len_start);
        ws->cap_n = new_n;
    }
    if (w + 1 > ws->cap_w) {
        int new_w = (w + 1) > 4096 ? (w + 1) : 4096;
        ws->lit_frontiers   = (MonotonicFrontier *)realloc(ws->lit_frontiers,   (size_t)new_w * sizeof(MonotonicFrontier));
        ws->rep_frontiers   = (MonotonicFrontier *)realloc(ws->rep_frontiers,   (size_t)new_w * sizeof(MonotonicFrontier));
        ws->lit_min_base    = (int *)realloc(ws->lit_min_base,    (size_t)new_w * sizeof(int));
        ws->new_off_overhead = (int *)realloc(ws->new_off_overhead, (size_t)new_w * sizeof(int));
        ws->match_lengths   = (int *)realloc(ws->match_lengths,   (size_t)new_w * sizeof(int));
        ws->last_match_pos  = (int *)realloc(ws->last_match_pos,  (size_t)new_w * sizeof(int));
        ws->rep_valid_pos   = (int *)realloc(ws->rep_valid_pos,   (size_t)new_w * sizeof(int));
        ws->final_lit_cost  = (int *)realloc(ws->final_lit_cost,  (size_t)new_w * sizeof(int));
        ws->final_lit_node  = (int *)realloc(ws->final_lit_node,  (size_t)new_w * sizeof(int));
        ws->final_mat_cost  = (int *)realloc(ws->final_mat_cost,  (size_t)new_w * sizeof(int));
        ws->final_mat_node  = (int *)realloc(ws->final_mat_node,  (size_t)new_w * sizeof(int));
        assert(ws->lit_frontiers && ws->rep_frontiers && ws->lit_min_base &&
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
     * and 2-byte (short-range match) hash tables, then evaluate all 127 DH sentinels in O(127 * 256).
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

    for (int dh = 1; dh <= 127; dh++) {
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

    for (int dh = 1; dh <= 127; dh++) {
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
    for (int dh = 1; dh <= 127; dh++) {
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

    const int seeds[11] = {whole_dh, 2, 3, 4, 8, 9, 10, 11, 16, 32, 64};
    for (int si = 0; si < 11 && count < max_out; si++) {
        int sdh = seeds[si];
        if (sdh == exclude_dh || sdh < 1 || sdh > 127) {
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
    for (int dh = 1; dh <= 127; dh++) {
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
    int *run_start      = ws->run_start;
    int *best_any_cost  = ws->best_any_cost;
    int *best_any_node  = ws->best_any_node;
    int *best_len_cost  = ws->best_len_cost;
    int *best_len_start = ws->best_len_start;
    MonotonicFrontier *lit_frontiers = ws->lit_frontiers;
    MonotonicFrontier *rep_frontiers = ws->rep_frontiers;
    int *lit_min_base     = ws->lit_min_base;
    int *new_off_overhead = ws->new_off_overhead;
    int *match_lengths    = ws->match_lengths;
    int *last_match_pos   = ws->last_match_pos;
    int *rep_valid_pos    = ws->rep_valid_pos;
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
    }

    new_off_overhead[0] = 0;
    for (int off = 0; off <= max_window; off++) {
        if (off >= 1) {
            new_off_overhead[off] = compute_new_offset_overhead_bits(off, offset_lsb_bits);
        }
        lit_frontiers[off].count = 0;
        rep_frontiers[off].count = 0;
        lit_min_base[off] = COST_INFINITY;
        match_lengths[off] = 0;
        last_match_pos[off] = 0;
        rep_valid_pos[off] = -1;
        final_lit_cost[off] = COST_INFINITY;
        final_lit_node[off] = -1;
        final_mat_cost[off] = COST_INFINITY;
        final_mat_node[off] = -1;
    }

    int need_tokens = (plan_out != NULL);
    int gc_threshold = NODE_POOL_GC_THRESHOLD;

    /* Seed initial state at pos = 0 with base_cost = -1 so first literal block has 0 control bits */
    if (dh_mode == 0) {
        frontier_push(&lit_frontiers[ZX0_INITIAL_OFFSET], 0, -1, -1);
        lit_min_base[ZX0_INITIAL_OFFSET] = -1;
    }
    frontier_push(&lit_frontiers[0], 0, -1, -1);
    lit_min_base[0] = -1;
    best_any_cost[0] = COST_INFINITY;
    best_any_node[0] = -1;
    ws->blk64_cost[0] = COST_INFINITY;
    ws->blk64_pos[0] = 0;
    ws->blk1024_cost[0] = COST_INFINITY;
    ws->blk1024_pos[0] = 0;

    for (int pos = 1; pos <= input_size; pos++) {
        if (__builtin_expect(need_tokens && pool->count >= gc_threshold, 0)) {
            node_pool_compact(
                pool, best_any_node, pos - 1, lit_frontiers, rep_frontiers, rep_valid_pos, max_window
            );
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
            int min_idx2 = pos - loop_max_off;
            int idx1 = prev_same_byte[pos - 1];
            int L0 = run_start[pos - 1];
            int run0 = (pos - 1) - L0 + 1;
            uint8_t cur_byte = input_data[pos - 1];
            uint8_t nxt_byte = (pos < input_size) ? input_data[pos] : 0;
            uint8_t prv_byte = (pos >= 2) ? input_data[pos - 2] : 0;
            int cur_continues = (pos < input_size && nxt_byte == cur_byte);

            int deep = (input_size < 8192) || g_deep_mode;
            int chain_steps = 0;
            int match_steps = 0;
            int max_seen_ml = 1;
            int max_chain_steps = deep ? 32640 : 4096;
            int max_match_steps = deep ? 8192 : 512;
            int raw_pad = deep ? 64 : 16;
            int mat_pad = deep ? 64 : 24;
            int f_pad = deep ? 128 : 48;

            while (idx1 >= min_idx1) {
                if (++chain_steps > max_chain_steps) {
                    break;
                }
                int cur_idx1 = idx1;
                int off = (pos - 1) - cur_idx1;
                idx1 = prev_same_byte[cur_idx1];

                int ml;
                int is_rep_active = 0;

                if (run0 == 1 && (cur_idx1 == 0 || input_data[cur_idx1 - 1] != prv_byte)) {
                    int L1 = run_start[cur_idx1];
                    if (cur_idx1 > L1 + 1) {
                        int scan = cur_idx1 - 1;
                        while (scan > L1 && rep_valid_pos[(pos - 1) - scan] != pos - 1) {
                            scan--;
                        }
                        idx1 = scan;
                    }
                    ml = 1;
                    match_lengths[off] = 1;
                    last_match_pos[off] = pos;
                    if (rep_valid_pos[off] == pos - 1) {
                        rep_valid_pos[off] = pos;
                        is_rep_active = 1;
                    }
                } else {
                    match_steps++;
                    int L1 = run_start[cur_idx1];
                    int run1 = cur_idx1 - L1 + 1;
                    if (cur_continues) {
                        if (L1 == L0) {
                            if (cur_idx1 < pos - 2) {
                                idx1 = prev_same_byte[L0];
                            }
                        } else if (run1 > 3 && cur_idx1 > L1 + 1) {
                            int target = (run1 > run0) ? (L1 + run0 - 1) : (L1 + 1);
                            if (cur_idx1 > target) {
                                idx1 = target;
                            }
                        }
                    }
                    if (run1 < run0) {
                        ml = run1;
                    } else if (run1 > run0) {
                        ml = run0;
                    } else {
                        ml = (last_match_pos[off] >= L0)
                           ? (match_lengths[off] + (pos - last_match_pos[off]))
                           : (run0 == 1 ? 2 : run0);
                    }
                    match_lengths[off] = ml;
                    last_match_pos[off] = pos;

                    if (rep_valid_pos[off] >= pos - ml) {
                        rep_valid_pos[off] = pos;
                        is_rep_active = 1;
                    }
                }

                if (ml == 1 && !is_rep_active) {
                    continue;
                }
                if (match_steps > max_match_steps && ml <= max_seen_ml && !is_rep_active) {
                    continue;
                }
                if (ml > max_seen_ml) {
                    max_seen_ml = ml;
                }

                /*
                 * Evaluate Match (LAST_OFFSET or NEW_OFFSET) ending at pos with offset = off.
                 * By the Left-Maximal LAST_OFFSET Match Theorem (since every literal byte costs >= 3 bits
                 * across all formats 67B..128B while extending a LAST_OFFSET match by 1 byte costs <= 2 bits),
                 * no optimal parse ever transitions from a literal run to a LAST_OFFSET match at pos when
                 * cur_matches == 1 (Pass 1); all literal-to-LAST_OFFSET transitions occur in Pass 2.
                 */
                int min_base = lit_min_base[off];
                int mat_cost = COST_INFINITY;
                int mat_prev_node = -1;
                int max_viable_mat = (best_any_cost[pos] < COST_INFINITY)
                                   ? (best_any_cost[pos] + new_off_overhead[off] + mat_pad)
                                   : COST_INFINITY;

                if (is_rep_active) {
                    const MonotonicFrontier *rf = &rep_frontiers[off];
                    if (rf->entries[0].base_cost + 2 <= max_viable_mat) {
                        int rep_src_node = -1;
                        int rep_cost = frontier_evaluate_min(rf, pos, &rep_src_node);
                        if (rep_cost < mat_cost) {
                            mat_cost = rep_cost;
                            mat_prev_node = rep_src_node;
                        }
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
                        int new_off_cost = best_c + new_off_overhead[off];
                        if (new_off_cost < mat_cost) {
                            mat_cost = new_off_cost;
                            mat_prev_node = best_any_node[best_s];
                        }
                    }
                }

                if (mat_cost <= max_viable_mat && mat_cost < COST_INFINITY) {
                    int mat_base_cost = mat_cost - literal_pos_bits;
                    int enters_lit_frontier = (pos < input_size) &&
                                              ((min_base == COST_INFINITY) || ((mat_base_cost - min_base) <= f_pad));
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
                            frontier_push(&lit_frontiers[off], pos, mat_base_cost, created_mat_node);
                            lit_min_base[off] = lit_frontiers[off].entries[0].base_cost;
                        }
                    }
                }
            }

            /* Pass 2: Offsets with cur_matches == 0 && next_pos_matches == 1 */
            int idx2 = (pos < input_size) ? prev_same_byte[pos] : -1;
            int pass2_steps = 0;
            int max_pass2_steps = deep ? 32640 : (cur_continues ? 512 : 2048);
            int max_any_raw_allowed = (glob_raw < COST_INFINITY)
                                    ? (glob_raw + new_off_overhead[loop_max_off] + raw_pad)
                                    : COST_INFINITY;
            if (cur_continues) {
                /*
                 * When nxt_byte == cur_byte, every interior run position k > run_start[idx2]
                 * has input_data[k - 1] == cur_byte (already processed in Pass 1).
                 * Only the start of each run L2 = run_start[idx2] has cur_matches == 0!
                 */
                while (idx2 >= min_idx2) {
                    int L2 = run_start[idx2];
                    if (L2 >= min_idx2) {
                        int off = pos - L2;
                        int max_raw_allowed = (glob_raw < COST_INFINITY)
                                            ? (glob_raw + new_off_overhead[off] + raw_pad)
                                            : COST_INFINITY;
                        if (lit_min_base[off] <= max_raw_allowed) {
                            if (++pass2_steps > max_pass2_steps) {
                                break;
                            }
                            const MonotonicFrontier *lf = &lit_frontiers[off];
                            int lc = lf->count;
                            int min_elias = compute_elias_gamma_bits(pos - lf->entries[lc - 1].pos);
                            int raw_eval = COST_INFINITY;
                            int lit_src_node_pass2 = -1;
                            if (lit_min_base[off] + 1 + min_elias <= max_raw_allowed ||
                                lf->entries[lc - 1].base_cost + 1 + min_elias <= max_raw_allowed) {
                                raw_eval = frontier_evaluate_min(lf, pos, &lit_src_node_pass2);
                            }
                            if (raw_eval <= max_raw_allowed) {
                                int lit_cost = raw_eval + literal_pos_bits;
                                int created_lit_node = need_tokens ? node_pool_push(pool, lit_src_node_pass2, pos, 0) : 0;
                                if (lit_cost < best_any_cost[pos]) {
                                    best_any_cost[pos] = lit_cost;
                                    best_any_node[pos] = created_lit_node;
                                }
                                rep_frontiers[off].count = 0;
                                frontier_push(&rep_frontiers[off], pos, lit_cost, created_lit_node);
                                if (rep_frontiers[off].count > 0) {
                                    rep_valid_pos[off] = pos;
                                }
                            }
                        }
                    }
                    idx2 = prev_same_byte[L2];
                }
            } else {
                while (idx2 >= min_idx2) {
                    int cur_idx2 = idx2;
                    idx2 = prev_same_byte[cur_idx2];
                    int L2 = run_start[cur_idx2];
                    if (cur_idx2 > L2 + 1) {
                        int scan = cur_idx2 - 1;
                        while (scan > L2 && lit_min_base[pos - scan] > max_any_raw_allowed) {
                            scan--;
                        }
                        idx2 = scan;
                    }
                    int off = pos - cur_idx2;
                    int max_raw_allowed = (glob_raw < COST_INFINITY)
                                        ? (glob_raw + new_off_overhead[off] + raw_pad)
                                        : COST_INFINITY;
                    if (lit_min_base[off] > max_raw_allowed) {
                        continue;
                    }
                    if (off <= active_max_off && input_data[cur_idx2 - 1] == cur_byte) {
                        continue;
                    }
                    if (++pass2_steps > max_pass2_steps) {
                        break;
                    }
                    const MonotonicFrontier *lf = &lit_frontiers[off];
                    int lc = lf->count;
                    int min_elias = compute_elias_gamma_bits(pos - lf->entries[lc - 1].pos);
                    int raw_eval = COST_INFINITY;
                    int lit_src_node = -1;
                    if (lit_min_base[off] + 1 + min_elias <= max_raw_allowed ||
                        lf->entries[lc - 1].base_cost + 1 + min_elias <= max_raw_allowed) {
                        raw_eval = frontier_evaluate_min(lf, pos, &lit_src_node);
                    }
                    if (raw_eval <= max_raw_allowed) {
                        int lit_cost = raw_eval + literal_pos_bits;
                        int created_lit_node = need_tokens ? node_pool_push(pool, lit_src_node, pos, 0) : 0;
                        if (lit_cost < best_any_cost[pos]) {
                            best_any_cost[pos] = lit_cost;
                            best_any_node[pos] = created_lit_node;
                        }
                        rep_frontiers[off].count = 0;
                        frontier_push(&rep_frontiers[off], pos, lit_cost, created_lit_node);
                        if (rep_frontiers[off].count > 0) {
                            rep_valid_pos[off] = pos;
                        }
                    }
                }
            }
        } else {
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
                            ml = (last_match_pos[off] >= L0)
                               ? (match_lengths[off] + (pos - last_match_pos[off]))
                               : run0;
                        }
                    } else {
                        ml = (last_match_pos[off] == pos - 1) ? (match_lengths[off] + 1) : 1;
                    }
                    match_lengths[off] = ml;
                    last_match_pos[off] = pos;
                    if (wildcard_mask == NULL ? (rep_valid_pos[off] >= pos - ml) : (rep_valid_pos[off] == pos - 1)) {
                        rep_valid_pos[off] = pos;
                        is_rep_active = 1;
                    }
                } else {
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
                        int new_off_cost = best_c + new_off_overhead[off];
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
                            frontier_push(&lit_frontiers[off], pos, mat_base_cost, created_mat_node);
                            lit_min_base[off] = lit_frontiers[off].entries[0].base_cost;
                        }
                    }
                }

                /* Push literal state after match evaluation so LAST_OFFSET matches have length >= 1 */
                if (created_lit_node >= 0 && next_pos_matches) {
                    if (rep_valid_pos[off] != pos) {
                        rep_frontiers[off].count = 0;
                    }
                    frontier_push(&rep_frontiers[off], pos, lit_cost, created_lit_node);
                    if (rep_frontiers[off].count > 0) {
                        rep_valid_pos[off] = pos;
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
                int st = meta->filter_strides[p];
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

    if (format == ULTRA_FORMAT_ZX0_V2) {
        g_deep_mode = 1;
        ParsePlan plan = {NULL, 0, 0};
        int ok = solve_exact_parse(working_input, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, &plan);
        assert(ok);
        best_meta.compressed_bits = plan.payload_bits;
        best_stream = encode_plan_to_stream(working_input, &plan, format, ZX0_DEFAULT_LSB_BITS, &best_meta, &best_size);
        free(plan.tokens);
    } else if (format == ULTRA_FORMAT_ADPK_77B) {
        g_deep_mode = (input_size < 8192);
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
        g_deep_mode = 1;
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
        g_deep_mode = (input_size < 8192);
        int is_word_capable = (format == ULTRA_FORMAT_ULTRA_128B || format == ULTRA_FORMAT_PRO_111B);
        int is_multi_stage  = (format == ULTRA_FORMAT_HYBRID_108B || is_word_capable);
        int base_req_off = (format == ULTRA_FORMAT_ULTRA_128B) ? -200 : 0;
        int base_payload_bits = 0;
        {
            UltraMetadata base_meta;
            memset(&base_meta, 0, sizeof(base_meta));
            base_meta.format = format;
            base_meta.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
            base_meta.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                                   ? detect_buffer_literal_bits(working_input, input_size, NULL) : 8;
            base_payload_bits = evaluate_candidate_meta(
                working_input, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                base_req_off, &base_meta, &best_size, &best_meta, NULL
            );
            assert(base_payload_bits > 0);
        }
        size_t base_unfiltered_size = best_size;

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
            #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
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

                    /* For HYBRID_108B, PRO_111B: also evaluate chaining E8 (CALL rel16) + E9 (JMP rel16) */
                    if (is_multi_stage && format != ULTRA_FORMAT_ULTRA_128B &&
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
                            evaluate_candidate_meta(
                                e8e9_buf, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                base_req_off, &meta_e89, &best_size, &best_meta, NULL
                            );
                        }
                        free(e9_tmp);
                        free(e8e9_buf);
                    }
                }
                free(e8_buf);
            }
        }

        /* B. Multi-core search over candidate strides (STRD_80B, COMPAT_82B, STRD_82B, UNIV_93B, HYBRID_108B, PRO_111B, ULTRA_128B) */
        if (format != ULTRA_FORMAT_X86E8_84B) {
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

            for (int rsi = 0; rsi < n_raw_strides; rsi++) {
                int stride = raw_strides[rsi];
                int max_full_count = input_size - stride;
                int full_count = max_full_count > 65535 ? 65535 : max_full_count;
                if (full_count <= 0) {
                    continue;
                }
                int kadane_gain = 0;
                find_optimal_stride_tail_count(working_input, input_size, stride, &kadane_gain, NULL);
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
                int df_gain = eq_dif - eq_raw;
                if ((stride > 2 && kadane_gain < 16 && df_gain < (input_size - stride) / 16) ||
                    (stride > 8 && kadane_gain < 16 && !is_structural_stride && df_gain < (input_size - stride) / 12) ||
                    (input_size >= 4096 && kadane_gain < 16 && df_gain < 0)) {
                    continue;
                }
                int dominated_harmonic = 0;
                if (stride > 24) {
                    for (int prev_i = 0; prev_i < num_strides; prev_i++) {
                        int d = candidate_strides[prev_i];
                        if (d >= 2 && stride > d && (stride % d == 0) &&
                            cand_kg[prev_i] >= 300 &&
                            kadane_gain <= cand_kg[prev_i] + 64 &&
                            df_gain <= cand_df[prev_i] + 64) {
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

            if (input_size >= 8192 && num_strides > 12) {
                for (int i = 0; i < num_strides; i++) {
                    for (int j = i + 1; j < num_strides; j++) {
                        int si_score = cand_df[i] + cand_kg[i] * 2 + ((candidate_strides[i] <= 4 || candidate_strides[i] == 16) ? 300 : 0);
                        int sj_score = cand_df[j] + cand_kg[j] * 2 + ((candidate_strides[j] <= 4 || candidate_strides[j] == 16) ? 300 : 0);
                        if (sj_score > si_score) {
                            int ts = candidate_strides[i]; candidate_strides[i] = candidate_strides[j]; candidate_strides[j] = ts;
                            int tk = cand_kg[i]; cand_kg[i] = cand_kg[j]; cand_kg[j] = tk;
                            int td = cand_df[i]; cand_df[i] = cand_df[j]; cand_df[j] = td;
                        }
                    }
                }
                int cap_s = (input_size >= 24576) ? 12 : 14;
                if (num_strides > cap_s) {
                    num_strides = cap_s;
                }
            }

            int max_passes = (format == ULTRA_FORMAT_STRD_82B || is_multi_stage)
                           ? (input_size >= 16384 ? 4 : 5) : 1;
            int nested_parallel = omp_in_parallel();

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
                    apply_forward_stride_difference(cur_buf, nxt_buf, input_size, stride, full_count);
                    if (memcmp(nxt_buf, cur_buf, (size_t)input_size) == 0 &&
                        (pass > 1 || (format != ULTRA_FORMAT_STRD_80B && format != ULTRA_FORMAT_STRD_82B) || stride > 1)) {
                        break;
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
                        meta_p.filter_counts[p_idx] = full_count;
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
                if (max_passes > 1 && stride <= 32 && pass2_full_bits < COST_INFINITY && pass2_full_bits <= pass1_full_bits + 64) {
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

                /* Evaluate piecewise half-tail, Kadane-optimal tail, unbiased Kadane tail, and aligned tail */
                int min_pw_gap = (full_count / 4) > 32 ? (full_count / 4) : 32;
                int min_pw_tail = (stride + 32) > 64 ? (stride + 32) : 64;
                int allow_piecewise = (pass1_full_bits * 2 >= base_payload_bits) &&
                                      (input_size < 8192 ? (kadane_gain >= 32) : (si < 4 && kadane_gain >= 64)) &&
                                      ((kadane_tail >= min_pw_tail && kadane_tail < full_count - min_pw_gap) ||
                                       (unbiased_tail >= min_pw_tail && unbiased_tail < full_count - min_pw_gap));
                int cand_piecewise[10];
                int num_piecewise = 0;
                int half_count = input_size / 2;
                if (allow_piecewise && !nested_parallel && input_size >= 128 &&
                    half_count >= min_pw_tail && half_count < full_count) {
                    cand_piecewise[num_piecewise++] = half_count;
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
                    if (!dup && num_piecewise < 10) {
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

            /* C0. PRO_111B & ULTRA_128B Native 16-Bit Word-Stride-Delta Search (eliminates carry-ripple on 16-bit PCM/Coord/Ptr tables) */
            if (is_word_capable && input_size >= 4 && best_meta.compressed_bits > 200) {
                static const int word_strides[] = {
                    2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                    32, 36, 40, 48, 52, 60, 64, 72, 80, 96, 100, 112, 120, 128
                };
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
                    {
                        int eq_w = 0, eq_raw_w = 0;
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
                        }
                        if ((wstride <= 4 && eq_w <= eq_raw_w + (input_size >= 8192 ? (input_size - wstride) / 24 : 0)) ||
                            (wstride > 4 && eq_w <= eq_raw_w + (input_size - wstride) / (input_size >= 8192 ? 16 : 32))) {
                            continue;
                        }
                    }
                    uint8_t *wcur = (uint8_t *)malloc((size_t)input_size);
                    uint8_t *wnxt = (uint8_t *)malloc((size_t)input_size);
                    assert(wcur != NULL && wnxt != NULL);

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
                        apply_forward_word16_stride_difference(wcur, wnxt, input_size, wstride, max_wcount);
                        if (memcmp(wnxt, wcur, (size_t)input_size) == 0) {
                            break;
                        }
                        memcpy(wcur, wnxt, (size_t)input_size);
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
                            if (wbits > base_payload_bits + (input_size >= 8192 ? 512 : 1024) ||
                                (pass >= 2 && wbits >= prev_wbits)) {
                                break;
                            }
                            prev_wbits = wbits;
                        } else {
                            break;
                        }
                    }

                    /* 2) Harmonic 16-bit Word-Stride-Delta (wcount_p = ((input_size - pass * wstride) & ~1)) */
                    if (wstride <= 32 && wpass2_bits < COST_INFINITY && wpass2_bits <= wpass1_bits + 64) {
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
                                if (wh_bits <= 0 || wh_bits >= prev_wh_bits) {
                                    break;
                                }
                                prev_wh_bits = wh_bits;
                            }
                        }
                    }

                    /* 3) Hybrid 16-bit Word-Stride (Order 1..3) followed by 8-bit Byte-Stride (stride=1, 2, or wstride) */
                    int max_hyb_ws = 32;
                    if (wstride <= max_hyb_ws && prev_wbits < base_payload_bits + 512) {
                        memcpy(wcur, working_input, (size_t)input_size);
                        for (int wpass = 1; wpass <= 3; wpass++) {
                            if (wpass >= 2 && wpass2_bits > wpass1_bits + 256) break;
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
                                /* Decoder runs reverse: first 8-bit byte-add (bs), then wpass 16-bit word-add (wstride) */
                                meta_wb.filter_strides[0] = bs;
                                meta_wb.filter_counts[0] = bc;
                                meta_wb.filter_word_modes[0] = 0;
                                for (int pk = 0; pk < wpass; pk++) {
                                    meta_wb.filter_strides[1 + pk] = wstride;
                                    int wc_pk = (input_size - (wpass - pk) * wstride) & ~1;
                                    meta_wb.filter_counts[1 + pk] = wc_pk > 65534 ? 65534 : wc_pk;
                                    meta_wb.filter_word_modes[1 + pk] = 1;
                                }
                                evaluate_candidate_meta(
                                    wnxt, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                    stride_req_off, &meta_wb, &best_size, &best_meta, NULL
                                );
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

                #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
                for (int i1 = 0; i1 < n_s1; i1++) {
                    int s1 = pair_s1[i1];
                    if (s1 >= input_size || s1 > max_allowed_stride) continue;
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
                        if (s2 == s1 || s2 >= input_size || s2 > max_allowed_stride) continue;
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
                        evaluate_candidate_meta(
                            buf_p2, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                            stride_req_off, &meta_2d, &best_size, &best_meta, NULL
                        );
                    }

                    /* 3-pass heterogeneous (s1, s1, s2) for s1 in {1, 2} and s2 in {6, 8, 12, 16, 24, 26, 32} */
                    if ((s1 == 1 || s1 == 2) && input_size > 2 * s1 + 32) {
                        int c1b = (input_size - 2 * s1) > 65535 ? 65535 : (input_size - 2 * s1);
                        apply_forward_stride_difference(buf_p1, buf_p2, input_size, s1, c1b);
                        if (memcmp(buf_p2, buf_p1, (size_t)input_size) != 0) {
                            static const int s3_cands[] = {6, 8, 12, 16, 24, 26, 32};
                            for (size_t i3 = 0; i3 < sizeof(s3_cands) / sizeof(s3_cands[0]); i3++) {
                                int s2 = s3_cands[i3];
                                int c2 = (input_size - 2 * s1 - s2) > 65535 ? 65535 : (input_size - 2 * s1 - s2);
                                if (c2 <= 0) continue;
                                apply_forward_stride_difference(buf_p2, buf_p3, input_size, s2, c2);
                                if (memcmp(buf_p3, buf_p2, (size_t)input_size) == 0) continue;
                                int eq_p2b = 0, eq_p3b = 0;
                                for (int k = s2 + 1; k < input_size; k++) {
                                    if (buf_p2[k] == 0 || buf_p2[k] == buf_p2[k - 1]) eq_p2b++;
                                    if (buf_p3[k] == 0 || buf_p3[k] == buf_p3[k - 1] || buf_p3[k] == buf_p3[k - s2]) eq_p3b++;
                                }
                                if (eq_p2b < (input_size - s2) / 8 || eq_p3b <= eq_p2b + (input_size - s2) / 16) continue;
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
                                evaluate_candidate_meta(
                                    buf_p3, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                    stride_req_off, &meta_3d, &best_size, &best_meta, NULL
                                );
                            }
                        }
                    }

                    free(buf_p1);
                    free(buf_p2);
                    free(buf_p3);
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
                        #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
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
                                if (buf_w1[k] == 0 || buf_w1[k] == buf_w1[k - 2]) eq_w1++;
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
                            evaluate_candidate_meta(
                                buf_w2, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                stride_req_off, &meta_w2d, &best_size, &best_meta, NULL
                            );
                            free(buf_w2);
                        }
                    }
                    free(buf_w1);
                }

                if (best_e8_buf != NULL && best_e8_tail_chosen >= 3) {
                    static const int hyb_strides[] = {1, 2, 3, 4, 8, 12, 16, 24, 32, 64};

                    /* Evaluate E8 + Multi-Segment Stepped-Tail Word/Byte Stride-Delta on hybrid ROMs */
                    if (input_size >= 512) {
                        int half_t = input_size / 2;
                        int three_q_t = (3 * input_size) / 4;
                        int quarter_t = input_size / 4;
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
                                if (wc < 4) break;
                                apply_forward_word16_stride_difference(w_cur, st_buf, input_size, 2, wc);
                                memcpy(w_cur, st_buf, (size_t)input_size);
                                if (w_cur[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                    w_cur[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                    int h8w = apply_forward_x86_opcode_filter(
                                        w_cur, e8_on_st, input_size, best_e8_tail_chosen, X86_OPCODE_CALL_REL16
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
                                        meta_ew.filter_counts[0] = best_e8_tail_chosen;
                                        for (int pk = 0; pk < wp; pk++) {
                                            meta_ew.filter_strides[1 + pk] = 2;
                                            meta_ew.filter_counts[1 + pk] = (half_t - (wp - pk) * 2) & ~1;
                                            meta_ew.filter_word_modes[1 + pk] = 1;
                                        }
                                        evaluate_candidate_meta(
                                            e8_on_st, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                            stride_req_off, &meta_ew, &best_size, &best_meta, NULL
                                        );
                                    }
                                }
                            }

                            /* Stepped-tail multi-order 16-bit word-delta: 1 pass on 3*N/4 + 1..2 passes on N/2 + E8 */
                            int wc_tq = (three_q_t - 2) & ~1;
                            if (wc_tq >= 8) {
                                apply_forward_word16_stride_difference(working_input, w_cur, input_size, 2, wc_tq);
                                for (int wp = 1; wp <= 2; wp++) {
                                    int wc_h = (half_t - (wp + 1) * 2) & ~1;
                                    if (wc_h < 4) break;
                                    apply_forward_word16_stride_difference(w_cur, st_buf, input_size, 2, wc_h);
                                    memcpy(w_cur, st_buf, (size_t)input_size);
                                    if (w_cur[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                        w_cur[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                        int h8w = apply_forward_x86_opcode_filter(
                                            w_cur, e8_on_st, input_size, best_e8_tail_chosen, X86_OPCODE_CALL_REL16
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
                                            meta_st.filter_counts[0] = best_e8_tail_chosen;
                                            for (int pk = 0; pk < wp; pk++) {
                                                meta_st.filter_strides[1 + pk] = 2;
                                                meta_st.filter_counts[1 + pk] = (half_t - (wp + 1 - pk) * 2) & ~1;
                                                meta_st.filter_word_modes[1 + pk] = 1;
                                            }
                                            meta_st.filter_strides[1 + wp] = 2;
                                            meta_st.filter_counts[1 + wp] = wc_tq;
                                            meta_st.filter_word_modes[1 + wp] = 1;
                                            evaluate_candidate_meta(
                                                e8_on_st, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                                stride_req_off, &meta_st, &best_size, &best_meta, NULL
                                            );
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
                                            if (eq_d * 10 >= best_q3_eq * 9) {
                                                best_q3_s = d;
                                                best_q3_eq = eq_d;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if (best_q3_s >= 3 && best_q3_eq >= quarter_t / 4) {
                                    int c_q3 = quarter_t - best_q3_s;
                                    apply_forward_stride_difference(working_input, st_buf, input_size, best_q3_s, c_q3);
                                    apply_forward_word16_stride_difference(st_buf, w_cur, input_size, 2, wc_tq);
                                    for (int wp = 1; wp <= 2; wp++) {
                                        int wc_h = (half_t - (wp + 1) * 2) & ~1;
                                        if (wc_h < 4) break;
                                        apply_forward_word16_stride_difference(w_cur, st_buf, input_size, 2, wc_h);
                                        memcpy(w_cur, st_buf, (size_t)input_size);
                                        if (w_cur[input_size - 2] != X86_OPCODE_CALL_REL16 &&
                                            w_cur[input_size - 1] != X86_OPCODE_CALL_REL16) {
                                            int h8w = apply_forward_x86_opcode_filter(
                                                w_cur, e8_on_st, input_size, best_e8_tail_chosen, X86_OPCODE_CALL_REL16
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
                                                meta_q3.filter_counts[0] = best_e8_tail_chosen;
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
                                                evaluate_candidate_meta(
                                                    e8_on_st, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                                    stride_req_off, &meta_q3, &best_size, &best_meta, NULL
                                                );
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
                                    evaluate_candidate_meta(
                                        e8_on_st, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                        stride_req_off, &meta_eb, &best_size, &best_meta, NULL
                                    );
                                }
                            }
                        }
                        free(st_buf);
                        free(e8_on_st);
                        free(e9_on_st);
                    }

                    int n_hyb_s = (int)(sizeof(hyb_strides) / sizeof(hyb_strides[0]));
                    #pragma omp parallel for schedule(dynamic) if (!nested_parallel)
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
                                        evaluate_candidate_meta(
                                            e8_on_st, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                            stride_req_off, &meta_u1, &best_size, &best_meta, NULL
                                        );
                                    }
                                }

                                /* Decoder runs E8 -> E9 -> Stride, so encoder applies Stride -> E9 -> E8 */
                                if (format != ULTRA_FORMAT_ULTRA_128B &&
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
                                            evaluate_candidate_meta(
                                                e8_on_st, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                                                stride_req_off, &meta_hyb, &best_size, &best_meta, NULL
                                            );
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
        }
        free(best_e8_buf);

        /*
         * Final Exact DH Sentinel Refinement for ULTRA_FORMAT_ULTRA_128B:
         * Rank all 127 DH sentinels by exact byte-histogram cost AND by exact DP-parse literal
         * token distribution on the winning post-filtered buffer (and on working_input if
         * unfiltered remains competitive) and evaluate the top candidates in parallel.
         */
        if (format == ULTRA_FORMAT_ULTRA_128B) {
            int max_dh_cands = (input_size >= 8192) ? 10 : 12;
            if (best_meta.filter_passes == 0 || base_unfiltered_size <= best_size + 6) {
                int raw_est_dh = find_best_dh_sentinel(working_input, input_size, NULL);
                int uniq_dhs[16];
                int n_uniq = rank_top_dh_sentinels(working_input, input_size, raw_est_dh, uniq_dhs, max_dh_cands);
                ParsePlan ref_plan8 = {NULL, 0, 0};
                if (solve_exact_parse(working_input, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, &ref_plan8)) {
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
                #pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
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
                int n_ref_dhs = rank_top_dh_sentinels(f_cur, input_size, est_dh, uniq_ref_dhs, max_dh_cands);
                ParsePlan f_plan8 = {NULL, 0, 0};
                if (solve_exact_parse(f_cur, input_size, NULL, ZX0_DEFAULT_LSB_BITS, effective_limit, 0, &f_plan8)) {
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
                #pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
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
         * Reconstruct the winning filtered buffer once, run deep exact parse (g_deep_mode = 1),
         * and generate the final encoded bitstream.
         */
        g_deep_mode = 1;
        if (input_size >= 8192 && best_meta.filter_passes > 0 && base_unfiltered_size <= best_size + 32) {
            UltraMetadata umeta;
            memset(&umeta, 0, sizeof(umeta));
            umeta.format = format;
            umeta.offset_lsb_bits = ZX0_DEFAULT_LSB_BITS;
            umeta.dh_sentinel = (format == ULTRA_FORMAT_ULTRA_128B)
                              ? find_best_dh_sentinel(working_input, input_size, NULL) : 0;
            umeta.literal_bits = (format == ULTRA_FORMAT_ULTRA_128B)
                               ? detect_buffer_literal_bits(working_input, input_size, NULL) : 8;
            int req_u = (format == ULTRA_FORMAT_ULTRA_128B) ? -umeta.dh_sentinel : 0;
            evaluate_candidate_meta(
                working_input, input_size, format, ZX0_DEFAULT_LSB_BITS, effective_limit,
                req_u, &umeta, &best_size, &best_meta, NULL
            );
        }

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
        int final_req_off = (format == ULTRA_FORMAT_ULTRA_128B)
                          ? -(best_meta.dh_sentinel >= 1 && best_meta.dh_sentinel <= 127
                              ? best_meta.dh_sentinel
                              : find_best_dh_sentinel(win_buf, input_size, NULL))
                          : (best_meta.filter_passes > 0 && !is_multi_stage && best_meta.filter_stride > 0)
                            ? best_meta.filter_stride : 0;
        ParsePlan final_plan = {NULL, 0, 0};
        int ok = solve_exact_parse(
            win_buf, input_size, NULL, best_meta.offset_lsb_bits, effective_limit, final_req_off, &final_plan
        );
        assert(ok);
        best_meta.compressed_bits = final_plan.payload_bits;
        best_stream = encode_plan_to_stream(
            win_buf, &final_plan, format, best_meta.offset_lsb_bits, &best_meta, &best_size
        );
        free(final_plan.tokens);
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

    *compressed_size_out = best_size;
    if (metadata_out != NULL) {
        *metadata_out = best_meta;
    }
    return best_stream;
}
