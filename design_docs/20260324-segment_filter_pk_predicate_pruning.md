# MEP: PK Predicate Segment Pruning in QueryNode

- **Created:** 2026-03-24
- **Author(s):** @xiaofan-luan
- **Status:** Implemented
- **Component:** Proxy, QueryNode
- **Related Issues:** #47804
- **Released:** TBD

## Summary

Prune sealed segments at the QueryNode segments layer before dispatching C++ search/query calls, by compiling primary-key (PK) predicates from the query plan into a small internal expression IR and evaluating it in batch against per-segment PK statistics (min/max range and bloom filter). A lightweight proxy-side hint avoids unnecessary plan compilation when no PK predicate exists.

## Motivation

In Milvus, each search or query request is dispatched to all sealed segments that fall within the requested partition/time range. For collections with many sealed segments, this causes significant unnecessary computation when the query contains PK-based predicates (e.g., `pk in [1, 2, 3]` or `pk > 100`).

**Current behavior:** Every sealed segment receives the request. Only inside the C++ expression evaluator does per-row filtering happen, by which point the vector-search kernel has already scanned data that is guaranteed not to contain matching rows.

**Key observation:** Each sealed segment records its min/max PK range (in the statistics blob) and a bloom filter seeded from all PKs in the segment. Together these can *definitively exclude* a segment before any C++ code is called:

- **Point queries** (`pk = X` or `pk IN [list]`): Bloom filters can definitively exclude segments that don't contain the target PKs.
- **Range queries** (`pk > X`, `pk < Y`): Min/max PK statistics can exclude segments whose range doesn't overlap the predicate.

The pruning happens in Go before the C++ call, so the segment pinning, context setup, and vector-search kernel are never invoked for pruned segments.

## Non-Goals

- Pruning on non-PK predicates.
- Filtering growing/streaming segments (their statistics are incomplete).
- Modifying C++ expression evaluation.

## Public Interfaces

### New Configuration Parameters

| Parameter | Key | Default | Description |
|-----------|-----|---------|-------------|
| `EnableSegmentFilter` | `queryNode.enableSegmentFilter` | `true` | Enable segments-layer PK predicate pruning (min/max + bloom filter) |

### Proto Changes

**`internal.proto`** — New field on `SearchRequest` and `RetrieveRequest`:

```protobuf
message SearchRequest {
    // ... existing fields ...
    int32 pk_filter = N;  // Proxy-set hint: 0=not checked, 1=has PK predicate, 2=no PK predicate
}

message RetrieveRequest {
    // ... existing fields ...
    int32 pk_filter = N;  // Same semantics as above
}
```

**Constants** (in `pkg/common/common.go`):

```go
const (
    PkFilterNotChecked  = int32(0) // Proxy did not analyse the plan (backward compat)
    PkFilterHasPkFilter = int32(1) // Proxy confirmed optimisable PK predicate exists
    PkFilterNoPkFilter  = int32(2) // Proxy confirmed no optimisable PK predicate
)
```

### New Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `milvus_querynode_segment_filter_total_segment_num` | Histogram | nodeID, collectionID, queryType | Total sealed segments considered |
| `milvus_querynode_segment_filter_skipped_segment_num` | Histogram | nodeID, collectionID, queryType | Segments pruned by PK filter |
| `milvus_querynode_segment_filter_hit_segment_num` | Histogram | nodeID, collectionID, queryType | Segments that passed the filter |

## Design Details

### Architecture Overview

```
┌─────────┐     ┌────────────────────┐     ┌──────────────────────────────────┐
│  Proxy  │────▶│  Shard Delegator   │────▶│  QueryNode Worker                │
│         │     │                    │     │                                  │
│ Analyze │     │ Forward PkFilter   │     │  validateOnHistorical()          │
│ plan,   │     │ hint to workers    │     │  ├─ buildSegmentFilterExprFrom   │
│ set     │     │ (no filtering      │     │  │   Plan(plan, pkFilter) → IR   │
│ PkFilter│     │  here)             │     │  ├─ checkSegmentFilter(IR, segs) │
│  hint   │     │                    │     │  │   ├─ min/max range check      │
└─────────┘     └────────────────────┘     │  │   └─ bloom filter check      │
                                           │  └─ unpin skipped segments       │
                                           └──────────────────────────────────┘
```

### Proxy-Side Hint

`HasOptimizablePkPredicate(plan)` in `internal/util/exprutil/expr_checker.go` walks the expression tree and returns `true` when the root expression is:

- `pk = x` (UnaryRangeExpr with EQ op)
- `pk IN [x₁, …]` (TermExpr on PK field)
- `lo < pk < hi` (BinaryRangeExpr on PK field)
- `AND` of any of the above

Before dispatching, the proxy sets `PkFilter` on the request:

- **`PkFilterHasPkFilter` (1):** Optimisable PK predicate found → worker compiles and applies the IR.
- **`PkFilterNoPkFilter` (2):** No PK predicate → worker skips IR compilation entirely.
- **`PkFilterNotChecked` (0):** Old proxy / rolling upgrade → worker attempts compilation anyway.

This avoids the cost of plan compilation on the worker for the majority of queries that have no PK predicate.

### Delegator: Hint Forwarding

The delegator does **not** filter segments. It only forwards the `PkFilter` hint when fan-out sub-requests are constructed via `shallowCopySearchRequest` / `shallowCopyRetrieveRequest`, so the worker shard that owns the sealed segments receives the hint.

### Segment Filter IR

`internal/querynodev2/segments/segment_filter.go` defines a small internal expression tree compiled once per request:

```
segmentFilterExpr
  ├── segmentFilterTermExpr           pk IN [v₁, v₂, …]
  ├── segmentFilterUnaryRangeExpr     pk op value
  ├── segmentFilterBinaryRangeExpr    lo (< / <=) pk (< / <=) hi
  ├── segmentFilterLogicalExpr        AND / OR of sub-expressions
  └── segmentFilterUnsupportedExpr    fallback — no pruning (keep all)
```

**Building:** `buildSegmentFilterExprFromPlan(plan, pkFilter)` returns `nil` immediately when `pkFilter == PkFilterNoPkFilter`. Otherwise it calls `extractSegmentFilterPredicates(plan)` to walk the plan and `buildSegmentFilterExpr(predicates)` to compile into the IR.

**Evaluating:** `checkSegmentFilter(expr, segments)` tests each sealed segment against the IR and returns a `segmentMatchSet`:

- `{all: true}` — keep all segments (used for unsupported expressions as conservative fallback).
- `{ids: Set[int64]}` — the explicit set of segment IDs that passed.

Per-node evaluation:

| Node type | Logic |
|---|---|
| `TermExpr` | For each value, check `[segMin, segMax]` inclusion; if in range probe bloom filter via `BatchPkExist`. Segment matches if any value passes both. |
| `UnaryRangeExpr` | Boundary arithmetic on segment min/max PK. |
| `BinaryRangeExpr` | Both lower and upper bound arithmetic on segment min/max PK. |
| `AND` | Evaluate left; **short-circuit** return empty set if left is already empty. Intersect with right otherwise. |
| `OR` | Evaluate left; **short-circuit** return `all` if left is already `all`. Union with right otherwise. |
| `Unsupported` | Return `{all: true}` — segment kept. |

### Validate Layer

`validateOnHistorical` in `internal/querynodev2/segments/validate.go`:

1. Calls `validate(...)` to pin all candidate sealed segments.
2. If `queryNode.enableSegmentFilter = true` **and** the compiled `expr != nil`:
   - Calls `checkSegmentFilter(expr, segments)`.
   - Unpins skipped segments immediately so their resources are released.
3. Records the three Prometheus histograms.

`validateOnStream` ignores the expression entirely — growing segments are always queried.

### Search and Retrieve Entrypoints

| Function | Applies filter? | Notes |
|---|---|---|
| `SearchHistorical` | Yes | `buildSegmentFilterExprFromPlan(plan, searchReq.PkFilter())` |
| `SearchStreaming` | No | Does not accept a plan node |
| `Retrieve` (historical scope) | Yes | Same builder, inside `DataScope_Historical` branch |
| `Retrieve` / `RetrieveStream` (streaming scope) | No | `nil` passed to `validateOnStream` |

The IR is compiled **once** in Go before any C++ call, so compilation cost is O(1) regardless of segment count.

### Data Flow

```
Search/Query Request
│
├── Proxy: HasOptimizablePkPredicate(plan) → set PkFilter hint
│
├── Delegator: shallowCopySearchRequest / shallowCopyRetrieveRequest
│   └── Forward PkFilter to worker sub-requests
│
└── QueryNode Worker: SearchHistorical / Retrieve
    ├── buildSegmentFilterExprFromPlan(plan, pkFilter)
    │   ├── Return nil immediately if pkFilter == PkFilterNoPkFilter
    │   └── Otherwise compile plan predicates → IR
    ├── validateOnHistorical(ctx, manager, ..., expr)
    │   ├── Pin all candidate sealed segments
    │   ├── checkSegmentFilter(expr, segments) → segmentMatchSet
    │   │   ├── For each segment:
    │   │   │   ├── TermExpr: range check + BatchPkExist bloom filter
    │   │   │   ├── UnaryRangeExpr: boundary arithmetic
    │   │   │   └── BinaryRangeExpr: lower + upper boundary arithmetic
    │   │   ├── AND: intersect (short-circuit on empty)
    │   │   └── OR: union (short-circuit on all)
    │   ├── Unpin skipped segments
    │   └── Record metrics (total / skipped / hit)
    └── searchSegments / retrieveSegments (C++ calls, filtered set only)
```

### Correctness Guarantees

- **No false negatives:** Bloom filter testing only produces false positives (a segment may be kept when it doesn't need to be), never false negatives. The min/max range check is exact. A segment is only skipped when it is *proven* not to contain a matching PK.
- **Min/max is conservative:** If statistics are unavailable, the segment is kept.
- **Growing segments untouched:** Growing segments are never pruned. Their statistics are updated asynchronously and may be incomplete.
- **Unsupported expressions:** Any expression that cannot be compiled into the IR becomes `segmentFilterUnsupportedExpr` → `{all: true}` → all segments kept.
- **Empty TermExpr:** `pk IN []` (impossible predicate) correctly evaluates to the empty match set, pruning all sealed segments.

## Compatibility, Deprecation, and Migration Plan

- **Rolling upgrade safe:** `PkFilter` defaults to `0` (NotChecked). Old proxies that don't set the hint cause the worker to attempt IR compilation anyway — no pruning regression, just slightly more work.
- **Feature flag:** `queryNode.enableSegmentFilter` defaults to `true` but can be disabled at runtime.
- **No API changes:** Purely an internal optimisation. External search/query APIs are unchanged.

## Test Plan

- **Unit tests** (`segment_filter_test.go`): IR evaluation for all node types — Term, UnaryRange, BinaryRange, AND, OR; edge cases (empty Term, unsupported expr, nil plan); AND/OR short-circuit paths.
- **Validate-layer tests** (`validate_test.go`): `validateOnHistorical` unpins skipped segments; does not filter when expr is nil; does not filter when `enableSegmentFilter=false`; metric observation.
- **Integration tests** (`search_test.go`, `retrieve_test.go`): End-to-end `SearchHistorical` and `Retrieve` with real segments and bloom filters loaded — verify correct number of segments scanned.
- **Proxy-side tests** (`task_search_pk_hint_test.go`): `HasOptimizablePkPredicate` for all plan types.
- **E2E:** Existing search/query integration tests ensure result correctness is preserved.

## Rejected Alternatives

### Delegator-Side Filtering

An earlier design applied bloom-filter and min/max checks inside the shard delegator before routing sub-requests to worker nodes. This was rejected because:

- The delegator does not pin or own sealed segments — filtering there would require an extra metadata lookup or cache layer.
- Applying filtering in the segments layer (where segments are already pinned) is simpler and covers both the delegator-routed and direct-query code paths from a single implementation point.
- The dominant cost being saved is the C++ vector-search kernel per segment. Filtering at the segments layer — just before the C++ call — achieves the same saving without adding delegator complexity.

### Pass PK Values from Proxy via Proto

An earlier design considered having the proxy extract PK values and pass them explicitly in the request proto. This was rejected because:
- It duplicates data already in the serialised plan, increasing message size.
- For large IN-lists the overhead of serialising PK values twice is wasteful.
- The hint field (PkFilter) achieves the avoidance of plan compilation at negligible cost.

### Prune Growing Segments

Growing segments have bloom filters and PK statistics, but they are mutable under concurrent inserts. Pruning them would require either locking (latency cost) or accepting race conditions (correctness risk). Since growing segments are typically small, the benefit doesn't justify the complexity.

## References

- PR: https://github.com/milvus-io/milvus/pull/47805
- Issue: https://github.com/milvus-io/milvus/issues/47804
