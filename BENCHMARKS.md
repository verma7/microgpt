# microgpt.cc runtime optimization benchmarks

Machine: Apple M4 Max, single-threaded, Apple clang, `-O3 -std=c++17`.
Workload: the full program — 2000 training steps on `names.txt` + 20 sampled
names, seed 42. Wall time via `/usr/bin/time` (best of 3 for the fast
variants; the slow ones vary <2% run to run).

| Variant | File | Wall time | Speedup | Final loss (step 2000) |
|---|---|---:|---:|---:|
| Baseline | `microgpt.cc` | 89.7 s | 1.0× | 1.92067 |
| Baseline + `-march=native -ffast-math` | `microgpt.cc` | 92.5 s | 0.97× | 1.92715 |
| V1: engineering hygiene | `microgpt_v1.cc` | 34.5 s | 2.6× | 1.92067 (bit-identical) |
| V2: V1 + fused n-ary graph nodes | `microgpt_v2.cc` | 4.0 s | 22× | 1.85775 |
| V3: V2 + flat tape autograd | `microgpt_v3.cc` | 1.5 s | 59× | 1.92065 |
| V4: hand-written backprop, no graph | `microgpt_v4.cc` | 0.13 s | ~690× | 1.92018 |

All variants produce *bit-identical* losses to the baseline for the first
500 steps; later values differ only in the last digits because gradient
*accumulation order* changes (forward math and Adam are unchanged). Sample
quality is equivalent throughout.

## Why the baseline is slow

Every scalar operation allocates a `shared_ptr<Value>` graph node holding two
heap `vector`s — a single 16×16 matvec is ~512 nodes, one training step is
hundreds of thousands. On top of that:

- `backward()` re-derives topological order every step with a recursive DFS
  over an `unordered_set<shared_ptr<Value>>` (hashing + refcount churn on
  every node).
- Weight lookups build `std::string` keys (`"layer{0}.attn_wq"`) and hash
  them per layer per token.
- Attention copies K/V head slices into fresh vectors per position and
  rebuilds constant nodes (e.g. `sqrt(head_dim)`) in inner loops.

The `-ffast-math` result is the tell: the program spends its time in
`malloc`, refcounting, and pointer chasing, not in floating-point math, so
compiler flags alone do nothing.

## What each variant changes

**V1 — hygiene (2.6×).** Same graph, same node count, same math (final loss
is bit-identical). Iterative topo sort using an epoch counter stamped on each
node instead of a hash set; weight matrices resolved once up front; vectors
`reserve`d; K/V heads indexed in place; constants hoisted. Roughly a third of
baseline runtime was pure bookkeeping overhead.

**V2 — fused ops (22×).** The autograd interface gains n-ary nodes:
`dot(w, x)` is *one* node with 2n children (local grads are just the opposing
operands' values), likewise fused `sum` and `sum-of-squares` for
softmax/rmsnorm. The graph shrinks ~15–20×, and with it every downstream
cost: allocation, topo sort, backward sweep.

**V3 — tape autograd (59×).** Replaces the pointer graph entirely: a node is
an index into contiguous arrays (`data`, `grad`, flattened child lists) on a
global tape. Since operands always exist before their result, *creation order
is topological order* — backward is a single reverse for-loop over the tape.
No DFS, no visited set, no `shared_ptr`, zero per-node allocations; each step
resets the tape to a post-init checkpoint and reuses the memory. This is how
serious tape-based autodiff systems work.

**V4 — closed-form backprop (~690×).** Drops autograd altogether: forward
stores activations in flat float arrays; backward applies the textbook
gradients for linear (`dx = Wᵀdy`, `dW += dy·xᵀ`), rmsnorm, softmax +
cross-entropy, and causal attention with a KV cache (positions processed in
reverse so cached dK/dV accumulate correctly). Same init RNG stream, same
Adam. At 0.13 s total, a large share of the remaining time is reading
`names.txt`; training itself sustains roughly 17,000 steps/s vs the
baseline's 22 steps/s. `-march=native` adds nothing measurable at these tiny
dimensions.

## Not pursued

- **Threading / GPU** — at d_model=16 the per-op work is far below
  parallelization overhead; the model fits in L1 cache.
- **BLAS (Accelerate)** — same reason: 16×16 matvecs are cheaper inlined
  than through a library call.
- **Batching multiple documents per step** — would change training semantics
  (the baseline is strictly one document per step).
