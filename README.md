# microgpt.cc
C++ implementation of Andrej Karpathy's microgpt

`microgpt.cc` is the readable reference implementation: a scalar autograd
engine training a 4-layer character-level transformer on `names.txt`.

```bash
./run.sh
```

## Optimized variants

Progressively faster rewrites of the same model — same math, same RNG
stream, near-identical losses. Full analysis and methodology in
[BENCHMARKS.md](BENCHMARKS.md) (Apple M4 Max, `-O3`, full 2000-step run):

| File | What changes | Wall time | Speedup |
|---|---|---:|---:|
| [microgpt.cc](microgpt.cc) | baseline | 89.7 s | 1.0× |
| [microgpt_v1.cc](microgpt_v1.cc) | engineering hygiene: iterative topo sort, no string-key lookups, reserved vectors | 34.5 s | 2.6× |
| [microgpt_v2.cc](microgpt_v2.cc) | fused n-ary autograd nodes (a dot product is one node) | 4.0 s | 22× |
| [microgpt_v3.cc](microgpt_v3.cc) | flat tape autograd: creation order = topological order, zero per-node allocations | 1.5 s | 59× |
| [microgpt_v4.cc](microgpt_v4.cc) | no graph at all: hand-derived backprop over float arrays | 0.13 s | ~690× |

## Model quality lab

[microgpt_v5.cc](microgpt_v5.cc) builds on v4: a train/validation split and
switchable architecture components (RoPE, weight tying, final RMSNorm,
learnable norm gains, GeLU/SwiGLU, residual init scaling, gradient
batching), each with hand-derived gradients validated by a built-in
numerical checker (`--gradcheck`). Findings from ~70 training runs are in
[QUALITY.md](QUALITY.md); the best recipe reaches val perplexity 7.7 vs
10.1 for the original setup:

```bash
c++ -O3 -std=c++17 microgpt_v5.cc -o microgpt_v5
./microgpt_v5 --embd 32 --hidden 128 --steps 32000 --batch 8 --gains --finalnorm < names.txt
```
