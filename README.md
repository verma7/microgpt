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

**Current record: 1.12 val / 1.09 test bpc** — a 24-hour scaling ladder
(Muon optimizer, 17.8M params, 1024 context, dynamic evaluation) documented
in [LADDER.md](LADDER.md).

## enwik8 (Hutter Prize)

[microgpt_v6.cc](microgpt_v6.cc) scales the engine to a real benchmark:
byte-level LM on the first 100 MB of Wikipedia. Sequence-level GEMMs
(Apple Accelerate) + threaded data parallelism take throughput from 433 to
141k tok/s (327×), and the "modern" components that were useless on
names.txt win by 0.45 bpc here. **1.71 test bpc with 1.2M params in ~28
minutes of laptop CPU.** Full story in [ENWIK8.md](ENWIK8.md).

```bash
curl -sL -o enwik8.zip http://mattmahoney.net/dc/enwik8.zip && unzip enwik8.zip
c++ -O3 -std=c++17 microgpt_v6.cc -o microgpt_v6 -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
./microgpt_v6 --threads 8 --steps 30000 --rope --tie --finalnorm --gains --residscale --mlp swiglu --hidden 344
```
