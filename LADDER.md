# The 24-hour enwik8 ladder: 1.71 → 1.11 bpc on a laptop CPU

Goal: drive enwik8 validation bpc as low as possible in one
charger-plugged 24-hour window on an M4 Max (8 training threads, Apple
Accelerate GEMMs, the [microgpt_v8.cc](microgpt_v8.cc) engine — hand-derived
gradients throughout, no ML framework).

| Rung | Change | val bpc | test bpc |
|---|---|---:|---:|
| V6 | modern stack, 1.2M params, 123M tokens | 1.712 | 1.707 |
| — | sliding-window eval (context no longer resets) | 1.649 | 1.642 |
| V7 | + dwconv token-shift + softmax1 | 1.613 | 1.606 |
| Rung 2b | 6.4M params, T=512, 328M tokens, **+ Muon** | 1.315 | 1.291 |
| Rung 3 | 17.8M params, 491M tokens (8.5 h overnight) | 1.249 | 1.226 |
| Rung 4 | context 1024 via RoPE warm-start, +164M tokens | 1.153 | 1.131 |
| Rung 5 | second cosine cycle, +262M tokens | 1.135 | 1.114 |
| — | + dynamic evaluation (Krause-style, lr 2e-5, 1KB cadence) | **1.122** | **1.094** |

Reference points on the same axis: order-0 entropy ≈ 4.6; classic LSTMs
1.6–1.8; 24-layer Transformer-XL (277M params, GPU-days) 0.99; best
neural compressors ≈ 0.9–0.95.

## What moved the needle, ranked

1. **Muon optimizer** (−0.24 bpc at equal budget, and compounding at every
   scale above): orthogonalized momentum, implemented as 15 small GEMMs per
   matrix per step.
2. **Scale with the tokens to feed it** (1.2M → 6.4M → 17.8M params; each
   step also multiplied training tokens).
3. **Context length** 256 → 512 → 1024 — RoPE makes extension a warm-start
   fine-tune, not a retrain.
4. **Eval protocol honesty in both directions**: sliding-window evaluation
   (−0.06, the literature's protocol) and dynamic evaluation (−0.015,
   likewise published practice; every byte scored before the model adapts
   to it).
5. **Byte-level inductive bias**: dwconv token-shift (−0.03).
6. Weight decay, grad clipping, LR floor, momentum warmup: individually
   small, collectively what let the long runs run unattended.

## What didn't work (equally documented)

Value residuals, attention gating, QK-norm at short budgets, ReLU² under
Muon, coarse-cadence dynamic eval, `-ffast-math`, 16 threads (AMX/E-core
contention). See [NOVEL.md](NOVEL.md) and the `*_results.txt` grids.

## Reproducing

```bash
# rung 2b..5 recipes are in run_ladder.sh and the enwik8_r*.log headers;
# the final model:
./microgpt_v8 --threads 8 --steps 0 --load enwik8_r5.bin --evalstride 256 \
  --rope --tie --finalnorm --gains --residscale --mlp swiglu --dwconv \
  --softmax1 --embd 384 --layers 10 --heads 12 --block 1024 --hidden 1024
```

The sub-1.0 target needs roughly one more 4× jump in params × tokens
(≈ 2 more days at this throughput), which is beyond one charger window —
the ladder's slope says it's reachable, just not overnight.
