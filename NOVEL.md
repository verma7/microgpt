# V7: novel architecture components on enwik8

[microgpt_v7.cc](microgpt_v7.cc) extends V6 with five experimental
components, each with hand-derived gradients validated by `--gradcheck`
(worst top-gradient error < 0.25%, individually and combined):

| Flag | Component | Idea |
|---|---|---|
| `--dwconv` | causal depthwise conv | learnable per-channel 3-tap conv on the normed inputs of attention and MLP (RWKV token-shift / canon-layer lineage) — local byte n-gram mixing |
| `--softmax1` | off-by-one softmax | attention denominator +1: heads can attend to *nothing* (the softmax backward formula is provably unchanged) |
| `--valres` | value residual | layer l's V blends with layer 0's V, learnable per-channel (ResFormer-style) |
| `--attngate` | attention output gate | elementwise σ(Wg·x) on attention output |
| `--qknorm` | QK-norm | RMS-normalize q,k per head + learnable per-head scale |

Also added for long runs: AdamW decoupled weight decay (`--wd`, matrices
only), global grad-norm clipping (`--clip`), and **sliding-window
evaluation** (`--evalstride N`: overlapping windows scoring only the last N
bytes, so every scored byte gets ≥ block−N bytes of context — the protocol
the literature uses, vs. V6's context-resetting windows).

## Single-component ablation

On top of the V6 "modern stack" (RoPE + tie + finalnorm + gains + SwiGLU +
residscale), E=128 L=6 T=256, fixed 8M-token budget, context-reset eval:

| Config | val bpc | Δ | params |
|---|---:|---:|---:|
| modern baseline | 2.315 | — | 1.220M |
| **+ dwconv** | **2.283** | **−0.032** | 1.225M |
| + softmax1 | 2.308 | −0.007 | 1.220M |
| + valres | 2.322 | +0.007 | 1.221M |
| + attngate | 2.333 | +0.018 | 1.319M |
| + qknorm | 2.359 | +0.044 | 1.220M |
| + dwconv + softmax1 | 2.282 | −0.033 | 1.225M |

The byte-level hypothesis held: cheap local mixing (dwconv, +4.6k params,
<1% slower) is worth more than everything else combined. Attention gating
doesn't pay for its 98k params; qk-norm slows optimization at this budget.

## Full-budget result (30k steps, 123M tokens, same recipe as V6's run)

| Model | val bpc | test bpc |
|---|---:|---:|
| V6 modern (context-reset eval) | 1.712 | 1.707 |
| V6 modern (sliding eval, stride 64) | 1.649 | 1.642 |
| **V7 modern + dwconv + softmax1 (sliding eval)** | **1.613** | **1.606** |

Same 28-minute training budget as V6; −0.036 bpc from the new components
and −0.063 from evaluating with proper context. Checkpoint:
`enwik8_model_v7.bin`.

Scaling runs toward lower bpc (larger models, longer context, weight decay,
longer training) are logged in [enwik8_r2.log](enwik8_r2.log) and successors.

# V8: speedrun-literature techniques

[microgpt_v8.cc](microgpt_v8.cc) implements the highest-ROI ideas from the
2024–25 nanoGPT-speedrun line of work
([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt),
[Muon](https://kellerjordan.github.io/posts/muon/)):

- `--muon` — **Muon optimizer** for hidden matrices: Nesterov momentum
  (warmed 0.85→0.95) orthogonalized by 5 Newton–Schulz iterations with the
  quintic coefficients (3.4445, −4.7750, 2.0315), step scaled by
  √max(1, rows/cols); embeddings/head/vector params stay on AdamW. Pure
  GEMM work, ~8% step overhead in this engine, threaded across matrices.
- `--mlp relu2` — ReLU² activation
- `--valemb` — per-layer zero-init token→value embedding tables
- `--softcap C` — logit soft-capping C·tanh(z/C)
- LR floor at 0.1× peak on the cosine schedule.

Ablation on top of the V7 winner stack (E=128, L=6, T=256, 8M tokens):

| Config | val bpc | Δ |
|---|---:|---:|
| V7 winner stack (AdamW) | 2.255 | — |
| **+ Muon (lr 0.02)** | **2.016** | **−0.239** |
| + Muon (lr 0.01) | 2.016 | −0.239 |
| + Muon (lr 0.04) | 2.094 | −0.161 |
| + ReLU² (AdamW, param-matched) | 2.229 | −0.026 |
| + Muon + ReLU² | 2.027 | −0.228 |
| + Muon + valemb | 2.027 | −0.228 |
| + softcap 15 | 2.255 | 0 |

**Muon is the single largest improvement in this entire project** —
−0.24 bpc at equal token budget, dwarfing every architecture change.
Under Muon, SwiGLU retains a small edge over ReLU²; value embeddings and
softcap are neutral at this scale/budget. Final recipe: V7 stack + Muon.
