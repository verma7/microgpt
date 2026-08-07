# Improving model quality: architecture & training study

Built on the fast V4-style engine ([microgpt_v5.cc](microgpt_v5.cc)), which
made a real experimental protocol affordable: **held-out validation loss**
(mean per-token NLL on a 10% split of names.txt the model never trains on),
3 seeds for the main grid, ~70 full training runs total. The original
autograd baseline could never have run this study — one 2000-step run took
90 s; V5 does a 2000-step run in ~0.2 s.

Every architectural addition (RoPE, SwiGLU, GeLU, weight tying, final
RMSNorm, learnable norm gains) has hand-derived gradients, validated against
numerical differentiation (`--gradcheck`, worst top-gradient error <1% in
float32, gains perturbed off identity so their backward path is exercised).

## Headline result

| Model | Params | Val loss | Val ppl |
|---|---:|---:|---:|
| Original setup (E=16, 2000 single-doc steps) | 13k | 2.313 ± .007 | 10.1 |
| Champion (E=32, finalnorm+gains, batch 8, 32k steps) | 52k | **2.044** | **7.7** |

~0.27 nats/char improvement, a 24% perplexity reduction, in ~5 minutes of
single-threaded training.

Champion recipe:

    ./microgpt_v5 --embd 32 --hidden 128 --steps 32000 --batch 8 \
                  --gains --finalnorm < names.txt

## What we learned (ablations)

**1. At the original size, "modern" components do nothing — or hurt.**
E=16, 10k steps, 3 seeds, val loss mean:

| Arch (cumulative unless noted) | Val loss |
|---|---:|
| base | **2.232** |
| + final rmsnorm | 2.263 |
| + weight tying | 2.272 |
| + RoPE | 2.277 |
| + GeLU | 2.273 |
| + SwiGLU (param-matched) | 2.259 |
| + SwiGLU + residscale init | 2.247 |

Seed noise is ±0.01–0.02; every modern stack is at or below base. A 13k-param
model on 200k training chars is capacity-limited — inductive-bias refinements
have nothing to work with. (This mirrors the common finding that
architecture tweaks are second-order compared to data/compute at small
scale.)

**2. Width helps, but exposes instability.** E=32 base reached 2.167 — and
diverged to inf on 1 of 2 seeds at lr 0.01 with single-doc steps. E=64
(201k params ≈ dataset size) was worse than E=32 at every budget tried.

**3. Gradient batching was the single biggest lever.** Averaging gradients
over 8 documents per Adam step (`--batch 8`):
- E=32 base: 2.167-or-diverged → **2.090 ± .001**, no divergence
- The one-doc-per-step regime of the original is extremely high-variance;
  most of what looked like an architecture problem was an optimization
  problem.

**4. Norms earn their keep only at longer training.** With batch 8:

| Steps | base | finalnorm+gains |
|---:|---:|---:|
| 8k | **2.090** | 2.118 |
| 16k | **2.062** | 2.078 |
| 32k | 2.051 | **2.044** |

The final RMSNorm (with learnable per-channel gains — the gainless version
is strictly worse, it pins the logit scale) starts behind but wins once
training runs long enough. Still improving at ~9 epochs; no overfitting yet
at 52k params.

**5. Things that didn't help:** lr 0.02 (worse than 0.01), E=48/E=64 at
these budgets, RoPE/tying/SwiGLU at any scale tested.

## Quality vs memorization

`novel%` = share of 500 sampled names (temp 0.5) not present in names.txt.
It falls as val loss improves (67% → ~35%): a better model of real names
assigns more mass to actual names. Val loss is the honest metric — it's
measured on held-out names.

## Files

- [microgpt_v5.cc](microgpt_v5.cc) — configurable model/training lab; see
  flags at top of file. `--gradcheck` validates gradients for any config.
- [ablation_results.txt](ablation_results.txt) — E=16 architecture grid, 3 seeds
- [scale_results.txt](scale_results.txt), [scale2_results.txt](scale2_results.txt),
  [scale3_results.txt](scale3_results.txt), [final_results.txt](final_results.txt)
  — scaling / batching / LR rounds
- [champion_out.txt](champion_out.txt) — full training log + samples of the
  best model
