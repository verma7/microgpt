# enwik8: scaling microgpt to a real benchmark

[microgpt_v6.cc](microgpt_v6.cc) trains a byte-level transformer LM on
**enwik8** (first 100 MB of English Wikipedia, the Hutter Prize corpus;
standard splits: train = first 90 MB, val = next 5 MB, test = last 5 MB) and
reports **bits per character (bpc)** — the metric the compression/LM
literature uses, so results are directly comparable to published numbers.

## Headline result

**1.71 bpc on the held-out test split** with a 1.2M-parameter model trained
for ~28 minutes on a MacBook (M4 Max, CPU only, 8 threads):

    RESULT params=1220352 val_bpc=1.7119 test_bpc=1.7065

For reference (different sizes/budgets, but the same axis): order-0 entropy
of enwik8 is ~4.6 bpc, random bytes 8.0, classic LSTM baselines ~1.6–1.8,
Transformer-XL ~0.99, best compressors ~0.9. A 1.2M-param model inside the
LSTM band after half an hour of laptop CPU is the "this is a real
implementation" checkpoint. The loss was still falling at the end — more
steps and a bigger model are the obvious next wins.

Training recipe:

    c++ -O3 -std=c++17 microgpt_v6.cc -o microgpt_v6 \
        -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
    ./microgpt_v6 --threads 8 --steps 30000 --rope --tie --finalnorm \
        --gains --residscale --mlp swiglu --hidden 344 --save enwik8_model.bin

Full log: [enwik8_final.log](enwik8_final.log). The saved model can be
re-evaluated or sampled with `--load enwik8_model.bin --steps 0`.

## Performance optimizations (v5 → v6)

Two structural changes, benchmarked at the final config (E=128, L=6,
T=256, batch 16):

| Engine | tok/s | speedup |
|---|---:|---:|
| per-position scalar loops (v4/v5 style), 1 thread | 433 | 1× |
| sequence-level GEMMs (Apple Accelerate), 1 thread | 32,010 | 74× |
| GEMMs + 4 threads | 94,060 | 217× |
| **GEMMs + 8 threads** | **141,496** | **327×** |
| GEMMs + 16 threads | 119,948 | 277× |

1. **Every layer is a GEMM.** v4/v5 process one token at a time, so each
   linear layer is a skinny matrix–vector product — too small for BLAS to
   help (why we skipped it at E=16). v6 processes a whole 256-byte window at
   once: activations are `[T×E]` matrices, weights `[E×E]`, so forward
   *and* backward are matrix–matrix products (`dX = dY·W`, `dW = dYᵀ·X`)
   dispatched to `cblas_sgemm`. Attention likewise (`Q·Kᵀ`, `A·V` per head,
   with causal-masked softmax rows). On M-series chips Accelerate routes
   GEMM through the AMX matrix units — hardware plain compiled loops can't
   reach. `--noblas` swaps in naive loops for benchmarking.
2. **Data-parallel batch across threads.** Each Adam step processes
   `--batch 16` windows; `--threads` workers each run forward+backward into
   a thread-local gradient buffer, reduced in fixed order (training is
   bit-deterministic for any thread count) and fused with the Adam update,
   which is itself parallelized over parameter chunks. 8 threads is the M4
   Max sweet spot — beyond that, E-cores and AMX contention give it back.

At 141k tok/s, one enwik8 epoch (90M bytes) takes ~11 minutes on a laptop.

## Model optimizations

Everything from the v5 quality study carries over, with hand-derived
gradients validated by `--gradcheck` (worst top-gradient error < 0.3%):
RoPE, weight tying, final RMSNorm, learnable per-channel norm gains,
SwiGLU, GeLU, residual-projection init scaling — plus v6 additions:
GPT-2 init (std 0.02), Adam β=(0.9, 0.95), cosine LR with linear warmup.

Config shootout at a fixed 8M-token budget (E=128, L=6, T=256):

| Config | val bpc |
|---|---:|
| plain: learned wpe, ReLU, no final norm | 2.761 |
| + final RMSNorm + gains + residscale | 2.739 |
| **+ RoPE + weight tying + SwiGLU (full modern)** | **2.315** |

This is the punchline of the whole project arc: on 230 KB of names
([QUALITY.md](QUALITY.md)) these same components were within seed noise or
harmful; on 90 MB of Wikipedia the modern stack wins by 0.45 bpc at equal
budget. Architecture improvements are scale-dependent — the tiny setting
actively misleads about them.

## Eval protocol note

bpc is measured over consecutive non-overlapping 256-byte windows with no
context carried across windows, so early bytes of each window are predicted
with little context. This slightly pessimizes the number vs. sliding-window
evaluation — the reported bpc is honest, not flattered.

## Sample (temp 0.8, primed with 64 bytes of validation text)

> hese contrasts are not designed to be independently head of particularly
> focusing on the short-version of the anime and transvestion for a
> receiving standard, and failed to other congregational amputees of the
> Republican Chinese Amplifiers meant that remained a state of that added
> it is always one of the defensive nature of the articles of
> [[equilibrium|explicit]]s. The tradestinian credible effects of the
> project that their snap by [[Savathato the Savantas]] and [[Nobel
> crediteration in the United States]]

Byte-level grammar, wiki markup (`[[...|...]]` links), and plausible
English morphology — from 1.2M parameters.
