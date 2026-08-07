// V7 = V6 + novel/experimental architecture components, each with
// hand-derived gradients (validated by --gradcheck):
//   --dwconv    learnable per-channel causal 3-tap depthwise conv on the
//               normed inputs of attention and MLP (RWKV token-shift /
//               canon-layer lineage; local byte n-gram mixing)
//   --valres    value residual: layer l's V is a learnable per-channel
//               blend with layer 0's V (ResFormer-style)
//   --attngate  elementwise sigmoid gate sigma(Wg x) on the attention
//               output before the out projection
//   --qknorm    RMS-normalize q and k per head, learnable per-head scale
//   --softmax1  off-by-one attention softmax (denominator +1): heads can
//               attend to nothing. Backward formula is unchanged.
//
// V6: byte-level transformer LM for enwik8 (Hutter Prize) with two new
// performance axes on top of v4/v5's hand-written backprop:
//   1. sequence-level math: a training example is a T-byte window and every
//      linear layer is a GEMM over the whole window (Apple Accelerate BLAS;
//      --noblas falls back to naive loops for benchmarking)
//   2. data parallelism: each Adam step processes --batch windows across
//      --threads worker threads with thread-local gradient buffers,
//      reduced deterministically
// Model options carried over from v5 (all with hand-derived gradients):
//   --rope --finalnorm --gains --residscale --mlp relu|gelu|swiglu --tie
// plus cosine LR with warmup (--warmup), GPT-2 style init std 0.02.
// Metric: bits per character (bpc) on the standard enwik8 splits
// (train = first 90MB, val = next 5MB, test = last 5MB).
//
// Build: c++ -O3 -std=c++17 microgpt_v6.cc -o microgpt_v6 \
//            -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

using namespace std;

static bool g_use_blas = true;

// C[m,n] (+)= alpha * op(A) * op(B); row-major with explicit leading dims.
static void gemm(bool transA, bool transB, int m, int n, int k, float alpha,
                 const float *A, int lda, const float *B, int ldb, float beta,
                 float *C, int ldc) {
#ifdef USE_ACCELERATE
  if (g_use_blas) {
    cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, m, n, k, alpha, A, lda, B,
                ldb, beta, C, ldc);
    return;
  }
#endif
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      float s = 0;
      for (int p = 0; p < k; p++) {
        float a = transA ? A[p * lda + i] : A[i * lda + p];
        float b = transB ? B[j * ldb + p] : B[p * ldb + j];
        s += a * b;
      }
      C[i * ldc + j] = alpha * s + beta * C[i * ldc + j];
    }
}

static inline float gelu_f(float z) {
  const float c = 0.7978845608f, a = 0.044715f;
  float t = tanhf(c * (z + a * z * z * z));
  return 0.5f * z * (1 + t);
}
static inline float gelu_df(float z) {
  const float c = 0.7978845608f, a = 0.044715f;
  float t = tanhf(c * (z + a * z * z * z));
  return 0.5f * (1 + t) + 0.5f * z * (1 - t * t) * c * (1 + 3 * a * z * z);
}
static inline float silu_f(float z) {
  float s = 1.0f / (1.0f + expf(-z));
  return z * s;
}
static inline float silu_df(float z) {
  float s = 1.0f / (1.0f + expf(-z));
  return s * (1 + z * (1 - s));
}

struct Config {
  int layers = 6, embd = 128, heads = 8, block = 256, hidden = 512;
  int steps = 24000, seed = 42, eval_every = 1000, batch = 16, warmup = 500;
  int threads = 0; // 0 = auto
  float lr = 1e-3f;
  float wd = 0.0f;   // AdamW decoupled weight decay (matrices only)
  float clip = 0.0f; // global grad-norm clip, 0 = off
  int evalstride = 0; // final eval: overlapping windows scoring last N bytes
  bool rope = false, tie = false, finalnorm = false, residscale = false;
  bool gains = false, gradcheck = false, noblas = false, nofinal = false;
  bool dwconv = false, valres = false, attngate = false, qknorm = false,
       softmax1 = false;
  string mlp = "relu";
  string data = "enwik8";
  string save = "", load = "";
  long train_bytes = 90000000, val_bytes = 5000000; // test = remainder
  long eval_bytes = 500000; // periodic eval subset; final eval uses all
};

struct Model {
  Config cfg;
  static const int V = 256; // byte vocab
  int D;
  int o_wte, o_wpe, o_lm, o_gemb, o_gfin;
  vector<int> o_wq, o_wk, o_wv, o_wo, o_fc1, o_fc2, o_fc3, o_gattn, o_gmlp;
  vector<int> o_dwa, o_dwm, o_wg, o_vres, o_qkg; // v7 components
  int wd_end; // params below this index get weight decay (the matrices)
  int NP;
  vector<float> P, G, am, av;
  bool swiglu, gelu;
  float inv_sqrt_d;
  vector<float> rope_cos, rope_sin; // [block, D/2]

  void init(mt19937 &rng) {
    D = cfg.embd / cfg.heads;
    inv_sqrt_d = 1.0f / sqrtf((float)D);
    swiglu = cfg.mlp == "swiglu";
    gelu = cfg.mlp == "gelu";
    int L = cfg.layers, E = cfg.embd, M = cfg.hidden, T = cfg.block;
    int off = 0;
    o_wte = off; off += V * E;
    if (!cfg.rope) { o_wpe = off; off += T * E; } else o_wpe = -1;
    if (!cfg.tie) { o_lm = off; off += V * E; } else o_lm = o_wte;
    o_wq.resize(L); o_wk.resize(L); o_wv.resize(L); o_wo.resize(L);
    o_fc1.resize(L); o_fc2.resize(L); o_fc3.resize(L);
    for (int l = 0; l < L; l++) {
      o_wq[l] = off; off += E * E;
      o_wk[l] = off; off += E * E;
      o_wv[l] = off; off += E * E;
      o_wo[l] = off; off += E * E;
      o_fc1[l] = off; off += M * E;
      o_fc2[l] = off; off += E * M;
      if (swiglu) { o_fc3[l] = off; off += M * E; }
    }
    o_dwa.assign(L, -1); o_dwm.assign(L, -1); o_wg.assign(L, -1);
    o_vres.assign(L, -1); o_qkg.assign(L, -1);
    wd_end = off; // everything above is a matrix; below: taps/gates/gains
    int H = cfg.heads;
    for (int l = 0; l < L; l++) {
      if (cfg.dwconv) { o_dwa[l] = off; off += 3 * E; o_dwm[l] = off; off += 3 * E; }
      if (cfg.attngate) { o_wg[l] = off; off += E * E; }
      if (cfg.valres && l > 0) { o_vres[l] = off; off += E; }
      if (cfg.qknorm) { o_qkg[l] = off; off += H; }
    }
    o_gattn.assign(L, -1); o_gmlp.assign(L, -1);
    o_gemb = o_gfin = -1;
    if (cfg.gains) {
      o_gemb = off; off += E;
      for (int l = 0; l < L; l++) {
        o_gattn[l] = off; off += E;
        o_gmlp[l] = off; off += E;
      }
      if (cfg.finalnorm) { o_gfin = off; off += E; }
    }
    NP = off;
    P.assign(NP, 0); G.assign(NP, 0); am.assign(NP, 0); av.assign(NP, 0);
    normal_distribution<float> d(0, 0.02f); // GPT-2 init
    for (int i = 0; i < NP; i++)
      P[i] = d(rng);
    if (cfg.residscale) {
      float sc = 1.0f / sqrtf(2.0f * L);
      for (int l = 0; l < L; l++) {
        for (int i = 0; i < E * E; i++)
          P[o_wo[l] + i] *= sc;
        for (int i = 0; i < E * M; i++)
          P[o_fc2[l] + i] *= sc;
      }
    }
    if (cfg.gains)
      for (int i = (cfg.gains ? o_gemb : NP); i < NP; i++)
        P[i] = 1.0f;
    for (int l = 0; l < L; l++) {
      if (cfg.dwconv) { // taps init mostly-identity: [0.8, 0.15, 0.05]
        const float tap[3] = {0.8f, 0.15f, 0.05f};
        for (int j = 0; j < 3; j++)
          for (int i = 0; i < E; i++) {
            P[o_dwa[l] + j * E + i] = tap[j];
            P[o_dwm[l] + j * E + i] = tap[j];
          }
      }
      if (cfg.valres && l > 0) // sigma(1.1) ~ 0.75 toward own value
        for (int i = 0; i < E; i++)
          P[o_vres[l] + i] = 1.1f;
      if (cfg.qknorm)
        for (int h = 0; h < cfg.heads; h++)
          P[o_qkg[l] + h] = 1.0f;
    }
    if (cfg.rope) {
      rope_cos.resize(T * (D / 2));
      rope_sin.resize(T * (D / 2));
      for (int t = 0; t < T; t++)
        for (int i = 0; i < D / 2; i++) {
          float freq = powf(10000.0f, -2.0f * i / D);
          rope_cos[t * (D / 2) + i] = cosf(t * freq);
          rope_sin[t * (D / 2) + i] = sinf(t * freq);
        }
    }
  }
};

// Per-thread workspace: all activations for one T-window plus local grads.
struct Workspace {
  const Model *m;
  int T, E, M, L, H, D;
  // activations, all [T,rowdim] row-major
  vector<float> X0pre, X0;
  vector<float> Ain, Anorm, Q, K, Vv, Attn, Min, Mnorm, H1pre, H1, Gpre, Upre;
  // per layer copies (stacked [L][T,E] etc.)
  vector<float> S; // attention probs [L][H][T,T]
  vector<float> Xf, Xfn, Probs, sr; // sr: rms scales, [ (2L+2) * T ]
  vector<float> Glocal;
  // v7 activations
  vector<float> Xsa, Xsm;        // dwconv outputs, [L][T,E]
  vector<float> Gt, AttnG;       // attngate: gate + gated attn, [L][T,E]
  vector<float> Vraw;            // valres: pre-mix V, [L][T,E]
  vector<float> Qraw, Kraw;      // qknorm: pre-norm q/k, [L][T,E]
  vector<float> sq, sk;          // qknorm rms scales, [L][T,H]
  // backward scratch
  vector<float> dX, dXn, dQ, dK, dV, dAttn, dH1, dS, dRow, dMin;
  vector<float> dXs, dV0acc, dGp;
  double loss_sum = 0;
  long tok_count = 0;

  void alloc(const Model *model) {
    m = model;
    T = m->cfg.block; E = m->cfg.embd; M = m->cfg.hidden; L = m->cfg.layers;
    H = m->cfg.heads; D = m->D;
    auto z = [](vector<float> &v, size_t n) { v.assign(n, 0.0f); };
    z(X0pre, (size_t)T * E); z(X0, (size_t)T * E);
    z(Ain, (size_t)L * T * E); z(Anorm, (size_t)L * T * E);
    z(Q, (size_t)L * T * E); z(K, (size_t)L * T * E); z(Vv, (size_t)L * T * E);
    z(Attn, (size_t)L * T * E); z(Min, (size_t)L * T * E);
    z(Mnorm, (size_t)L * T * E);
    z(H1pre, (size_t)L * T * M); z(H1, (size_t)L * T * M);
    if (m->swiglu) { z(Gpre, (size_t)L * T * M); z(Upre, (size_t)L * T * M); }
    z(S, (size_t)L * H * T * T);
    z(Xf, (size_t)T * E); z(Xfn, (size_t)T * E);
    z(Probs, (size_t)T * Model::V);
    z(sr, (size_t)(2 * L + 2) * T);
    z(Glocal, m->NP);
    z(dX, (size_t)T * E); z(dXn, (size_t)T * E); z(dQ, (size_t)T * E);
    z(dK, (size_t)T * E); z(dV, (size_t)T * E); z(dAttn, (size_t)T * E);
    z(dH1, (size_t)T * M); z(dS, (size_t)T * T); z(dRow, max(E, M));
    z(dMin, (size_t)T * E);
    if (m->cfg.dwconv) { z(Xsa, (size_t)L * T * E); z(Xsm, (size_t)L * T * E); }
    if (m->cfg.attngate) {
      z(Gt, (size_t)L * T * E); z(AttnG, (size_t)L * T * E);
      z(dGp, (size_t)T * E);
    }
    if (m->cfg.valres) { z(Vraw, (size_t)L * T * E); z(dV0acc, (size_t)T * E); }
    if (m->cfg.qknorm) {
      z(Qraw, (size_t)L * T * E); z(Kraw, (size_t)L * T * E);
      z(sq, (size_t)L * T * H); z(sk, (size_t)L * T * H);
    }
    z(dXs, (size_t)T * E);
  }

  // rmsnorm rows of X[T,E] -> Y, scales into sr[srow*T + t], optional gains
  void rms_rows(const float *X, float *Y, int srow, int goff) {
    const float *Pp = m->P.data();
    for (int t = 0; t < T; t++) {
      const float *x = X + (size_t)t * E;
      float *y = Y + (size_t)t * E;
      float ms = 0;
      for (int i = 0; i < E; i++)
        ms += x[i] * x[i];
      ms /= E;
      float s = 1.0f / sqrtf(ms + 1e-5f);
      sr[(size_t)srow * T + t] = s;
      if (goff >= 0)
        for (int i = 0; i < E; i++)
          y[i] = x[i] * s * Pp[goff + i];
      else
        for (int i = 0; i < E; i++)
          y[i] = x[i] * s;
    }
  }
  // backward of rms_rows: dY given, accumulate into dXacc; gain grads local
  void rms_rows_bwd(const float *X, const float *dY, float *dXacc, int srow,
                    int goff) {
    const float *Pp = m->P.data();
    for (int t = 0; t < T; t++) {
      const float *x = X + (size_t)t * E;
      const float *dy = dY + (size_t)t * E;
      float *dx = dXacc + (size_t)t * E;
      float s = sr[(size_t)srow * T + t];
      float dyx = 0;
      if (goff >= 0) {
        for (int i = 0; i < E; i++) {
          Glocal[goff + i] += dy[i] * x[i] * s;
          dRow[i] = dy[i] * Pp[goff + i];
        }
        for (int i = 0; i < E; i++)
          dyx += dRow[i] * x[i];
        float c = s * s * s * dyx / E;
        for (int i = 0; i < E; i++)
          dx[i] += s * dRow[i] - c * x[i];
      } else {
        for (int i = 0; i < E; i++)
          dyx += dy[i] * x[i];
        float c = s * s * s * dyx / E;
        for (int i = 0; i < E; i++)
          dx[i] += s * dy[i] - c * x[i];
      }
    }
  }

  // causal per-channel 3-tap conv: y[t] = sum_j w[j] (.) x[t-j]
  void dwconv_fwd(const float *X, float *Y, int woff) {
    const float *w = m->P.data() + woff;
    for (int t = 0; t < T; t++) {
      const float *x0 = X + (size_t)t * E;
      float *y = Y + (size_t)t * E;
      for (int i = 0; i < E; i++)
        y[i] = w[i] * x0[i];
      for (int j = 1; j < 3 && j <= t; j++) {
        const float *xj = X + (size_t)(t - j) * E;
        for (int i = 0; i < E; i++)
          y[i] += w[j * E + i] * xj[i];
      }
    }
  }
  // dX += conv^T(dY); tap grads into Glocal
  void dwconv_bwd(const float *X, const float *dY, float *dXacc, int woff) {
    const float *w = m->P.data() + woff;
    float *gw = Glocal.data() + woff;
    for (int t = 0; t < T; t++) {
      const float *dy = dY + (size_t)t * E;
      for (int j = 0; j < 3 && j <= t; j++) {
        const float *xj = X + (size_t)(t - j) * E;
        float *dxj = dXacc + (size_t)(t - j) * E;
        for (int i = 0; i < E; i++) {
          dxj[i] += w[j * E + i] * dy[i];
          gw[j * E + i] += dy[i] * xj[i];
        }
      }
    }
  }
  // rms-normalize each head slice of Q_[T,E] in place; save scales
  void qknorm_fwd(float *Q_, float *scales, const float *Raw) {
    for (int t = 0; t < T; t++)
      for (int h = 0; h < H; h++) {
        float *v = Q_ + (size_t)t * E + h * D;
        float ms = 0;
        for (int i = 0; i < D; i++)
          ms += v[i] * v[i];
        float s = 1.0f / sqrtf(ms / D + 1e-6f);
        scales[(size_t)t * H + h] = s;
        for (int i = 0; i < D; i++)
          v[i] *= s;
      }
    (void)Raw;
  }
  // given dNormed (grad wrt normalized head vecs), produce grad wrt raw
  void qknorm_bwd(const float *Raw, const float *scales, float *dQio) {
    for (int t = 0; t < T; t++)
      for (int h = 0; h < H; h++) {
        const float *x = Raw + (size_t)t * E + h * D;
        float *dy = dQio + (size_t)t * E + h * D; // in: dNormed, out: dRaw
        float s = scales[(size_t)t * H + h];
        float dyx = 0;
        for (int i = 0; i < D; i++)
          dyx += dy[i] * x[i];
        float c = s * s * s * dyx / D;
        for (int i = 0; i < D; i++)
          dy[i] = s * dy[i] - c * x[i];
      }
  }

  void rope_rows(float *Q_, int sign) { // sign=+1 fwd, -1 bwd (transpose)
    int half = D / 2;
    for (int t = 0; t < T; t++)
      for (int h = 0; h < H; h++) {
        float *v = Q_ + (size_t)t * E + h * D;
        for (int i = 0; i < half; i++) {
          float c = m->rope_cos[t * half + i];
          float s = m->rope_sin[t * half + i] * sign;
          float a = v[2 * i], b = v[2 * i + 1];
          v[2 * i] = a * c - b * s;
          v[2 * i + 1] = a * s + b * c;
        }
      }
  }

  // Forward one window; tokens has T+1 bytes (inputs + shifted targets).
  // If accumulate_grads, also run backward into Glocal.
  float run_window(const uint8_t *tokens, bool accumulate_grads) {
    const Model &mm = *m;
    const float *Pp = mm.P.data();
    int E_ = E, T_ = T, M_ = M, H_ = H, D_ = D, Vc = Model::V;
    // embeddings
    for (int t = 0; t < T_; t++) {
      const float *we = Pp + mm.o_wte + (size_t)tokens[t] * E_;
      float *x = &X0pre[(size_t)t * E_];
      if (mm.cfg.rope)
        memcpy(x, we, sizeof(float) * E_);
      else {
        const float *wp = Pp + mm.o_wpe + (size_t)t * E_;
        for (int i = 0; i < E_; i++)
          x[i] = we[i] + wp[i];
      }
    }
    rms_rows(X0pre.data(), X0.data(), 0, mm.o_gemb);
    float *X = X0.data();
    for (int l = 0; l < L; l++) {
      float *ain = &Ain[(size_t)l * T_ * E_];
      if (X != ain)
        memcpy(ain, X, sizeof(float) * T_ * E_);
      float *xn = &Anorm[(size_t)l * T_ * E_];
      rms_rows(ain, xn, 1 + 2 * l, mm.o_gattn[l]);
      float *xs = xn;
      if (mm.cfg.dwconv) {
        xs = &Xsa[(size_t)l * T_ * E_];
        dwconv_fwd(xn, xs, mm.o_dwa[l]);
      }
      float *q = &Q[(size_t)l * T_ * E_];
      float *k = &K[(size_t)l * T_ * E_];
      float *v = &Vv[(size_t)l * T_ * E_];
      float *vproj = mm.cfg.valres ? &Vraw[(size_t)l * T_ * E_] : v;
      gemm(false, true, T_, E_, E_, 1, xs, E_, Pp + mm.o_wq[l], E_, 0, q, E_);
      gemm(false, true, T_, E_, E_, 1, xs, E_, Pp + mm.o_wk[l], E_, 0, k, E_);
      gemm(false, true, T_, E_, E_, 1, xs, E_, Pp + mm.o_wv[l], E_, 0, vproj,
           E_);
      if (mm.cfg.valres) {
        if (l == 0)
          memcpy(v, vproj, sizeof(float) * T_ * E_);
        else {
          const float *lam = Pp + mm.o_vres[l];
          const float *v0 = &Vv[0]; // layer 0's (mixed==raw) V
          for (int t = 0; t < T_; t++)
            for (int i = 0; i < E_; i++) {
              float a = 1.0f / (1.0f + expf(-lam[i]));
              v[(size_t)t * E_ + i] = a * vproj[(size_t)t * E_ + i] +
                                      (1 - a) * v0[(size_t)t * E_ + i];
            }
        }
      }
      if (mm.cfg.qknorm) {
        memcpy(&Qraw[(size_t)l * T_ * E_], q, sizeof(float) * T_ * E_);
        memcpy(&Kraw[(size_t)l * T_ * E_], k, sizeof(float) * T_ * E_);
        qknorm_fwd(q, &sq[(size_t)l * T_ * H_], nullptr);
        qknorm_fwd(k, &sk[(size_t)l * T_ * H_], nullptr);
      }
      if (mm.cfg.rope) { rope_rows(q, +1); rope_rows(k, +1); }
      if (mm.cfg.qknorm) // fold per-head scale into q: logits = g_h q^.k^/sqrt(D)
        for (int t = 0; t < T_; t++)
          for (int h = 0; h < H_; h++) {
            float g = Pp[mm.o_qkg[l] + h];
            float *qv = q + (size_t)t * E_ + h * D_;
            for (int i = 0; i < D_; i++)
              qv[i] *= g;
          }
      float *attn = &Attn[(size_t)l * T_ * E_];
      for (int h = 0; h < H_; h++) {
        float *Sh = &S[(((size_t)l * H_) + h) * T_ * T_];
        gemm(false, true, T_, T_, D_, mm.inv_sqrt_d, q + h * D_, E_,
             k + h * D_, E_, 0, Sh, T_);
        // causal softmax rows (cols > t masked out)
        for (int t = 0; t < T_; t++) {
          float *srow_ = Sh + (size_t)t * T_;
          float mx = srow_[0];
          for (int u = 1; u <= t; u++)
            mx = max(mx, srow_[u]);
          if (mm.cfg.softmax1)
            mx = max(mx, 0.0f); // implicit sink logit 0
          float tot = mm.cfg.softmax1 ? expf(-mx) : 0.0f;
          for (int u = 0; u <= t; u++) {
            srow_[u] = expf(srow_[u] - mx);
            tot += srow_[u];
          }
          float inv = 1.0f / tot;
          for (int u = 0; u <= t; u++)
            srow_[u] *= inv;
          for (int u = t + 1; u < T_; u++)
            srow_[u] = 0.0f;
        }
        gemm(false, false, T_, D_, T_, 1, Sh, T_, v + h * D_, E_, 0,
             attn + h * D_, E_);
      }
      float *attn_used = attn;
      if (mm.cfg.attngate) { // gate = sigma(xs Wg^T), gated = gate (.) attn
        float *g = &Gt[(size_t)l * T_ * E_];
        float *ag = &AttnG[(size_t)l * T_ * E_];
        gemm(false, true, T_, E_, E_, 1, xs, E_, Pp + mm.o_wg[l], E_, 0, g,
             E_);
        for (size_t i = 0; i < (size_t)T_ * E_; i++) {
          g[i] = 1.0f / (1.0f + expf(-g[i]));
          ag[i] = g[i] * attn[i];
        }
        attn_used = ag;
      }
      // out projection + residual -> X
      gemm(false, true, T_, E_, E_, 1, attn_used, E_, Pp + mm.o_wo[l], E_, 0,
           dX.data(), E_);
      float *min = &Min[(size_t)l * T_ * E_];
      for (size_t i = 0; i < (size_t)T_ * E_; i++)
        min[i] = dX[i] + ain[i];
      // MLP
      float *mn = &Mnorm[(size_t)l * T_ * E_];
      rms_rows(min, mn, 2 + 2 * l, mm.o_gmlp[l]);
      float *ms = mn;
      if (mm.cfg.dwconv) {
        ms = &Xsm[(size_t)l * T_ * E_];
        dwconv_fwd(mn, ms, mm.o_dwm[l]);
      }
      float *h1p = &H1pre[(size_t)l * T_ * M_];
      float *h1 = &H1[(size_t)l * T_ * M_];
      if (mm.swiglu) {
        float *gp = &Gpre[(size_t)l * T_ * M_];
        float *up = &Upre[(size_t)l * T_ * M_];
        gemm(false, true, T_, M_, E_, 1, ms, E_, Pp + mm.o_fc1[l], E_, 0, gp,
             M_);
        gemm(false, true, T_, M_, E_, 1, ms, E_, Pp + mm.o_fc3[l], E_, 0, up,
             M_);
        for (size_t i = 0; i < (size_t)T_ * M_; i++)
          h1[i] = silu_f(gp[i]) * up[i];
      } else {
        gemm(false, true, T_, M_, E_, 1, ms, E_, Pp + mm.o_fc1[l], E_, 0, h1p,
             M_);
        if (mm.gelu)
          for (size_t i = 0; i < (size_t)T_ * M_; i++)
            h1[i] = gelu_f(h1p[i]);
        else
          for (size_t i = 0; i < (size_t)T_ * M_; i++)
            h1[i] = max(h1p[i], 0.0f);
      }
      gemm(false, true, T_, E_, M_, 1, h1, M_, Pp + mm.o_fc2[l], M_, 0,
           dX.data(), E_);
      float *xout = (l + 1 < L) ? &Ain[(size_t)(l + 1) * T_ * E_] : Xf.data();
      // write into next layer's Ain buffer directly (memcpy'd there anyway)
      for (size_t i = 0; i < (size_t)T_ * E_; i++)
        xout[i] = dX[i] + min[i];
      X = xout;
    }
    const float *head_in = Xf.data();
    if (mm.cfg.finalnorm) {
      rms_rows(Xf.data(), Xfn.data(), 2 * L + 1, mm.o_gfin);
      head_in = Xfn.data();
    }
    // logits + softmax + loss
    gemm(false, true, T_, Vc, E_, 1, head_in, E_, Pp + mm.o_lm, E_, 0,
         Probs.data(), Vc);
    float loss = 0;
    for (int t = 0; t < T_; t++) {
      float *row = &Probs[(size_t)t * Vc];
      float mx = row[0];
      for (int j = 1; j < Vc; j++)
        mx = max(mx, row[j]);
      float tot = 0;
      for (int j = 0; j < Vc; j++) {
        row[j] = expf(row[j] - mx);
        tot += row[j];
      }
      float inv = 1.0f / tot;
      for (int j = 0; j < Vc; j++)
        row[j] *= inv;
      loss += -logf(row[tokens[t + 1]] + 1e-12f);
    }
    loss /= T_;
    if (!accumulate_grads)
      return loss;

    // ---------- backward ----------
    // dLogits reuses Probs buffer
    for (int t = 0; t < T_; t++) {
      float *row = &Probs[(size_t)t * Vc];
      row[tokens[t + 1]] -= 1.0f;
      for (int j = 0; j < Vc; j++)
        row[j] /= T_;
    }
    float *dlog = Probs.data();
    // lm head
    gemm(true, false, Vc, E_, T_, 1, dlog, Vc, head_in, E_, 1,
         &Glocal[mm.o_lm], E_);
    fill(dX.begin(), dX.end(), 0.0f);
    if (mm.cfg.finalnorm) {
      gemm(false, false, T_, E_, Vc, 1, dlog, Vc, Pp + mm.o_lm, E_, 0,
           dXn.data(), E_);
      rms_rows_bwd(Xf.data(), dXn.data(), dX.data(), 2 * L + 1, mm.o_gfin);
    } else {
      gemm(false, false, T_, E_, Vc, 1, dlog, Vc, Pp + mm.o_lm, E_, 0,
           dX.data(), E_);
    }

    for (int l = L - 1; l >= 0; l--) {
      float *min = &Min[(size_t)l * T_ * E_];
      float *mn = &Mnorm[(size_t)l * T_ * E_];
      float *ms = mm.cfg.dwconv ? &Xsm[(size_t)l * T_ * E_] : mn;
      float *h1p = &H1pre[(size_t)l * T_ * M_];
      float *h1 = &H1[(size_t)l * T_ * M_];
      // MLP backward: dX is grad at block output
      gemm(false, false, T_, M_, E_, 1, dX.data(), E_, Pp + mm.o_fc2[l], M_, 0,
           dH1.data(), M_);
      gemm(true, false, E_, M_, T_, 1, dX.data(), E_, h1, M_, 1,
           &Glocal[mm.o_fc2[l]], M_);
      memcpy(dMin.data(), dX.data(), sizeof(float) * T_ * E_); // residual
      fill(dXn.begin(), dXn.end(), 0.0f);
      if (mm.swiglu) {
        float *gp = &Gpre[(size_t)l * T_ * M_];
        float *up = &Upre[(size_t)l * T_ * M_];
        // reuse H1 buffer for du, dH1 for dg
        for (size_t i = 0; i < (size_t)T_ * M_; i++) {
          float dh = dH1[i];
          h1[i] = dh * silu_f(gp[i]);                 // du
          dH1[i] = dh * up[i] * silu_df(gp[i]);       // dg
        }
        gemm(false, false, T_, E_, M_, 1, dH1.data(), M_, Pp + mm.o_fc1[l], E_,
             0, dXn.data(), E_);
        gemm(true, false, M_, E_, T_, 1, dH1.data(), M_, ms, E_, 1,
             &Glocal[mm.o_fc1[l]], E_);
        gemm(false, false, T_, E_, M_, 1, h1, M_, Pp + mm.o_fc3[l], E_, 1,
             dXn.data(), E_);
        gemm(true, false, M_, E_, T_, 1, h1, M_, ms, E_, 1,
             &Glocal[mm.o_fc3[l]], E_);
      } else {
        if (mm.gelu)
          for (size_t i = 0; i < (size_t)T_ * M_; i++)
            dH1[i] *= gelu_df(h1p[i]);
        else
          for (size_t i = 0; i < (size_t)T_ * M_; i++)
            dH1[i] *= (h1p[i] > 0 ? 1.0f : 0.0f);
        gemm(false, false, T_, E_, M_, 1, dH1.data(), M_, Pp + mm.o_fc1[l], E_,
             0, dXn.data(), E_);
        gemm(true, false, M_, E_, T_, 1, dH1.data(), M_, ms, E_, 1,
             &Glocal[mm.o_fc1[l]], E_);
      }
      if (mm.cfg.dwconv) { // dXn is grad wrt conv output; pull back to mn
        fill(dXs.begin(), dXs.end(), 0.0f);
        dwconv_bwd(mn, dXn.data(), dXs.data(), mm.o_dwm[l]);
        rms_rows_bwd(min, dXs.data(), dMin.data(), 2 + 2 * l, mm.o_gmlp[l]);
      } else {
        rms_rows_bwd(min, dXn.data(), dMin.data(), 2 + 2 * l, mm.o_gmlp[l]);
      }
      // attention block backward; dMin is grad at attn-block output
      float *ain = &Ain[(size_t)l * T_ * E_];
      float *xn = &Anorm[(size_t)l * T_ * E_];
      float *q = &Q[(size_t)l * T_ * E_];
      float *k = &K[(size_t)l * T_ * E_];
      float *v = &Vv[(size_t)l * T_ * E_];
      float *attn = &Attn[(size_t)l * T_ * E_];
      float *xs = mm.cfg.dwconv ? &Xsa[(size_t)l * T_ * E_] : xn;
      float *attn_used =
          mm.cfg.attngate ? &AttnG[(size_t)l * T_ * E_] : attn;
      gemm(false, false, T_, E_, E_, 1, dMin.data(), E_, Pp + mm.o_wo[l], E_,
           0, dAttn.data(), E_);
      gemm(true, false, E_, E_, T_, 1, dMin.data(), E_, attn_used, E_, 1,
           &Glocal[mm.o_wo[l]], E_);
      if (mm.cfg.attngate) { // dAttn is grad wrt gated output
        float *g = &Gt[(size_t)l * T_ * E_];
        for (size_t i = 0; i < (size_t)T_ * E_; i++) {
          float da = dAttn[i];
          dGp[i] = da * attn[i] * g[i] * (1 - g[i]); // grad wrt gate pre-act
          dAttn[i] = da * g[i];                      // grad wrt raw attn
        }
      }
      // per-head attention backward
      fill(dQ.begin(), dQ.end(), 0.0f);
      fill(dK.begin(), dK.end(), 0.0f);
      fill(dV.begin(), dV.end(), 0.0f);
      for (int h = 0; h < H_; h++) {
        float *Sh = &S[(((size_t)l * H_) + h) * T_ * T_];
        // dS = dAttn_h * V_h^T
        gemm(false, true, T_, T_, D_, 1, dAttn.data() + h * D_, E_, v + h * D_,
             E_, 0, dS.data(), T_);
        // dV_h += S^T * dAttn_h
        gemm(true, false, T_, D_, T_, 1, Sh, T_, dAttn.data() + h * D_, E_, 1,
             dV.data() + h * D_, E_);
        // softmax backward per row (masked cols already have S=0, dS ignored)
        for (int t = 0; t < T_; t++) {
          float *sr_ = Sh + (size_t)t * T_;
          float *dsr = &dS[(size_t)t * T_];
          float dot = 0;
          for (int u = 0; u <= t; u++)
            dot += sr_[u] * dsr[u];
          for (int u = 0; u <= t; u++)
            dsr[u] = sr_[u] * (dsr[u] - dot);
          for (int u = t + 1; u < T_; u++)
            dsr[u] = 0.0f;
        }
        // dQ_h = dS * K_h * scale ; dK_h = dS^T * Q_h * scale
        gemm(false, false, T_, D_, T_, mm.inv_sqrt_d, dS.data(), T_,
             k + h * D_, E_, 0, dQ.data() + h * D_, E_);
        gemm(true, false, T_, D_, T_, mm.inv_sqrt_d, dS.data(), T_, q + h * D_,
             E_, 0, dK.data() + h * D_, E_);
      }
      if (mm.cfg.qknorm) { // dQ is grad wrt gamma_h * rope(q_hat)
        for (int h = 0; h < H_; h++) {
          float gh = Pp[mm.o_qkg[l] + h];
          float dg = 0;
          for (int t = 0; t < T_; t++) {
            const float *qf = q + (size_t)t * E_ + h * D_;
            float *dq = dQ.data() + (size_t)t * E_ + h * D_;
            for (int i = 0; i < D_; i++) {
              dg += dq[i] * qf[i];
              dq[i] *= gh; // now grad wrt rope(q_hat)
            }
          }
          Glocal[mm.o_qkg[l] + h] += dg / gh;
        }
      }
      if (mm.cfg.rope) { rope_rows(dQ.data(), -1); rope_rows(dK.data(), -1); }
      if (mm.cfg.qknorm) {
        qknorm_bwd(&Qraw[(size_t)l * T_ * E_], &sq[(size_t)l * T_ * H_],
                   dQ.data());
        qknorm_bwd(&Kraw[(size_t)l * T_ * E_], &sk[(size_t)l * T_ * H_],
                   dK.data());
      }
      if (mm.cfg.valres) { // dV is grad wrt mixed V
        if (l > 0) {
          const float *lam = Pp + mm.o_vres[l];
          const float *vraw = &Vraw[(size_t)l * T_ * E_];
          const float *v0 = &Vv[0];
          for (int t = 0; t < T_; t++)
            for (int i = 0; i < E_; i++) {
              size_t ix = (size_t)t * E_ + i;
              float a = 1.0f / (1.0f + expf(-lam[i]));
              float dv = dV[ix];
              Glocal[mm.o_vres[l] + i] +=
                  dv * (vraw[ix] - v0[ix]) * a * (1 - a);
              dV0acc[ix] += (1 - a) * dv;
              dV[ix] = a * dv; // grad wrt this layer's raw V
            }
        } else { // layer 0's raw V also fed every later layer's mix
          for (size_t i = 0; i < (size_t)T_ * E_; i++) {
            dV[i] += dV0acc[i];
            dV0acc[i] = 0.0f;
          }
        }
      }
      // q/k/v projections (inputs were xs = conv(xn) when dwconv is on)
      fill(dXn.begin(), dXn.end(), 0.0f);
      gemm(false, false, T_, E_, E_, 1, dQ.data(), E_, Pp + mm.o_wq[l], E_, 1,
           dXn.data(), E_);
      gemm(true, false, E_, E_, T_, 1, dQ.data(), E_, xs, E_, 1,
           &Glocal[mm.o_wq[l]], E_);
      gemm(false, false, T_, E_, E_, 1, dK.data(), E_, Pp + mm.o_wk[l], E_, 1,
           dXn.data(), E_);
      gemm(true, false, E_, E_, T_, 1, dK.data(), E_, xs, E_, 1,
           &Glocal[mm.o_wk[l]], E_);
      gemm(false, false, T_, E_, E_, 1, dV.data(), E_, Pp + mm.o_wv[l], E_, 1,
           dXn.data(), E_);
      gemm(true, false, E_, E_, T_, 1, dV.data(), E_, xs, E_, 1,
           &Glocal[mm.o_wv[l]], E_);
      if (mm.cfg.attngate) { // gate also consumed xs
        gemm(false, false, T_, E_, E_, 1, dGp.data(), E_, Pp + mm.o_wg[l], E_,
             1, dXn.data(), E_);
        gemm(true, false, E_, E_, T_, 1, dGp.data(), E_, xs, E_, 1,
             &Glocal[mm.o_wg[l]], E_);
      }
      if (mm.cfg.dwconv) { // dXn is grad wrt conv output; pull back to xn
        fill(dXs.begin(), dXs.end(), 0.0f);
        dwconv_bwd(xn, dXn.data(), dXs.data(), mm.o_dwa[l]);
        rms_rows_bwd(ain, dXs.data(), dMin.data(), 1 + 2 * l, mm.o_gattn[l]);
      } else {
        rms_rows_bwd(ain, dXn.data(), dMin.data(), 1 + 2 * l, mm.o_gattn[l]);
      }
      // dMin now holds grad at block input == grad at previous block output
      memcpy(dX.data(), dMin.data(), sizeof(float) * T_ * E_);
    }
    // embedding backward
    fill(dXn.begin(), dXn.end(), 0.0f);
    rms_rows_bwd(X0pre.data(), dX.data(), dXn.data(), 0, mm.o_gemb);
    for (int t = 0; t < T_; t++) {
      float *g = &Glocal[mm.o_wte + (size_t)tokens[t] * E_];
      const float *dxp = &dXn[(size_t)t * E_];
      for (int i = 0; i < E_; i++)
        g[i] += dxp[i];
      if (!mm.cfg.rope) {
        float *gp = &Glocal[mm.o_wpe + (size_t)t * E_];
        for (int i = 0; i < E_; i++)
          gp[i] += dxp[i];
      }
    }
    return loss;
  }
};

int main(int argc, char **argv) {
  Config cfg;
  for (int i = 1; i < argc; i++) {
    string a = argv[i];
    auto next = [&]() { return string(argv[++i]); };
    if (a == "--layers") cfg.layers = stoi(next());
    else if (a == "--embd") cfg.embd = stoi(next());
    else if (a == "--heads") cfg.heads = stoi(next());
    else if (a == "--hidden") cfg.hidden = stoi(next());
    else if (a == "--block") cfg.block = stoi(next());
    else if (a == "--steps") cfg.steps = stoi(next());
    else if (a == "--seed") cfg.seed = stoi(next());
    else if (a == "--lr") cfg.lr = stof(next());
    else if (a == "--mlp") cfg.mlp = next();
    else if (a == "--batch") cfg.batch = stoi(next());
    else if (a == "--threads") cfg.threads = stoi(next());
    else if (a == "--warmup") cfg.warmup = stoi(next());
    else if (a == "--eval-every") cfg.eval_every = stoi(next());
    else if (a == "--data") cfg.data = next();
    else if (a == "--save") cfg.save = next();
    else if (a == "--load") cfg.load = next();
    else if (a == "--rope") cfg.rope = true;
    else if (a == "--tie") cfg.tie = true;
    else if (a == "--finalnorm") cfg.finalnorm = true;
    else if (a == "--gains") cfg.gains = true;
    else if (a == "--residscale") cfg.residscale = true;
    else if (a == "--gradcheck") cfg.gradcheck = true;
    else if (a == "--noblas") cfg.noblas = true;
    else if (a == "--nofinal") cfg.nofinal = true;
    else if (a == "--wd") cfg.wd = stof(next());
    else if (a == "--clip") cfg.clip = stof(next());
    else if (a == "--evalstride") cfg.evalstride = stoi(next());
    else if (a == "--dwconv") cfg.dwconv = true;
    else if (a == "--valres") cfg.valres = true;
    else if (a == "--attngate") cfg.attngate = true;
    else if (a == "--qknorm") cfg.qknorm = true;
    else if (a == "--softmax1") cfg.softmax1 = true;
    else { cerr << "unknown arg " << a << endl; return 1; }
  }
  g_use_blas = !cfg.noblas;
  if (cfg.threads <= 0)
    cfg.threads = min((unsigned)16, thread::hardware_concurrency());
  cfg.threads = min(cfg.threads, cfg.batch);

  // load data
  ifstream f(cfg.data, ios::binary);
  if (!f) { cerr << "cannot open " << cfg.data << endl; return 1; }
  vector<uint8_t> data((istreambuf_iterator<char>(f)),
                       istreambuf_iterator<char>());
  long N = (long)data.size();
  long train_n = min(cfg.train_bytes, N);
  long val_start = train_n;
  long val_n = min(cfg.val_bytes, N - train_n);
  long test_start = val_start + val_n;
  long test_n = N - test_start;

  mt19937 rng(cfg.seed);
  Model model;
  model.cfg = cfg;
  model.init(rng);
  if (!cfg.load.empty()) {
    ifstream lf(cfg.load, ios::binary);
    lf.read((char *)model.P.data(), model.NP * sizeof(float));
    if (!lf) { cerr << "failed to load " << cfg.load << endl; return 1; }
  }
  cout << "config: layers=" << cfg.layers << " embd=" << cfg.embd
       << " heads=" << cfg.heads << " hidden=" << cfg.hidden
       << " block=" << cfg.block << " mlp=" << cfg.mlp << " rope=" << cfg.rope
       << " tie=" << cfg.tie << " finalnorm=" << cfg.finalnorm
       << " gains=" << cfg.gains << " residscale=" << cfg.residscale
       << " dwconv=" << cfg.dwconv << " valres=" << cfg.valres
       << " attngate=" << cfg.attngate << " qknorm=" << cfg.qknorm
       << " softmax1=" << cfg.softmax1
       << " batch=" << cfg.batch << " threads=" << cfg.threads
       << " blas=" << (g_use_blas ? 1 : 0) << " steps=" << cfg.steps
       << " lr=" << cfg.lr << " seed=" << cfg.seed << " params=" << model.NP
       << " | data: train=" << train_n << " val=" << val_n
       << " test=" << test_n << endl;

  vector<Workspace> ws(cfg.threads);
  for (auto &w : ws)
    w.alloc(&model);

  if (cfg.gradcheck) {
    Workspace &w = ws[0];
    vector<uint8_t> tok(data.begin(), data.begin() + cfg.block + 1);
    w.run_window(tok.data(), true);
    vector<float> Ga = w.Glocal;
    vector<int> order(model.NP);
    for (int i = 0; i < model.NP; i++)
      order[i] = i;
    sort(order.begin(), order.end(),
         [&](int a, int b) { return fabs(Ga[a]) > fabs(Ga[b]); });
    float worst = 0;
    int checked = 0;
    for (int r = 0; r < 60; r++) {
      int idx = order[r];
      float h = 2e-3f, orig = model.P[idx];
      model.P[idx] = orig + h;
      float lp = w.run_window(tok.data(), false);
      model.P[idx] = orig - h;
      float lm_ = w.run_window(tok.data(), false);
      model.P[idx] = orig;
      float num = (lp - lm_) / (2 * h);
      float rel = fabs(num - Ga[idx]) / max(1e-3f, fabs(num) + fabs(Ga[idx]));
      worst = max(worst, rel);
      checked++;
    }
    cout << "gradcheck: " << checked << " params, worst rel err = " << worst
         << endl;
    return worst < 0.05 ? 0 : 2;
  }

  const double LN2 = 0.6931471805599453;
  // evaluate bpc over a byte range, threaded. stride=0: consecutive
  // non-overlapping windows (fast, context resets). stride>0: overlapping
  // windows scoring only the last `stride` bytes of each, so every scored
  // byte sees >= block-stride bytes of context (window 0 scores all).
  auto eval_range = [&](long start, long nbytes, int stride) {
    int T = cfg.block;
    long step_ = stride > 0 ? stride : T;
    long nwin = max(1L, (nbytes - (stride > 0 ? (long)(T - stride) : 0L)) /
                            step_);
    atomic<long> next_win(0);
    vector<double> sums(cfg.threads, 0.0);
    vector<long> cnts(cfg.threads, 0);
    vector<thread> th;
    for (int ti = 0; ti < cfg.threads; ti++)
      th.emplace_back([&, ti]() {
        long wi;
        double s = 0;
        long c = 0;
        while ((wi = next_win.fetch_add(1)) < nwin) {
          long off = start + wi * step_;
          if (off + T + 1 > N)
            break;
          int from = (stride > 0 && wi > 0) ? T - stride : 0;
          ws[ti].run_window(&data[off], false);
          for (int t = from; t < T; t++) {
            s += -log((double)ws[ti].Probs[(size_t)t * Model::V +
                                           data[off + t + 1]] +
                      1e-12);
            c++;
          }
        }
        sums[ti] = s;
        cnts[ti] = c;
      });
    for (auto &t : th)
      t.join();
    double total = 0;
    long count = 0;
    for (int ti = 0; ti < cfg.threads; ti++) {
      total += sums[ti];
      count += cnts[ti];
    }
    return total / max(1L, count) / LN2; // bpc
  };

  float beta1 = 0.9f, beta2 = 0.95f, eps_adam = 1e-8f;
  double run_loss = 0;
  long run_cnt = 0;
  auto t_start = chrono::steady_clock::now();
  long tokens_done = 0;
  for (int step = 0; step < cfg.steps; step++) {
    // sample window offsets (deterministic, from main rng)
    vector<long> offs(cfg.batch);
    for (int b = 0; b < cfg.batch; b++)
      offs[b] = uniform_int_distribution<long>(
          0, train_n - cfg.block - 2)(rng);
    // workers
    atomic<int> next_b(0);
    vector<thread> th;
    for (int ti = 0; ti < cfg.threads; ti++)
      th.emplace_back([&, ti]() {
        ws[ti].loss_sum = 0;
        ws[ti].tok_count = 0;
        int b;
        while ((b = next_b.fetch_add(1)) < cfg.batch) {
          float l = ws[ti].run_window(&data[offs[b]], true);
          ws[ti].loss_sum += l;
          ws[ti].tok_count++;
        }
      });
    for (auto &t : th)
      t.join();
    for (int ti = 0; ti < cfg.threads; ti++) {
      run_loss += ws[ti].loss_sum;
      run_cnt += ws[ti].tok_count;
    }
    tokens_done += (long)cfg.batch * cfg.block;
    // reduce grads (parallel over param chunks, deterministic worker order)
    // + Adam in the same pass
    float lr_t;
    if (step < cfg.warmup)
      lr_t = cfg.lr * (step + 1) / cfg.warmup;
    else {
      float prog = (float)(step - cfg.warmup) / max(1, cfg.steps - cfg.warmup);
      lr_t = cfg.lr * 0.5f * (1 + cosf(prog * (float)M_PI));
    }
    float bc1 = 1 - powf(beta1, step + 1);
    float bc2 = 1 - powf(beta2, step + 1);
    {
      int nch = cfg.threads;
      long chunk = (model.NP + nch - 1) / nch;
      // phase 1: reduce thread-local grads into model.G, gather sumsq
      vector<double> ss(nch, 0.0);
      vector<thread> rt;
      for (int ci = 0; ci < nch; ci++)
        rt.emplace_back([&, ci]() {
          long lo = ci * chunk, hi = min<long>(model.NP, lo + chunk);
          double s2 = 0;
          for (long i = lo; i < hi; i++) {
            float g = 0;
            for (int ti = 0; ti < cfg.threads; ti++) {
              g += ws[ti].Glocal[i];
              ws[ti].Glocal[i] = 0.0f;
            }
            g /= cfg.batch;
            model.G[i] = g;
            s2 += (double)g * g;
          }
          ss[ci] = s2;
        });
      for (auto &t : rt)
        t.join();
      float scale = 1.0f;
      if (cfg.clip > 0) {
        double n2 = 0;
        for (double s : ss)
          n2 += s;
        float norm = (float)sqrt(n2);
        if (norm > cfg.clip)
          scale = cfg.clip / norm;
      }
      // phase 2: AdamW
      vector<thread> at;
      for (int ci = 0; ci < nch; ci++)
        at.emplace_back([&, ci]() {
          long lo = ci * chunk, hi = min<long>(model.NP, lo + chunk);
          for (long i = lo; i < hi; i++) {
            float g = model.G[i] * scale;
            model.am[i] = beta1 * model.am[i] + (1 - beta1) * g;
            model.av[i] = beta2 * model.av[i] + (1 - beta2) * g * g;
            model.P[i] -= lr_t * (model.am[i] / bc1) /
                          (sqrtf(model.av[i] / bc2) + eps_adam);
            if (cfg.wd > 0 && i < model.wd_end)
              model.P[i] -= lr_t * cfg.wd * model.P[i];
          }
        });
      for (auto &t : at)
        t.join();
    }

    if ((step + 1) % cfg.eval_every == 0 || step + 1 == cfg.steps) {
      auto now = chrono::steady_clock::now();
      double secs = chrono::duration<double>(now - t_start).count();
      double vb = eval_range(val_start, min(cfg.eval_bytes, val_n), 0);
      cout << "step " << (step + 1) << " | train bpc " << setprecision(4)
           << (run_loss / run_cnt / LN2) << " | val bpc " << vb << " | "
           << (long)(tokens_done / secs) << " tok/s" << endl;
      run_loss = 0;
      run_cnt = 0;
    }
  }

  // final full evaluation
  if (cfg.nofinal)
    return 0;
  double val_bpc = eval_range(val_start, val_n, cfg.evalstride);
  double test_bpc = eval_range(test_start, test_n, cfg.evalstride);
  cout << "RESULT params=" << model.NP << " val_bpc=" << setprecision(5)
       << val_bpc << " test_bpc=" << test_bpc << endl;

  if (!cfg.save.empty()) {
    ofstream sf(cfg.save, ios::binary);
    sf.write((const char *)model.P.data(), model.NP * sizeof(float));
    cout << "saved " << cfg.save << endl;
  }

  // sample: prime with a val snippet, generate 512 bytes greedy-ish (temp .8)
  {
    Workspace &w = ws[0];
    int T = cfg.block;
    vector<uint8_t> ctx(data.begin() + val_start, data.begin() + val_start + 64);
    mt19937 sg(cfg.seed + 999);
    string out;
    for (int gen = 0; gen < 512; gen++) {
      int L0 = (int)ctx.size();
      int off = max(0, L0 - T);
      vector<uint8_t> win(ctx.begin() + off, ctx.end());
      int tpos = (int)win.size() - 1;
      win.resize(T + 1, 0); // pad; positions after tpos don't affect tpos
      w.run_window(win.data(), false);
      float *row = &w.Probs[(size_t)tpos * Model::V];
      // temperature over the probs: p^(1/temp) renormalized
      double temp = 0.8, tot = 0;
      vector<double> pw(Model::V);
      for (int j = 0; j < Model::V; j++) {
        pw[j] = pow((double)row[j], 1.0 / temp);
        tot += pw[j];
      }
      double r = uniform_real_distribution<double>(0, tot)(sg), acc = 0;
      int pick = 0;
      for (int j = 0; j < Model::V; j++) {
        acc += pw[j];
        if (r <= acc) { pick = j; break; }
      }
      ctx.push_back((uint8_t)pick);
      out += (char)pick;
    }
    cout << "--- sample (primed with 64 bytes of val) ---" << endl;
    for (char c : out)
      cout << (isprint((unsigned char)c) || c == '\n' ? c : '?');
    cout << endl;
  }
  return 0;
}
