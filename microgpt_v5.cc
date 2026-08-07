// V5: architecture-improvement lab built on V4's hand-written backprop.
// Adds a train/val split (val loss = model quality metric) and switchable
// architectural upgrades, each with hand-derived gradients:
//   --finalnorm   final RMSNorm before the LM head (GPT-2 has this)
//   --tie         weight tying: lm_head shares the wte matrix
//   --rope        rotary position embeddings instead of learned wpe
//   --mlp relu|gelu|swiglu
//   --residscale  init wo/fc2 scaled by 1/sqrt(2*n_layer)
// Size/training knobs: --embd --layers --heads --hidden --steps --lr --seed
// Sanity: --gradcheck runs a numerical-vs-analytic gradient comparison.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

struct Config {
  int layers = 4, embd = 16, heads = 4, block = 16, hidden = 64;
  int steps = 10000, seed = 42, eval_every = 2000, batch = 1;
  float lr = 0.01f;
  bool rope = false, tie = false, finalnorm = false, residscale = false;
  bool gains = false; // learnable per-channel rmsnorm gains (LLaMA-style)
  bool gradcheck = false;
  string mlp = "relu";
};

static void matvec(const float *W, const float *x, float *y, int rows,
                   int cols) {
  for (int i = 0; i < rows; i++) {
    float s = 0;
    const float *w = W + i * cols;
    for (int j = 0; j < cols; j++)
      s += w[j] * x[j];
    y[i] = s;
  }
}
static void matvecT_acc(const float *W, const float *dy, float *dx, int rows,
                        int cols) {
  for (int i = 0; i < rows; i++) {
    float d = dy[i];
    const float *w = W + i * cols;
    for (int j = 0; j < cols; j++)
      dx[j] += w[j] * d;
  }
}
static void outer_acc(float *dW, const float *dy, const float *x, int rows,
                      int cols) {
  for (int i = 0; i < rows; i++) {
    float d = dy[i];
    float *w = dW + i * cols;
    for (int j = 0; j < cols; j++)
      w[j] += d * x[j];
  }
}
static void rms_fwd(const float *x, float *y, float *s_out, int n) {
  float ms = 0;
  for (int i = 0; i < n; i++)
    ms += x[i] * x[i];
  ms /= n;
  float s = powf(ms + 1e-5f, -0.5f);
  for (int i = 0; i < n; i++)
    y[i] = x[i] * s;
  *s_out = s;
}
static void rms_bwd(const float *x, float s, const float *dy, float *dx_acc,
                    int n) {
  float dyx = 0;
  for (int i = 0; i < n; i++)
    dyx += dy[i] * x[i];
  float c = s * s * s * dyx / n;
  for (int i = 0; i < n; i++)
    dx_acc[i] += s * dy[i] - c * x[i];
}
static void softmax_fwd(const float *logits, float *probs, int n) {
  float mx = logits[0];
  for (int i = 1; i < n; i++)
    mx = max(mx, logits[i]);
  float total = 0;
  for (int i = 0; i < n; i++) {
    probs[i] = expf(logits[i] - mx);
    total += probs[i];
  }
  float inv = 1.0f / total;
  for (int i = 0; i < n; i++)
    probs[i] *= inv;
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

struct Model {
  Config cfg;
  int vocab, D; // head dim
  int o_wte, o_wpe, o_lm, o_gemb, o_gfin;
  vector<int> o_wq, o_wk, o_wv, o_wo, o_fc1, o_fc2, o_fc3, o_gattn, o_gmlp;
  int NP;
  vector<float> P, G, am, av;
  bool swiglu, gelu;
  float inv_sqrt_d;
  vector<vector<float>> rope_cos, rope_sin; // [block][D/2]

  // activations
  vector<vector<float>> x0pre, x0, xf, xfn, probs;
  vector<float> s0, sfin;
  vector<vector<vector<float>>> ain, anorm, q, K, V, attw, attnout, min_,
      mnorm, h1pre, h1, gpre, upre, dK, dV;
  vector<vector<float>> sattn, smlp;
  vector<float> xbuf, tmp, logits, dlog, dx, dmin, dattn, dain, danorm, dq_,
      dx0pre, dh1, dmnorm, dg_, du_, dkrot;

  void init(int vocab_, mt19937 &rng) {
    vocab = vocab_;
    D = cfg.embd / cfg.heads;
    inv_sqrt_d = 1.0f / sqrtf((float)D);
    swiglu = cfg.mlp == "swiglu";
    gelu = cfg.mlp == "gelu";
    int L = cfg.layers, E = cfg.embd, M = cfg.hidden, BS = cfg.block;
    int off = 0;
    o_wte = off; off += vocab * E;
    if (!cfg.rope) { o_wpe = off; off += BS * E; } else o_wpe = -1;
    if (!cfg.tie) { o_lm = off; off += vocab * E; } else o_lm = o_wte;
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
    normal_distribution<float> d(0, 0.08f);
    for (int i = 0; i < NP; i++)
      P[i] = d(rng);
    if (cfg.gains) { // gains start at identity
      for (int i = 0; i < E; i++)
        P[o_gemb + i] = 1.0f;
      for (int l = 0; l < L; l++)
        for (int i = 0; i < E; i++) {
          P[o_gattn[l] + i] = 1.0f;
          P[o_gmlp[l] + i] = 1.0f;
        }
      if (cfg.finalnorm)
        for (int i = 0; i < E; i++)
          P[o_gfin + i] = 1.0f;
    }
    if (cfg.residscale) {
      float sc = 1.0f / sqrtf(2.0f * L);
      for (int l = 0; l < L; l++) {
        for (int i = 0; i < E * E; i++)
          P[o_wo[l] + i] *= sc;
        for (int i = 0; i < E * M; i++)
          P[o_fc2[l] + i] *= sc;
      }
    }
    if (cfg.rope) {
      rope_cos.assign(BS, vector<float>(D / 2));
      rope_sin.assign(BS, vector<float>(D / 2));
      for (int t = 0; t < BS; t++)
        for (int i = 0; i < D / 2; i++) {
          float freq = powf(10000.0f, -2.0f * i / D);
          rope_cos[t][i] = cosf(t * freq);
          rope_sin[t][i] = sinf(t * freq);
        }
    }
    // activation buffers
    auto f2 = [&](int a, int b) {
      return vector<vector<float>>(a, vector<float>(b));
    };
    auto f3 = [&](int b) {
      return vector<vector<vector<float>>>(L, f2(BS, b));
    };
    x0pre = f2(BS, E); x0 = f2(BS, E); xf = f2(BS, E); xfn = f2(BS, E);
    probs = f2(BS, vocab);
    s0.assign(BS, 0); sfin.assign(BS, 0);
    ain = f3(E); anorm = f3(E); q = f3(E); K = f3(E); V = f3(E);
    attw = f3(cfg.heads * BS); attnout = f3(E); min_ = f3(E); mnorm = f3(E);
    h1pre = f3(M); h1 = f3(M); gpre = f3(M); upre = f3(M);
    dK = f3(E); dV = f3(E);
    sattn = f2(L, BS); smlp = f2(L, BS);
    xbuf.resize(E); tmp.resize(E); logits.resize(vocab); dlog.resize(vocab);
    dx.resize(E); dmin.resize(E); dattn.resize(E); dain.resize(E);
    danorm.resize(E); dq_.resize(E); dx0pre.resize(E); dh1.resize(M);
    dmnorm.resize(E); dg_.resize(M); du_.resize(M); dkrot.resize(E);
    dyg.resize(E);
  }

  void gain_fwd(float *y, int goff, int n) {
    if (goff < 0)
      return;
    for (int i = 0; i < n; i++)
      y[i] *= P[goff + i];
  }
  // backward through [rmsnorm -> per-channel gain]: dy is the grad wrt the
  // gained output; accumulates gain grads and the input grad.
  void gain_rms_bwd(const float *x, float s, const float *dy, float *dx_acc,
                    int goff, int n) {
    if (goff < 0) {
      rms_bwd(x, s, dy, dx_acc, n);
      return;
    }
    for (int i = 0; i < n; i++) {
      G[goff + i] += dy[i] * x[i] * s; // xhat_i = x_i * s
      dyg[i] = dy[i] * P[goff + i];
    }
    rms_bwd(x, s, dyg.data(), dx_acc, n);
  }
  vector<float> dyg;

  void rope_apply(float *v, int t) { // rotate all heads in place
    for (int h = 0; h < cfg.heads; h++) {
      int hs = h * D;
      for (int i = 0; i < D / 2; i++) {
        float c = rope_cos[t][i], s = rope_sin[t][i];
        float a = v[hs + 2 * i], b = v[hs + 2 * i + 1];
        v[hs + 2 * i] = a * c - b * s;
        v[hs + 2 * i + 1] = a * s + b * c;
      }
    }
  }
  void rope_apply_inv(float *v, int t) { // transpose rotation (for grads)
    for (int h = 0; h < cfg.heads; h++) {
      int hs = h * D;
      for (int i = 0; i < D / 2; i++) {
        float c = rope_cos[t][i], s = rope_sin[t][i];
        float a = v[hs + 2 * i], b = v[hs + 2 * i + 1];
        v[hs + 2 * i] = a * c + b * s;
        v[hs + 2 * i + 1] = -a * s + b * c;
      }
    }
  }

  void forward_pos(int t, int token_id) {
    int E = cfg.embd, L = cfg.layers, H = cfg.heads, M = cfg.hidden,
        BS = cfg.block;
    for (int i = 0; i < E; i++)
      x0pre[t][i] = P[o_wte + token_id * E + i] +
                    (cfg.rope ? 0.0f : P[o_wpe + t * E + i]);
    rms_fwd(x0pre[t].data(), x0[t].data(), &s0[t], E);
    gain_fwd(x0[t].data(), o_gemb, E);
    xbuf = x0[t];
    for (int l = 0; l < L; l++) {
      ain[l][t] = xbuf;
      rms_fwd(xbuf.data(), anorm[l][t].data(), &sattn[l][t], E);
      gain_fwd(anorm[l][t].data(), o_gattn[l], E);
      matvec(&P[o_wq[l]], anorm[l][t].data(), q[l][t].data(), E, E);
      matvec(&P[o_wk[l]], anorm[l][t].data(), K[l][t].data(), E, E);
      matvec(&P[o_wv[l]], anorm[l][t].data(), V[l][t].data(), E, E);
      if (cfg.rope) {
        rope_apply(q[l][t].data(), t);
        rope_apply(K[l][t].data(), t); // cache holds rotated keys
      }
      for (int h = 0; h < H; h++) {
        int hs = h * D;
        vector<float> lg(t + 1);
        for (int u = 0; u <= t; u++) {
          float s = 0;
          for (int j = 0; j < D; j++)
            s += q[l][t][hs + j] * K[l][u][hs + j];
          lg[u] = s * inv_sqrt_d;
        }
        softmax_fwd(lg.data(), &attw[l][t][h * BS], t + 1);
        for (int j = 0; j < D; j++) {
          float s = 0;
          for (int u = 0; u <= t; u++)
            s += attw[l][t][h * BS + u] * V[l][u][hs + j];
          attnout[l][t][hs + j] = s;
        }
      }
      matvec(&P[o_wo[l]], attnout[l][t].data(), tmp.data(), E, E);
      for (int i = 0; i < E; i++)
        xbuf[i] = tmp[i] + ain[l][t][i];
      min_[l][t] = xbuf;
      rms_fwd(xbuf.data(), mnorm[l][t].data(), &smlp[l][t], E);
      gain_fwd(mnorm[l][t].data(), o_gmlp[l], E);
      if (swiglu) {
        matvec(&P[o_fc1[l]], mnorm[l][t].data(), gpre[l][t].data(), M, E);
        matvec(&P[o_fc3[l]], mnorm[l][t].data(), upre[l][t].data(), M, E);
        for (int i = 0; i < M; i++)
          h1[l][t][i] = silu_f(gpre[l][t][i]) * upre[l][t][i];
      } else {
        matvec(&P[o_fc1[l]], mnorm[l][t].data(), h1pre[l][t].data(), M, E);
        for (int i = 0; i < M; i++)
          h1[l][t][i] =
              gelu ? gelu_f(h1pre[l][t][i]) : max(h1pre[l][t][i], 0.0f);
      }
      matvec(&P[o_fc2[l]], h1[l][t].data(), tmp.data(), E, M);
      for (int i = 0; i < E; i++)
        xbuf[i] = tmp[i] + min_[l][t][i];
    }
    xf[t] = xbuf;
    const float *head_in = xbuf.data();
    if (cfg.finalnorm) {
      rms_fwd(xbuf.data(), xfn[t].data(), &sfin[t], cfg.embd);
      gain_fwd(xfn[t].data(), o_gfin, cfg.embd);
      head_in = xfn[t].data();
    }
    matvec(&P[o_lm], head_in, logits.data(), vocab, E);
    softmax_fwd(logits.data(), probs[t].data(), vocab);
  }

  float forward_doc(const vector<int> &tokens, int n) {
    float loss = 0;
    for (int t = 0; t < n; t++) {
      forward_pos(t, tokens[t]);
      loss += -logf(probs[t][tokens[t + 1]]);
    }
    return loss / n;
  }

  void backward_doc(const vector<int> &tokens, int n) {
    int E = cfg.embd, L = cfg.layers, H = cfg.heads, M = cfg.hidden,
        BS = cfg.block;
    for (int l = 0; l < L; l++)
      for (int t = 0; t < n; t++) {
        fill(dK[l][t].begin(), dK[l][t].end(), 0.0f);
        fill(dV[l][t].begin(), dV[l][t].end(), 0.0f);
      }
    for (int t = n - 1; t >= 0; t--) {
      int tgt = tokens[t + 1];
      for (int j = 0; j < vocab; j++)
        dlog[j] = (probs[t][j] - (j == tgt ? 1.0f : 0.0f)) / n;
      fill(dx.begin(), dx.end(), 0.0f);
      const float *head_in = cfg.finalnorm ? xfn[t].data() : xf[t].data();
      if (cfg.finalnorm) {
        fill(tmp.begin(), tmp.end(), 0.0f);
        matvecT_acc(&P[o_lm], dlog.data(), tmp.data(), vocab, E);
        gain_rms_bwd(xf[t].data(), sfin[t], tmp.data(), dx.data(), o_gfin, E);
      } else {
        matvecT_acc(&P[o_lm], dlog.data(), dx.data(), vocab, E);
      }
      outer_acc(&G[o_lm], dlog.data(), head_in, vocab, E);

      for (int l = L - 1; l >= 0; l--) {
        // MLP block
        fill(dh1.begin(), dh1.end(), 0.0f);
        matvecT_acc(&P[o_fc2[l]], dx.data(), dh1.data(), E, M);
        outer_acc(&G[o_fc2[l]], dx.data(), h1[l][t].data(), E, M);
        dmin = dx;
        fill(dmnorm.begin(), dmnorm.end(), 0.0f);
        if (swiglu) {
          for (int i = 0; i < M; i++) {
            float gv = gpre[l][t][i];
            du_[i] = dh1[i] * silu_f(gv);
            dg_[i] = dh1[i] * upre[l][t][i] * silu_df(gv);
          }
          matvecT_acc(&P[o_fc1[l]], dg_.data(), dmnorm.data(), M, E);
          outer_acc(&G[o_fc1[l]], dg_.data(), mnorm[l][t].data(), M, E);
          matvecT_acc(&P[o_fc3[l]], du_.data(), dmnorm.data(), M, E);
          outer_acc(&G[o_fc3[l]], du_.data(), mnorm[l][t].data(), M, E);
        } else {
          for (int i = 0; i < M; i++)
            dh1[i] *= gelu ? gelu_df(h1pre[l][t][i])
                           : (h1pre[l][t][i] > 0 ? 1.0f : 0.0f);
          matvecT_acc(&P[o_fc1[l]], dh1.data(), dmnorm.data(), M, E);
          outer_acc(&G[o_fc1[l]], dh1.data(), mnorm[l][t].data(), M, E);
        }
        gain_rms_bwd(min_[l][t].data(), smlp[l][t], dmnorm.data(), dmin.data(),
                     o_gmlp[l], E);

        // Attention block
        fill(dattn.begin(), dattn.end(), 0.0f);
        matvecT_acc(&P[o_wo[l]], dmin.data(), dattn.data(), E, E);
        outer_acc(&G[o_wo[l]], dmin.data(), attnout[l][t].data(), E, E);
        dain = dmin;
        fill(dq_.begin(), dq_.end(), 0.0f);
        for (int h = 0; h < H; h++) {
          int hs = h * D;
          const float *aw = &attw[l][t][h * BS];
          vector<float> da(t + 1);
          float aw_da = 0;
          for (int u = 0; u <= t; u++) {
            float s = 0;
            for (int j = 0; j < D; j++)
              s += dattn[hs + j] * V[l][u][hs + j];
            da[u] = s;
            aw_da += aw[u] * s;
            for (int j = 0; j < D; j++)
              dV[l][u][hs + j] += aw[u] * dattn[hs + j];
          }
          for (int u = 0; u <= t; u++) {
            float dlg = aw[u] * (da[u] - aw_da);
            for (int j = 0; j < D; j++) {
              dq_[hs + j] += dlg * K[l][u][hs + j] * inv_sqrt_d;
              dK[l][u][hs + j] += dlg * q[l][t][hs + j] * inv_sqrt_d;
            }
          }
        }
        // q/k/v linears at position t (dK/dV[t] complete now)
        dkrot = dK[l][t];
        if (cfg.rope) {
          rope_apply_inv(dq_.data(), t);
          rope_apply_inv(dkrot.data(), t);
        }
        fill(danorm.begin(), danorm.end(), 0.0f);
        matvecT_acc(&P[o_wq[l]], dq_.data(), danorm.data(), E, E);
        outer_acc(&G[o_wq[l]], dq_.data(), anorm[l][t].data(), E, E);
        matvecT_acc(&P[o_wk[l]], dkrot.data(), danorm.data(), E, E);
        outer_acc(&G[o_wk[l]], dkrot.data(), anorm[l][t].data(), E, E);
        matvecT_acc(&P[o_wv[l]], dV[l][t].data(), danorm.data(), E, E);
        outer_acc(&G[o_wv[l]], dV[l][t].data(), anorm[l][t].data(), E, E);
        gain_rms_bwd(ain[l][t].data(), sattn[l][t], danorm.data(), dain.data(),
                     o_gattn[l], E);
        dx = dain;
      }
      fill(dx0pre.begin(), dx0pre.end(), 0.0f);
      gain_rms_bwd(x0pre[t].data(), s0[t], dx.data(), dx0pre.data(), o_gemb,
                   E);
      for (int i = 0; i < E; i++) {
        G[o_wte + tokens[t] * E + i] += dx0pre[i];
        if (!cfg.rope)
          G[o_wpe + t * E + i] += dx0pre[i];
      }
    }
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
    else if (a == "--steps") cfg.steps = stoi(next());
    else if (a == "--seed") cfg.seed = stoi(next());
    else if (a == "--lr") cfg.lr = stof(next());
    else if (a == "--mlp") cfg.mlp = next();
    else if (a == "--batch") cfg.batch = stoi(next());
    else if (a == "--gains") cfg.gains = true;
    else if (a == "--rope") cfg.rope = true;
    else if (a == "--tie") cfg.tie = true;
    else if (a == "--finalnorm") cfg.finalnorm = true;
    else if (a == "--residscale") cfg.residscale = true;
    else if (a == "--gradcheck") cfg.gradcheck = true;
    else { cerr << "unknown arg " << a << endl; return 1; }
  }

  vector<string> docs;
  string doc;
  while (cin >> doc)
    docs.push_back(doc);
  mt19937 g(cfg.seed);
  shuffle(docs.begin(), docs.end(), g);

  set<char> uchars;
  for (const auto &d0 : docs)
    for (char ch : d0)
      uchars.insert(ch);
  int BOS = uchars.size();
  int vocab = uchars.size() + 1;
  unordered_map<char, int> ch_to_token;
  unordered_map<int, char> token_to_ch;
  int tok = 0;
  for (char ch : uchars) {
    ch_to_token[ch] = tok;
    token_to_ch[tok] = ch;
    tok++;
  }
  ch_to_token['.'] = BOS;
  token_to_ch[BOS] = '.';

  int n_val = (int)(docs.size() / 10);
  int n_train = (int)docs.size() - n_val;
  vector<string> train_docs(docs.begin(), docs.begin() + n_train);
  vector<string> val_docs(docs.begin() + n_train, docs.end());

  Model model;
  model.cfg = cfg;
  model.init(vocab, g);
  cout << "config: layers=" << cfg.layers << " embd=" << cfg.embd
       << " heads=" << cfg.heads << " hidden=" << cfg.hidden
       << " mlp=" << cfg.mlp << " rope=" << cfg.rope << " tie=" << cfg.tie
       << " finalnorm=" << cfg.finalnorm << " residscale=" << cfg.residscale
       << " gains=" << cfg.gains << " batch=" << cfg.batch
       << " steps=" << cfg.steps << " lr=" << cfg.lr << " seed=" << cfg.seed
       << " params=" << model.NP << endl;

  auto tokenize = [&](const string &s) {
    vector<int> t = {BOS};
    for (char ch : s)
      t.push_back(ch_to_token[ch]);
    t.push_back(BOS);
    return t;
  };

  if (cfg.gradcheck) {
    if (cfg.gains) { // move gains off identity so their backward is exercised
      mt19937 gg(7);
      normal_distribution<float> dg(1.0f, 0.3f);
      for (int i = (cfg.gains ? model.o_gemb : model.NP); i < model.NP; i++)
        model.P[i] = dg(gg);
    }
    vector<int> tokens = tokenize(train_docs[0]);
    int n = min(cfg.block, (int)tokens.size() - 1);
    model.forward_doc(tokens, n);
    model.backward_doc(tokens, n);
    vector<float> Ga = model.G;
    // check the largest-magnitude gradients: float32 central differences are
    // too noisy to validate tiny ones
    vector<int> order(model.NP);
    for (int i = 0; i < model.NP; i++)
      order[i] = i;
    sort(order.begin(), order.end(),
         [&](int a, int b) { return fabs(Ga[a]) > fabs(Ga[b]); });
    float worst = 0;
    int checked = 0;
    vector<int> to_check;
    for (int rank = 0; rank < model.NP && (int)to_check.size() < 60; rank++)
      to_check.push_back(order[rank]);
    if (cfg.gains) { // top 20 gain grads, checked explicitly
      vector<int> gorder;
      for (int i = model.o_gemb; i < model.NP; i++)
        gorder.push_back(i);
      sort(gorder.begin(), gorder.end(),
           [&](int a, int b) { return fabs(Ga[a]) > fabs(Ga[b]); });
      for (int r = 0; r < 20 && r < (int)gorder.size(); r++)
        to_check.push_back(gorder[r]);
    }
    for (int idx : to_check) {
      float h = 2e-3f;
      float orig = model.P[idx];
      model.P[idx] = orig + h;
      float lp = model.forward_doc(tokens, n);
      model.P[idx] = orig - h;
      float lm = model.forward_doc(tokens, n);
      model.P[idx] = orig;
      float num = (lp - lm) / (2 * h);
      float rel = fabs(num - Ga[idx]) / max(1e-3f, fabs(num) + fabs(Ga[idx]));
      worst = max(worst, rel);
      checked++;
    }
    cout << "gradcheck: " << checked << " params, worst rel err = " << worst
         << endl;
    return worst < 0.05 ? 0 : 2;
  }

  float beta1 = 0.85f, beta2 = 0.99f, eps_adam = 1e-8f;
  auto eval_val = [&]() {
    double total = 0;
    long count = 0;
    for (auto &vd : val_docs) {
      vector<int> tokens = tokenize(vd);
      int n = min(cfg.block, (int)tokens.size() - 1);
      float l = model.forward_doc(tokens, n);
      total += (double)l * n;
      count += n;
    }
    return (float)(total / count);
  };

  double run_loss = 0;
  long run_count = 0;
  for (int step = 0; step < cfg.steps; step++) {
    for (int b = 0; b < cfg.batch; b++) {
      vector<int> tokens =
          tokenize(train_docs[((long)step * cfg.batch + b) % n_train]);
      int n = min(cfg.block, (int)tokens.size() - 1);
      float loss = model.forward_doc(tokens, n);
      model.backward_doc(tokens, n);
      run_loss += loss;
      run_count++;
    }
    if (cfg.batch > 1)
      for (int i = 0; i < model.NP; i++)
        model.G[i] /= cfg.batch;

    float lr_t = cfg.lr * (1 - (float)step / cfg.steps);
    float bc1 = 1 - pow(beta1, step + 1);
    float bc2 = 1 - pow(beta2, step + 1);
    for (int i = 0; i < model.NP; i++) {
      float grad = model.G[i];
      model.am[i] = beta1 * model.am[i] + (1 - beta1) * grad;
      model.av[i] = beta2 * model.av[i] + (1 - beta2) * grad * grad;
      model.P[i] -=
          lr_t * (model.am[i] / bc1) / (powf(model.av[i] / bc2, 0.5f) + eps_adam);
      model.G[i] = 0;
    }

    if ((step + 1) % cfg.eval_every == 0) {
      float vl = eval_val();
      cout << "step " << (step + 1) << " | train(avg) = " << setprecision(5)
           << run_loss / run_count << " | val = " << vl << endl;
      run_loss = 0;
      run_count = 0;
    }
  }
  float final_val = eval_val();

  // Sampling: 500 names for novelty stats, print first 20
  unordered_set<string> all_names(docs.begin(), docs.end());
  float temperature = 0.5f;
  int n_samples = 500, novel = 0;
  unordered_set<string> uniq;
  vector<string> first20;
  vector<float> tempered(vocab), sprobs(vocab);
  for (int s = 0; s < n_samples; s++) {
    int token_id = BOS;
    string sample = "";
    for (int pos = 0; pos < cfg.block; pos++) {
      model.forward_pos(pos, token_id);
      // recompute logits from stored final activations
      const float *head_in =
          cfg.finalnorm ? model.xfn[pos].data() : model.xf[pos].data();
      matvec(&model.P[model.o_lm], head_in, model.logits.data(), vocab,
             cfg.embd);
      for (int j = 0; j < vocab; j++)
        tempered[j] = model.logits[j] / temperature;
      softmax_fwd(tempered.data(), sprobs.data(), vocab);
      discrete_distribution<int> dist(sprobs.begin(), sprobs.end());
      token_id = dist(g);
      if (token_id == BOS)
        break;
      sample += token_to_ch[token_id];
    }
    if (!all_names.count(sample))
      novel++;
    uniq.insert(sample);
    if ((int)first20.size() < 20)
      first20.push_back(sample);
  }
  cout << endl << "--- samples ---" << endl;
  for (size_t i = 0; i < first20.size(); i++)
    cout << "Sample " << i << " : " << first20[i] << endl;
  cout << "novel = " << (100.0 * novel / n_samples)
       << "% unique = " << (100.0 * uniq.size() / n_samples) << "%" << endl;
  cout << "RESULT params=" << model.NP << " val_loss=" << setprecision(6)
       << final_val << " novel=" << (100.0 * novel / n_samples) << endl;
  return 0;
}
