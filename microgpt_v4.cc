// V4: no autograd graph at all. Forward pass stores activations in flat
// float arrays; the backward pass is hand-derived (closed-form gradients for
// linear, rmsnorm, softmax+cross-entropy, and causal attention with a KV
// cache). Same architecture, same init RNG stream, same Adam — the model is
// mathematically identical to the baseline; only float summation order
// differs.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

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
// dx += W^T dy
static void matvecT_acc(const float *W, const float *dy, float *dx, int rows,
                        int cols) {
  for (int i = 0; i < rows; i++) {
    float d = dy[i];
    const float *w = W + i * cols;
    for (int j = 0; j < cols; j++)
      dx[j] += w[j] * d;
  }
}
// dW += dy x^T
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
// y = x*s with s=(mean(x^2)+eps)^-1/2  =>  dx_j += s*dy_j - s^3 x_j (dy.x)/n
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

int main() {
  vector<string> docs;
  string doc;
  while (cin >> doc)
    docs.push_back(doc);

  mt19937 g(42);
  shuffle(docs.begin(), docs.end(), g);
  cout << "Num docs: " << docs.size() << endl;

  set<char> uchars;
  for (const auto &doc : docs)
    for (char ch : doc)
      uchars.insert(ch);
  int BOS = uchars.size();
  int vocab = uchars.size() + 1;
  cout << "Vocab size: " << vocab << endl;

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

  const int L = 4;  // layers
  const int E = 16; // embedding dim
  const int BS = 16; // block size
  const int H = 4;  // heads
  const int D = E / H;
  const int M = 4 * E; // mlp hidden
  const float inv_sqrt_d = 1.0f / sqrtf((float)D);

  // Flat parameter buffer; offsets per matrix. Same creation (RNG) order as
  // the baseline: wte, wpe, lm_head, then per layer wq,wk,wv,wo,fc1,fc2.
  int off = 0;
  int o_wte = off; off += vocab * E;
  int o_wpe = off; off += BS * E;
  int o_lm = off; off += vocab * E;
  int o_wq[L], o_wk[L], o_wv[L], o_wo[L], o_fc1[L], o_fc2[L];
  for (int l = 0; l < L; l++) {
    o_wq[l] = off; off += E * E;
    o_wk[l] = off; off += E * E;
    o_wv[l] = off; off += E * E;
    o_wo[l] = off; off += E * E;
    o_fc1[l] = off; off += M * E;
    o_fc2[l] = off; off += E * M;
  }
  const int NP = off;
  vector<float> P(NP), G(NP, 0.0f), am(NP, 0.0f), av(NP, 0.0f);
  float std_init = 0.08;
  normal_distribution<float> d(0, std_init);
  for (int i = 0; i < NP; i++)
    P[i] = d(g); // contiguous layout matches the baseline's draw order
  cout << "Num params: " << NP << endl;

  float learning_rate = 0.01;
  float beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;

  // Activation storage for one document (n <= BS positions)
  auto f2 = [&](int a, int b) { return vector<vector<float>>(a, vector<float>(b)); };
  auto x0pre = f2(BS, E);
  vector<float> s0(BS);
  auto x0 = f2(BS, E);
  vector<vector<vector<float>>> ain(L, f2(BS, E)), anorm(L, f2(BS, E)),
      q(L, f2(BS, E)), K(L, f2(BS, E)), V(L, f2(BS, E)),
      attw(L, f2(BS, H * BS)), attnout(L, f2(BS, E)), min_(L, f2(BS, E)),
      mnorm(L, f2(BS, E)), h1pre(L, f2(BS, M)), h1(L, f2(BS, M));
  vector<vector<float>> sattn(L, vector<float>(BS)), smlp(L, vector<float>(BS));
  auto xf = f2(BS, E);
  auto probs = f2(BS, vocab);
  vector<vector<vector<float>>> dK(L, f2(BS, E)), dV(L, f2(BS, E));
  vector<float> xbuf(E), tmp(E), logits(vocab), dlog(vocab), dx(E), dmin(E),
      dattn(E), dain(E), danorm(E), dq_(E), dx0pre(E), dh1(M), dmnorm(E);

  // Forward one position with KV cache; stores activations for backward.
  auto forward_pos = [&](int t, int token_id) {
    for (int i = 0; i < E; i++)
      x0pre[t][i] = P[o_wte + token_id * E + i] + P[o_wpe + t * E + i];
    rms_fwd(x0pre[t].data(), x0[t].data(), &s0[t], E);
    xbuf = x0[t];
    for (int l = 0; l < L; l++) {
      ain[l][t] = xbuf;
      rms_fwd(xbuf.data(), anorm[l][t].data(), &sattn[l][t], E);
      matvec(&P[o_wq[l]], anorm[l][t].data(), q[l][t].data(), E, E);
      matvec(&P[o_wk[l]], anorm[l][t].data(), K[l][t].data(), E, E);
      matvec(&P[o_wv[l]], anorm[l][t].data(), V[l][t].data(), E, E);
      for (int h = 0; h < H; h++) {
        int hs = h * D;
        float lg[BS];
        for (int u = 0; u <= t; u++) {
          float s = 0;
          for (int j = 0; j < D; j++)
            s += q[l][t][hs + j] * K[l][u][hs + j];
          lg[u] = s * inv_sqrt_d;
        }
        softmax_fwd(lg, &attw[l][t][h * BS], t + 1);
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
      matvec(&P[o_fc1[l]], mnorm[l][t].data(), h1pre[l][t].data(), M, E);
      for (int i = 0; i < M; i++)
        h1[l][t][i] = max(h1pre[l][t][i], 0.0f);
      matvec(&P[o_fc2[l]], h1[l][t].data(), tmp.data(), E, M);
      for (int i = 0; i < E; i++)
        xbuf[i] = tmp[i] + min_[l][t][i];
    }
    xf[t] = xbuf;
    matvec(&P[o_lm], xbuf.data(), logits.data(), vocab, E);
    softmax_fwd(logits.data(), probs[t].data(), vocab);
  };

  int num_steps = 2000;
  for (int step = 0; step < num_steps; step++) {
    string doc = docs[step % docs.size()];
    vector<int> tokens = {BOS};
    for (char ch : doc)
      tokens.push_back(ch_to_token[ch]);
    tokens.push_back(BOS);
    int n = min({BS, static_cast<int>(tokens.size()) - 1});

    // ----- forward -----
    float loss = 0;
    for (int t = 0; t < n; t++) {
      forward_pos(t, tokens[t]);
      loss += -logf(probs[t][tokens[t + 1]]);
    }
    loss /= n;

    // ----- backward -----
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
      matvecT_acc(&P[o_lm], dlog.data(), dx.data(), vocab, E);
      outer_acc(&G[o_lm], dlog.data(), xf[t].data(), vocab, E);

      for (int l = L - 1; l >= 0; l--) {
        // MLP block: x_out = fc2·relu(fc1·rmsnorm(m_in)) + m_in
        fill(dh1.begin(), dh1.end(), 0.0f);
        matvecT_acc(&P[o_fc2[l]], dx.data(), dh1.data(), E, M);
        outer_acc(&G[o_fc2[l]], dx.data(), h1[l][t].data(), E, M);
        dmin = dx; // residual path
        for (int i = 0; i < M; i++)
          if (h1pre[l][t][i] <= 0)
            dh1[i] = 0;
        fill(dmnorm.begin(), dmnorm.end(), 0.0f);
        matvecT_acc(&P[o_fc1[l]], dh1.data(), dmnorm.data(), M, E);
        outer_acc(&G[o_fc1[l]], dh1.data(), mnorm[l][t].data(), M, E);
        rms_bwd(min_[l][t].data(), smlp[l][t], dmnorm.data(), dmin.data(), E);

        // Attention block: x_mid = wo·attnout + a_in
        fill(dattn.begin(), dattn.end(), 0.0f);
        matvecT_acc(&P[o_wo[l]], dmin.data(), dattn.data(), E, E);
        outer_acc(&G[o_wo[l]], dmin.data(), attnout[l][t].data(), E, E);
        dain = dmin; // residual path
        fill(dq_.begin(), dq_.end(), 0.0f);
        for (int h = 0; h < H; h++) {
          int hs = h * D;
          const float *aw = &attw[l][t][h * BS];
          float da[BS];
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
            float dlg = aw[u] * (da[u] - aw_da); // softmax backward
            for (int j = 0; j < D; j++) {
              dq_[hs + j] += dlg * K[l][u][hs + j] * inv_sqrt_d;
              dK[l][u][hs + j] += dlg * q[l][t][hs + j] * inv_sqrt_d;
            }
          }
        }
        // q/k/v projections at position t (dK[l][t], dV[l][t] complete now
        // because later positions were already processed)
        fill(danorm.begin(), danorm.end(), 0.0f);
        matvecT_acc(&P[o_wq[l]], dq_.data(), danorm.data(), E, E);
        outer_acc(&G[o_wq[l]], dq_.data(), anorm[l][t].data(), E, E);
        matvecT_acc(&P[o_wk[l]], dK[l][t].data(), danorm.data(), E, E);
        outer_acc(&G[o_wk[l]], dK[l][t].data(), anorm[l][t].data(), E, E);
        matvecT_acc(&P[o_wv[l]], dV[l][t].data(), danorm.data(), E, E);
        outer_acc(&G[o_wv[l]], dV[l][t].data(), anorm[l][t].data(), E, E);
        rms_bwd(ain[l][t].data(), sattn[l][t], danorm.data(), dain.data(), E);
        dx = dain;
      }
      fill(dx0pre.begin(), dx0pre.end(), 0.0f);
      rms_bwd(x0pre[t].data(), s0[t], dx.data(), dx0pre.data(), E);
      for (int i = 0; i < E; i++) {
        G[o_wte + tokens[t] * E + i] += dx0pre[i];
        G[o_wpe + t * E + i] += dx0pre[i];
      }
    }

    // ----- Adam -----
    float lr_t = learning_rate * (1 - static_cast<float>(step) / num_steps);
    float bc1 = 1 - pow(beta1, step + 1);
    float bc2 = 1 - pow(beta2, step + 1);
    for (int i = 0; i < NP; i++) {
      float grad = G[i];
      am[i] = beta1 * am[i] + (1 - beta1) * grad;
      av[i] = beta2 * av[i] + (1 - beta2) * grad * grad;
      float m_hat = am[i] / bc1;
      float v_hat = av[i] / bc2;
      P[i] -= lr_t * m_hat / (powf(v_hat, 0.5f) + eps_adam);
      G[i] = 0;
    }

    if ((step + 1) % 100 == 0)
      cout << "Step " << (step + 1) << " / " << num_steps
           << " | loss = " << setprecision(6) << loss << endl;
  }

  float temperature = 0.5;
  cout << endl << "--- inference (new, hallucinated names) ---" << endl;
  for (int sample_idx = 0; sample_idx < 20; sample_idx++) {
    int token_id = BOS;
    string sample = "";
    vector<float> tempered(vocab), sprobs(vocab);
    for (int pos_id = 0; pos_id < BS; pos_id++) {
      forward_pos(pos_id, token_id); // reuses K/V rows [0..pos_id] as cache
      matvec(&P[o_lm], xf[pos_id].data(), logits.data(), vocab, E);
      float inv_temp = 1.0f / temperature;
      for (int j = 0; j < vocab; j++)
        tempered[j] = logits[j] * inv_temp;
      softmax_fwd(tempered.data(), sprobs.data(), vocab);
      std::discrete_distribution<int> distribution(sprobs.begin(),
                                                   sprobs.end());
      token_id = distribution(g);
      if (token_id == BOS)
        break;
      sample += token_to_ch[token_id];
    }
    cout << "Sample " << sample_idx << " : " << sample << endl;
  }
  return 0;
}
