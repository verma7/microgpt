// V3 = V2's fused ops + a flat tape ("arena") autograd:
//  - a node is an index into contiguous arrays (data/grad/children), not a
//    shared_ptr-managed object → zero per-node heap allocations
//  - creation order IS topological order, so backward() is a single reverse
//    sweep over the tape: no DFS, no visited set, no hashing
//  - each training step resets the tape to the post-init checkpoint and
//    reuses the same memory

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

using NodeId = uint32_t;

struct Tape {
  vector<float> data, grad;
  vector<uint32_t> cstart, ccount;
  vector<NodeId> cidx;
  vector<float> cgrad;

  NodeId alloc(float d, uint32_t nchildren) {
    data.push_back(d);
    grad.push_back(0.0f);
    cstart.push_back((uint32_t)cidx.size());
    ccount.push_back(nchildren);
    return (NodeId)(data.size() - 1);
  }
  void child(NodeId c, float g) {
    cidx.push_back(c);
    cgrad.push_back(g);
  }

  NodeId leaf(float d) { return alloc(d, 0); }
  NodeId add(NodeId a, NodeId b) {
    NodeId r = alloc(data[a] + data[b], 2);
    child(a, 1.0f);
    child(b, 1.0f);
    return r;
  }
  NodeId sub(NodeId a, NodeId b) {
    NodeId r = alloc(data[a] - data[b], 2);
    child(a, 1.0f);
    child(b, -1.0f);
    return r;
  }
  NodeId mul(NodeId a, NodeId b) {
    NodeId r = alloc(data[a] * data[b], 2);
    child(a, data[b]);
    child(b, data[a]);
    return r;
  }
  NodeId powc(NodeId a, float p) {
    NodeId r = alloc(std::pow(data[a], p), 1);
    child(a, p * std::pow(data[a], p - 1));
    return r;
  }
  NodeId logv(NodeId a) {
    NodeId r = alloc(std::log(data[a]), 1);
    child(a, 1.0f / data[a]);
    return r;
  }
  NodeId expv(NodeId a) {
    float e = std::exp(data[a]);
    NodeId r = alloc(e, 1);
    child(a, e);
    return r;
  }
  NodeId reluv(NodeId a) {
    NodeId r = alloc(std::max(data[a], 0.0f), 1);
    child(a, (float)(data[a] > 0));
    return r;
  }
  NodeId dot(const NodeId *w, const NodeId *x, int n) {
    float total = 0;
    for (int j = 0; j < n; j++)
      total += data[w[j]] * data[x[j]];
    NodeId r = alloc(total, 2 * n);
    for (int j = 0; j < n; j++) {
      child(w[j], data[x[j]]);
      child(x[j], data[w[j]]);
    }
    return r;
  }
  // dot of attention weights (contiguous ids) against a strided V column
  NodeId dot_col(const NodeId *w, const vector<vector<NodeId>> &V, int col,
                 int n) {
    float total = 0;
    for (int t = 0; t < n; t++)
      total += data[w[t]] * data[V[t][col]];
    NodeId r = alloc(total, 2 * n);
    for (int t = 0; t < n; t++) {
      child(w[t], data[V[t][col]]);
      child(V[t][col], data[w[t]]);
    }
    return r;
  }
  NodeId sum(const NodeId *xs, int n) {
    float total = 0;
    for (int i = 0; i < n; i++)
      total += data[xs[i]];
    NodeId r = alloc(total, n);
    for (int i = 0; i < n; i++)
      child(xs[i], 1.0f);
    return r;
  }
  NodeId sumsq(const NodeId *xs, int n) {
    float total = 0;
    for (int i = 0; i < n; i++)
      total += data[xs[i]] * data[xs[i]];
    NodeId r = alloc(total, n);
    for (int i = 0; i < n; i++)
      child(xs[i], 2 * data[xs[i]]);
    return r;
  }

  size_t mark_nodes() const { return data.size(); }
  size_t mark_children() const { return cidx.size(); }
  void reset(size_t n_nodes, size_t n_children) {
    data.resize(n_nodes);
    grad.resize(n_nodes);
    cstart.resize(n_nodes);
    ccount.resize(n_nodes);
    cidx.resize(n_children);
    cgrad.resize(n_children);
  }

  // Reverse sweep over the tape; nodes below `stop` are leaves (params).
  void backward(NodeId loss, NodeId stop) {
    grad[loss] = 1.0f;
    for (NodeId i = loss; i >= stop && i != (NodeId)-1; i--) {
      float gi = grad[i];
      if (gi == 0.0f)
        continue;
      uint32_t s = cstart[i], e = s + ccount[i];
      for (uint32_t c = s; c < e; c++)
        grad[cidx[c]] += cgrad[c] * gi;
    }
  }
};

Tape T;

using Mat = vector<vector<NodeId>>;

Mat matrix(int nout, int nin, normal_distribution<float> &d, mt19937 &g) {
  Mat output(nout);
  for (int i = 0; i < nout; i++) {
    output[i].reserve(nin);
    for (int j = 0; j < nin; j++)
      output[i].push_back(T.leaf(d(g)));
  }
  return output;
}

struct LayerW {
  Mat *wq, *wk, *wv, *wo, *fc1, *fc2;
};

vector<NodeId> linear(const vector<NodeId> &x, const Mat &w) {
  vector<NodeId> output;
  output.reserve(w.size());
  for (size_t i = 0; i < w.size(); i++)
    output.push_back(T.dot(w[i].data(), x.data(), (int)w[i].size()));
  return output;
}

vector<NodeId> softmax(const vector<NodeId> &logits) {
  float max_val = numeric_limits<float>::lowest();
  for (size_t i = 0; i < logits.size(); i++)
    max_val = max(max_val, T.data[logits[i]]);
  vector<NodeId> exps;
  exps.reserve(logits.size());
  NodeId max_value = T.leaf(max_val);
  for (size_t i = 0; i < logits.size(); i++)
    exps.push_back(T.expv(T.sub(logits[i], max_value)));
  NodeId inv_total = T.powc(T.sum(exps.data(), (int)exps.size()), -1);
  vector<NodeId> outputs;
  outputs.reserve(exps.size());
  for (size_t i = 0; i < exps.size(); i++)
    outputs.push_back(T.mul(exps[i], inv_total));
  return outputs;
}

vector<NodeId> rmsnorm(const vector<NodeId> &x) {
  NodeId ms = T.mul(T.sumsq(x.data(), (int)x.size()),
                    T.leaf(1.0f / (float)x.size()));
  NodeId scale = T.powc(T.add(ms, T.leaf(1e-5)), -0.5f);
  vector<NodeId> output;
  output.reserve(x.size());
  for (size_t i = 0; i < x.size(); i++)
    output.push_back(T.mul(x[i], scale));
  return output;
}

vector<NodeId> gpt(int token_id, int pos_id, vector<vector<vector<NodeId>>> *keys,
                   vector<vector<vector<NodeId>>> *values, Mat &wte, Mat &wpe,
                   Mat &lm_head, vector<LayerW> &layers, int n_layer,
                   int n_head, int head_dim, NodeId inv_sqrt_hd) {
  vector<NodeId> &tok_emb = wte[token_id];
  vector<NodeId> &pos_emb = wpe[pos_id];
  vector<NodeId> x;
  x.reserve(tok_emb.size());
  for (size_t i = 0; i < tok_emb.size(); i++)
    x.push_back(T.add(tok_emb[i], pos_emb[i]));
  x = rmsnorm(x);

  for (int li = 0; li < n_layer; li++) {
    // 1) Multi-head Attention block
    vector<NodeId> x_residual = x;
    x = rmsnorm(x);
    vector<NodeId> q = linear(x, *layers[li].wq);
    vector<NodeId> k = linear(x, *layers[li].wk);
    vector<NodeId> v = linear(x, *layers[li].wv);
    auto &K = (*keys)[li];
    auto &V = (*values)[li];
    K.push_back(std::move(k));
    V.push_back(std::move(v));
    int Tn = (int)K.size();
    vector<NodeId> x_attn;
    x_attn.reserve(n_head * head_dim);
    for (int h = 0; h < n_head; h++) {
      int hs = h * head_dim;
      vector<NodeId> attn_logits;
      attn_logits.reserve(Tn);
      for (int t = 0; t < Tn; t++)
        attn_logits.push_back(
            T.mul(T.dot(q.data() + hs, K[t].data() + hs, head_dim),
                  inv_sqrt_hd));
      vector<NodeId> attn_weights = softmax(attn_logits);
      for (int j = 0; j < head_dim; j++)
        x_attn.push_back(T.dot_col(attn_weights.data(), V, hs + j, Tn));
    }
    x = linear(x_attn, *layers[li].wo);
    for (size_t i = 0; i < x.size(); i++)
      x[i] = T.add(x[i], x_residual[i]);

    // 2) MLP block
    x_residual = x;
    x = rmsnorm(x);
    x = linear(x, *layers[li].fc1);
    for (size_t i = 0; i < x.size(); i++)
      x[i] = T.reluv(x[i]);
    x = linear(x, *layers[li].fc2);
    for (size_t i = 0; i < x.size(); i++)
      x[i] = T.add(x[i], x_residual[i]);
  }
  return linear(x, lm_head);
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
  int vocab_size = uchars.size() + 1;
  cout << "Vocab size: " << vocab_size << endl;

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

  int n_layer = 4;
  int n_embd = 16;
  int block_size = 16;
  int n_head = 4;
  int head_dim = n_embd / n_head;
  float std = 0.08;
  normal_distribution<float> d(0, std);
  unordered_map<string, Mat> state_dict;
  state_dict["wte"] = matrix(vocab_size, n_embd, d, g);
  state_dict["wpe"] = matrix(block_size, n_embd, d, g);
  state_dict["lm_head"] = matrix(vocab_size, n_embd, d, g);
  for (int i = 0; i < n_layer; i++) {
    state_dict["layer{" + to_string(i) + "}.attn_wq"] =
        matrix(n_embd, n_embd, d, g);
    state_dict["layer{" + to_string(i) + "}.attn_wk"] =
        matrix(n_embd, n_embd, d, g);
    state_dict["layer{" + to_string(i) + "}.attn_wv"] =
        matrix(n_embd, n_embd, d, g);
    state_dict["layer{" + to_string(i) + "}.attn_wo"] =
        matrix(n_embd, n_embd, d, g);
    state_dict["layer{" + to_string(i) + "}.mlp_fc1"] =
        matrix(4 * n_embd, n_embd, d, g);
    state_dict["layer{" + to_string(i) + "}.mlp_fc2"] =
        matrix(n_embd, 4 * n_embd, d, g);
  }
  Mat &wte = state_dict["wte"];
  Mat &wpe = state_dict["wpe"];
  Mat &lm_head = state_dict["lm_head"];
  vector<LayerW> layers(n_layer);
  for (int i = 0; i < n_layer; i++) {
    string p = "layer{" + to_string(i) + "}.";
    layers[i] = {&state_dict[p + "attn_wq"], &state_dict[p + "attn_wk"],
                 &state_dict[p + "attn_wv"], &state_dict[p + "attn_wo"],
                 &state_dict[p + "mlp_fc1"], &state_dict[p + "mlp_fc2"]};
  }

  vector<NodeId> params;
  for (auto &pair : state_dict) {
    auto &vv = pair.second;
    for (size_t i = 0; i < vv.size(); i++)
      for (size_t j = 0; j < vv[i].size(); j++)
        params.push_back(vv[i][j]);
  }
  cout << "Num params: " << params.size() << endl;

  float learning_rate = 0.01;
  float beta1 = 0.85;
  float beta2 = 0.99;
  float eps_adam = 1e-8;
  vector<float> m(params.size(), 0.0);
  vector<float> v(params.size(), 0.0);

  NodeId inv_sqrt_hd = T.leaf(1.0f / std::sqrt((float)head_dim));

  const size_t ckpt_nodes = T.mark_nodes();
  const size_t ckpt_children = T.mark_children();
  const NodeId first_dynamic = (NodeId)ckpt_nodes;

  int num_steps = 2000;
  for (int step = 0; step < num_steps; step++) {
    string doc = docs[step % docs.size()];
    vector<int> tokens = {BOS};
    for (char ch : doc)
      tokens.push_back(ch_to_token[ch]);
    tokens.push_back(BOS);
    int n = min({block_size, static_cast<int>(tokens.size()) - 1});

    T.reset(ckpt_nodes, ckpt_children);
    vector<vector<vector<NodeId>>> keys(n_layer);
    vector<vector<vector<NodeId>>> values(n_layer);

    vector<NodeId> losses;
    for (int pos_id = 0; pos_id < n; pos_id++) {
      int token_id = tokens[pos_id];
      int target_id = tokens[pos_id + 1];
      vector<NodeId> logits =
          gpt(token_id, pos_id, &keys, &values, wte, wpe, lm_head, layers,
              n_layer, n_head, head_dim, inv_sqrt_hd);
      vector<NodeId> probs = softmax(logits);
      NodeId neg1 = T.leaf(-1);
      losses.push_back(T.mul(neg1, T.logv(probs[target_id])));
    }
    NodeId loss =
        T.mul(T.leaf(1.0f / n), T.sum(losses.data(), (int)losses.size()));

    T.backward(loss, first_dynamic - 0); // params below stay leaves

    float lr_t = learning_rate * (1 - static_cast<float>(step) / num_steps);
    for (size_t i = 0; i < params.size(); i++) {
      float grad = T.grad[params[i]];
      m[i] = beta1 * m[i] + (1 - beta1) * grad;
      v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
      float m_hat = m[i] / (1 - pow(beta1, step + 1));
      float v_hat = v[i] / (1 - pow(beta2, step + 1));
      T.data[params[i]] -= lr_t * m_hat / (pow(v_hat, 0.5) + eps_adam);
      T.grad[params[i]] = 0;
    }

    if ((step + 1) % 100 == 0)
      cout << "Step " << (step + 1) << " / " << num_steps
           << " | loss = " << setprecision(6) << T.data[loss] << endl;
  }

  float temperature = 0.5;
  cout << endl << "--- inference (new, hallucinated names) ---" << endl;
  for (int sample_idx = 0; sample_idx < 20; sample_idx++) {
    T.reset(ckpt_nodes, ckpt_children);
    vector<vector<vector<NodeId>>> keys(n_layer);
    vector<vector<vector<NodeId>>> values(n_layer);
    int token_id = BOS;
    string sample = "";
    for (int pos_id = 0; pos_id < block_size; pos_id++) {
      vector<NodeId> logits =
          gpt(token_id, pos_id, &keys, &values, wte, wpe, lm_head, layers,
              n_layer, n_head, head_dim, inv_sqrt_hd);
      vector<NodeId> tempered_logits;
      tempered_logits.reserve(logits.size());
      NodeId inv_temp = T.leaf(1.0f / temperature);
      for (auto &val : logits)
        tempered_logits.push_back(T.mul(val, inv_temp));
      vector<NodeId> probs = softmax(tempered_logits);
      vector<float> weights;
      weights.reserve(probs.size());
      for (auto &prob : probs)
        weights.push_back(T.data[prob]);
      std::discrete_distribution<int> distribution(weights.begin(),
                                                   weights.end());
      token_id = distribution(g);
      if (token_id == BOS)
        break;
      sample += token_to_ch[token_id];
    }
    cout << "Sample " << sample_idx << " : " << sample << endl;
  }
  return 0;
}
