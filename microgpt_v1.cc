// V1: same autograd granularity as baseline microgpt.cc, but with
// engineering hygiene:
//  - iterative topo sort using epoch-marked visited flags (no unordered_set
//    of shared_ptr, no recursion)
//  - weight matrices resolved once (no per-call string building + hashing)
//  - vectors reserved, K/V head slices indexed in place (no copies)
//  - constant nodes hoisted out of inner loops

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

static uint32_t g_epoch = 0;

class Value : public std::enable_shared_from_this<Value> {
public:
  float data;
  float grad;
  uint32_t visit_epoch = 0;
  using Ptr = std::shared_ptr<Value>;

  static Ptr make(float data, std::vector<Ptr> children = {},
                  std::vector<float> grads = {}) {
    return std::make_shared<Value>(data, std::move(children), std::move(grads));
  }

  Value(float data, vector<Ptr> children = {}, vector<float> local_grads = {})
      : data(data), grad(0), children_(std::move(children)),
        local_grads_(std::move(local_grads)) {}

  friend Ptr operator+(const Ptr &lhs, const Ptr &rhs) {
    return make(lhs->data + rhs->data, {lhs, rhs}, {1.0f, 1.0f});
  }
  friend Ptr operator-(const Ptr &lhs, const Ptr &rhs) {
    return make(lhs->data - rhs->data, {lhs, rhs}, {1.0f, -1.0f});
  }
  friend Ptr operator*(const Ptr &lhs, const Ptr &rhs) {
    return make(lhs->data * rhs->data, {lhs, rhs}, {rhs->data, lhs->data});
  }
  static Ptr pow(const Ptr &lhs, float other) {
    return make(std::pow(lhs->data, other), {lhs},
                {other * std::pow(lhs->data, other - 1)});
  }
  friend Ptr operator/(const Ptr &lhs, const Ptr &rhs) {
    return lhs * pow(rhs, -1);
  }
  static Ptr log(const Ptr &p) {
    return make(std::log(p->data), {p}, {1.0f / p->data});
  }
  static Ptr exp(const Ptr &p) {
    return make(std::exp(p->data), {p}, {std::exp(p->data)});
  }
  static Ptr relu(const Ptr &p) {
    return make(std::max({p->data, 0.0f}), {p},
                {static_cast<float>(p->data > 0)});
  }

  void backward() {
    ++g_epoch;
    vector<Value *> topo;
    vector<pair<Value *, size_t>> stack;
    visit_epoch = g_epoch;
    stack.push_back({this, 0});
    while (!stack.empty()) {
      auto &top = stack.back();
      Value *node = top.first;
      if (top.second < node->children_.size()) {
        Value *child = node->children_[top.second++].get();
        if (child->visit_epoch != g_epoch) {
          child->visit_epoch = g_epoch;
          stack.push_back({child, 0});
        }
      } else {
        topo.push_back(node);
        stack.pop_back();
      }
    }
    grad = 1;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      Value *v = *it;
      for (size_t i = 0; i < v->children_.size(); i++)
        v->children_[i]->grad += v->local_grads_[i] * v->grad;
    }
  }

private:
  vector<Ptr> children_;
  vector<float> local_grads_;
};

using Mat = vector<vector<Value::Ptr>>;

Mat matrix(int nout, int nin, normal_distribution<float> &d, mt19937 &g) {
  Mat output(nout);
  for (int i = 0; i < nout; i++) {
    output[i].reserve(nin);
    for (int j = 0; j < nin; j++)
      output[i].push_back(Value::make(d(g)));
  }
  return output;
}

struct LayerW {
  Mat *wq, *wk, *wv, *wo, *fc1, *fc2;
};

vector<Value::Ptr> linear(const vector<Value::Ptr> &x, const Mat &w) {
  vector<Value::Ptr> output;
  output.reserve(w.size());
  for (size_t i = 0; i < w.size(); i++) {
    Value::Ptr total = Value::make(0);
    for (size_t j = 0; j < w[i].size(); j++)
      total = total + w[i][j] * x[j];
    output.push_back(total);
  }
  return output;
}

vector<Value::Ptr> softmax(const vector<Value::Ptr> &logits) {
  float max_val = numeric_limits<float>::lowest();
  for (size_t i = 0; i < logits.size(); i++)
    max_val = max(max_val, logits[i]->data);
  vector<Value::Ptr> exps;
  exps.reserve(logits.size());
  Value::Ptr total = Value::make(0);
  Value::Ptr max_value = Value::make(max_val);
  for (size_t i = 0; i < logits.size(); i++) {
    exps.push_back(Value::exp(logits[i] - max_value));
    total = total + exps[i];
  }
  vector<Value::Ptr> outputs;
  outputs.reserve(exps.size());
  for (size_t i = 0; i < exps.size(); i++)
    outputs.push_back(exps[i] / total);
  return outputs;
}

vector<Value::Ptr> rmsnorm(const vector<Value::Ptr> &x) {
  Value::Ptr ms = Value::make(0);
  for (size_t i = 0; i < x.size(); i++)
    ms = ms + x[i] * x[i];
  ms = ms / Value::make(x.size());
  Value::Ptr scale = Value::pow(ms + Value::make(1e-5), -0.5);
  vector<Value::Ptr> output;
  output.reserve(x.size());
  for (size_t i = 0; i < x.size(); i++)
    output.push_back(x[i] * scale);
  return output;
}

vector<Value::Ptr> gpt(int token_id, int pos_id,
                       vector<vector<vector<Value::Ptr>>> *keys,
                       vector<vector<vector<Value::Ptr>>> *values, Mat &wte,
                       Mat &wpe, Mat &lm_head, vector<LayerW> &layers,
                       int n_layer, int n_head, int head_dim,
                       const Value::Ptr &inv_sqrt_hd) {
  vector<Value::Ptr> &tok_emb = wte[token_id];
  vector<Value::Ptr> &pos_emb = wpe[pos_id];
  vector<Value::Ptr> x;
  x.reserve(tok_emb.size());
  for (size_t i = 0; i < tok_emb.size(); i++)
    x.push_back(tok_emb[i] + pos_emb[i]);
  x = rmsnorm(x);

  for (int li = 0; li < n_layer; li++) {
    // 1) Multi-head Attention block
    vector<Value::Ptr> x_residual = x;
    x = rmsnorm(x);
    vector<Value::Ptr> q = linear(x, *layers[li].wq);
    vector<Value::Ptr> k = linear(x, *layers[li].wk);
    vector<Value::Ptr> v = linear(x, *layers[li].wv);
    auto &K = (*keys)[li];
    auto &V = (*values)[li];
    K.push_back(std::move(k));
    V.push_back(std::move(v));
    int T = K.size();
    vector<Value::Ptr> x_attn;
    x_attn.reserve(n_head * head_dim);
    for (int h = 0; h < n_head; h++) {
      int hs = h * head_dim;
      vector<Value::Ptr> attn_logits;
      attn_logits.reserve(T);
      for (int t = 0; t < T; t++) {
        Value::Ptr total = Value::make(0);
        for (int j = 0; j < head_dim; j++)
          total = total + q[hs + j] * K[t][hs + j];
        attn_logits.push_back(total * inv_sqrt_hd);
      }
      vector<Value::Ptr> attn_weights = softmax(attn_logits);
      for (int j = 0; j < head_dim; j++) {
        Value::Ptr total = Value::make(0);
        for (int t = 0; t < T; t++)
          total = total + attn_weights[t] * V[t][hs + j];
        x_attn.push_back(total);
      }
    }
    x = linear(x_attn, *layers[li].wo);
    for (size_t i = 0; i < x.size(); i++)
      x[i] = x[i] + x_residual[i];

    // 2) MLP block
    x_residual = x;
    x = rmsnorm(x);
    x = linear(x, *layers[li].fc1);
    for (size_t i = 0; i < x.size(); i++)
      x[i] = Value::relu(x[i]);
    x = linear(x, *layers[li].fc2);
    for (size_t i = 0; i < x.size(); i++)
      x[i] = x[i] + x_residual[i];
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
  // Resolve weight lookups once instead of per token position.
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

  vector<Value::Ptr> params;
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

  Value::Ptr inv_sqrt_hd = Value::make(1.0f / std::sqrt((float)head_dim));

  int num_steps = 2000;
  for (int step = 0; step < num_steps; step++) {
    string doc = docs[step % docs.size()];
    vector<int> tokens = {BOS};
    for (char ch : doc)
      tokens.push_back(ch_to_token[ch]);
    tokens.push_back(BOS);
    int n = min({block_size, static_cast<int>(tokens.size()) - 1});

    vector<vector<vector<Value::Ptr>>> keys(n_layer);
    vector<vector<vector<Value::Ptr>>> values(n_layer);

    Value::Ptr sum_losses = Value::make(0);
    for (int pos_id = 0; pos_id < n; pos_id++) {
      int token_id = tokens[pos_id];
      int target_id = tokens[pos_id + 1];
      vector<Value::Ptr> logits =
          gpt(token_id, pos_id, &keys, &values, wte, wpe, lm_head, layers,
              n_layer, n_head, head_dim, inv_sqrt_hd);
      vector<Value::Ptr> probs = softmax(logits);
      Value::Ptr neg1 = Value::make(-1);
      Value::Ptr loss_t = neg1 * Value::log(probs[target_id]);
      sum_losses = sum_losses + loss_t;
    }
    Value::Ptr loss = Value::make(1.0 / n) * sum_losses;

    loss->backward();

    float lr_t = learning_rate * (1 - static_cast<float>(step) / num_steps);
    for (size_t i = 0; i < params.size(); i++) {
      float grad = params[i]->grad;
      m[i] = beta1 * m[i] + (1 - beta1) * grad;
      v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
      float m_hat = m[i] / (1 - pow(beta1, step + 1));
      float v_hat = v[i] / (1 - pow(beta2, step + 1));
      params[i]->data -= lr_t * m_hat / (pow(v_hat, 0.5) + eps_adam);
      params[i]->grad = 0;
    }

    if ((step + 1) % 100 == 0)
      cout << "Step " << (step + 1) << " / " << num_steps
           << " | loss = " << setprecision(6) << loss->data << endl;
  }

  float temperature = 0.5;
  cout << endl << "--- inference (new, hallucinated names) ---" << endl;
  for (int sample_idx = 0; sample_idx < 20; sample_idx++) {
    vector<vector<vector<Value::Ptr>>> keys(n_layer);
    vector<vector<vector<Value::Ptr>>> values(n_layer);
    int token_id = BOS;
    string sample = "";
    for (int pos_id = 0; pos_id < block_size; pos_id++) {
      vector<Value::Ptr> logits =
          gpt(token_id, pos_id, &keys, &values, wte, wpe, lm_head, layers,
              n_layer, n_head, head_dim, inv_sqrt_hd);
      vector<Value::Ptr> tempered_logits;
      tempered_logits.reserve(logits.size());
      Value::Ptr temp = Value::make(temperature);
      for (auto &val : logits)
        tempered_logits.push_back(val / temp);
      vector<Value::Ptr> probs = softmax(tempered_logits);
      vector<float> weights;
      weights.reserve(probs.size());
      for (auto &prob : probs)
        weights.push_back(prob->data);
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
