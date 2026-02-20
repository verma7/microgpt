// C++ implementation inspired by Andrej Karpathy's MicroGPT: https://karpathy.github.io/2026/02/12/microgpt/
//
// @verma7

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

// Let there be Autograd to recursively apply the chain rule through a computation graph
class Value : public std::enable_shared_from_this<Value> {
  public:
	float data;
	float grad;
	using Ptr = std::shared_ptr<Value>;

	// Factory method for elegance: Value::make(0.5)
    	static Ptr make(float data, std::vector<Ptr> children = {}, std::vector<float> grads = {}) {
        	return std::make_shared<Value>(data, children, grads);
    	}

	Value(float data, const vector<Ptr>& children = {}, const vector<float>& local_grads = {}):
		data(data),
		grad(0),
		children_(children.begin(), children.end()),
		local_grads_(local_grads.begin(), local_grads.end()) {}

	friend Ptr operator+(const Ptr& lhs, const Ptr& rhs) {
		return make(lhs->data + rhs->data, {lhs, rhs}, {1.0f, 1.0f});
    	}
	friend Ptr operator-(const Ptr& lhs, const Ptr& rhs) {
		return make(lhs->data - rhs->data, {lhs, rhs}, {1.0f, -1.0f});
    	}
	friend Ptr operator*(const Ptr& lhs, const Ptr& rhs) {
		return make(lhs->data * rhs->data, {lhs, rhs}, {rhs->data, lhs->data});
    	}
	static Ptr pow(const Ptr& lhs, float other) {
		return make(std::pow(lhs->data, other), {lhs}, {other * std::pow(lhs->data, other - 1)});
	}
	friend Ptr operator/(const Ptr& lhs, const Ptr& rhs) {
        	return lhs * pow(rhs, -1);
    	}
	static Ptr log(const Ptr& p) {
		return make(std::log(p->data), {p}, {1.0f/p->data});
	}
	static Ptr exp(const Ptr& p) {
		return make(std::exp(p->data), {p}, {std::exp(p->data)});
	}
	static Ptr relu(const Ptr& p) {
		return make(std::max({p->data, 0.0f}), {p}, {static_cast<float>(p->data > 0)});
	}

	void backward() {
		vector<Ptr> topo;
		unordered_set<Ptr> visited;
		build_topo(shared_from_this(), &topo, &visited);
		reverse(topo.begin(), topo.end());
		grad = 1;
		for (auto& value : topo) for (int i = 0; i < value->children_.size(); i++) value->children_[i]->grad += value->local_grads_[i] * value->grad;
	}

	friend std::ostream& operator<<(std::ostream& os, const Ptr& p) {
	    os << "[data = " << p->data << ", grad = " << p->grad << "]";
	    return os;
	}

  private:
	vector<Ptr> children_;
	vector<float> local_grads_;

	void build_topo(Ptr node, vector<Ptr>* topo, unordered_set<Ptr>* visited) {
		if (visited->find(node) == visited->end()) {
			visited->insert(node);
			for (Ptr child : node->children_) build_topo(child, topo, visited);
			topo->push_back(node);
		}
	}
};

vector<vector<Value::Ptr>> matrix(int nout, int nin, normal_distribution<float>& d, mt19937& g) {
	vector<vector<Value::Ptr>> output;
	for (int i = 0; i < nout; i++) {
		vector<Value::Ptr> row;
		for (int j = 0; j < nin; j++) row.push_back(Value::make(d(g)));
		output.push_back(row);
	}
	return output;
}

// Define the model architecture: a function mapping tokens and parameters to logits over what comes next
// Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU

vector<Value::Ptr> linear(vector<Value::Ptr>& x, vector<vector<Value::Ptr>>& w) {
	vector<Value::Ptr> output;
	for (int i = 0; i < w.size(); i++) {
		Value::Ptr total = Value::make(0);
		for (int j = 0; j < w[i].size(); j++) total = total + w[i][j] * x[j];
		output.push_back(total);
	}
	return output;
}

vector<Value::Ptr> softmax(vector<Value::Ptr>& logits) {
	float max_val = numeric_limits<float>::min();
	for (int i = 0; i < logits.size(); i++) max_val = max(max_val, logits[i]->data);
	vector<Value::Ptr> exps;
	Value::Ptr total = Value::make(0);	
	Value::Ptr max_value = Value::make(max_val);
	for (int i = 0; i < logits.size(); i++) {
		exps.push_back(Value::exp(logits[i] - max_value));
		total = total + exps[i];
	}
	vector<Value::Ptr> outputs;
	for (int i = 0; i < exps.size(); i++) outputs.push_back(exps[i] / total);
	return outputs;
}

vector<Value::Ptr> rmsnorm(vector<Value::Ptr>& x) {
	Value::Ptr ms = Value::make(0);
	for (int i = 0; i < x.size(); i++) ms = ms + x[i] * x[i];
	ms = ms / Value::make(x.size());
	Value::Ptr scale = Value::pow(ms + Value::make(1e-5), -0.5);
	vector<Value::Ptr> output;
	for (int i = 0; i < x.size(); i++) output.push_back(x[i] * scale);
	return output;
}

vector<Value::Ptr> gpt(int token_id, int pos_id, vector<vector<vector<Value::Ptr>>>* keys, vector<vector<vector<Value::Ptr>>>* values,
	unordered_map<string, vector<vector<Value::Ptr>>>* state_dict, int n_layer, int n_head, int head_dim) {
	vector<Value::Ptr> tok_emb = (*state_dict)["wte"][token_id]; // token embedding
	vector<Value::Ptr> pos_emb = (*state_dict)["wpe"][pos_id];   // position embedding
	vector<Value::Ptr> x;
	for (int i = 0; i < tok_emb.size(); i++) x.push_back(tok_emb[i] + pos_emb[i]); // joint token and position embedding
	x = rmsnorm(x); // note: not redundant due to backward pass via the residual connection


	for (int li = 0; li < n_layer; li++) {
		// 1) Multi-head Attention block
		vector<Value::Ptr> x_residual = x;
		x = rmsnorm(x);
		vector<Value::Ptr> q = linear(x, (*state_dict)["layer{" + to_string(li) + "}.attn_wq"]); 
		vector<Value::Ptr> k = linear(x, (*state_dict)["layer{" + to_string(li) + "}.attn_wk"]); 
		vector<Value::Ptr> v = linear(x, (*state_dict)["layer{" + to_string(li) + "}.attn_wv"]); 
		(*keys)[li].push_back(k);
		(*values)[li].push_back(v);
		vector<Value::Ptr> x_attn;
		for (int h = 0; h < n_head; h++) {
			int hs = h * head_dim;
			vector<Value::Ptr> q_h;
			for (int hi = hs; hi < hs + head_dim; hi++) q_h.push_back(q[hi]);
			vector<vector<Value::Ptr>> k_h;
			for (auto ki : (*keys)[li]) {
				vector<Value::Ptr> kiv;
				for (int hi = hs; hi < hs + head_dim; hi++) kiv.push_back(ki[hi]);
				k_h.push_back(kiv);
			}
			vector<vector<Value::Ptr>> v_h;
			for (auto vi : (*values)[li]) {
				vector<Value::Ptr> viv;
				for (int hi = hs; hi < hs + head_dim; hi++) viv.push_back(vi[hi]);
				v_h.push_back(viv);
			}
			vector<Value::Ptr> attn_logits;
			for (int t = 0; t < k_h.size(); t++) {
				Value::Ptr total = Value::make(0);
				for (int j = 0; j < head_dim; j++) total = total + q_h[j] * k_h[t][j];
				Value::Ptr head_dim_val = Value::make(head_dim);
				Value::Ptr sqrt_head_dim = Value::pow(head_dim_val, 0.5);
				total = total / sqrt_head_dim;
				attn_logits.push_back(total);
			}
			vector<Value::Ptr> attn_weights = softmax(attn_logits);
			for (int j = 0; j < head_dim; j++) {
				Value::Ptr total = Value::make(0);
				for (int t = 0; t < v_h.size(); t++) {
					total = total + attn_weights[t] * v_h[t][j];
				}
				x_attn.push_back(total);
			}
		}
		x = linear(x_attn, (*state_dict)["layer{" + to_string(li) + "}.attn_wo"]);
		for (int i = 0; i < x.size(); i++) x[i] = x[i] + x_residual[i];

		// 2) MLP block
		x_residual = x;
		x = rmsnorm(x);
		
		x = linear(x, (*state_dict)["layer{" + to_string(li) + "}.mlp_fc1"]); 
		for (int i = 0; i < x.size(); i++) x[i] = Value::relu(x[i]);

		x = linear(x, (*state_dict)["layer{" + to_string(li) + "}.mlp_fc2"]); 
		for (int i = 0; i < x.size(); i++) x[i] = x[i] + x_residual[i];
	}
	vector<Value::Ptr> logits = linear(x, (*state_dict)["lm_head"]);
	return logits;
}

int main() {
	// Let there be a Dataset `docs` of documents (e.g. a list of names)
	vector<string> docs;
	string doc;
	while (cin >> doc) docs.push_back(doc);

	mt19937 g(42);  // Let there be order among chaos
	// mt19937 g(std::random_device{}());
	shuffle(docs.begin(), docs.end(), g);
	cout << "Num docs: " << docs.size() << endl;

	// Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
	set<char> uchars;
	for (const auto& doc : docs) for (char ch : doc) uchars.insert(ch);
	int BOS = uchars.size();
	int vocab_size = uchars.size() + 1;
	cout << "Vocab size: " << vocab_size << endl;

	// Char to Token mapping and vice-versa
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

	// Initialize the parameters, to store the knowledge of the model
	int n_layer = 4;     // depth of the transformer neural network (number of layers)
	int n_embd = 16;     // width of the network (embedding dimension)
	int block_size = 16; // maximum context length of the attention window (note: the longest name is 15 characters)
	int n_head = 4;      // number of attention heads
	int head_dim = n_embd / n_head; // derived dimension of each head
	float std = 0.08;
	normal_distribution<float> d(0, std);
	unordered_map<string, vector<vector<Value::Ptr>>> state_dict;
	state_dict["wte"] = matrix(vocab_size, n_embd, d, g); 
	state_dict["wpe"] = matrix(block_size, n_embd, d, g); 
	state_dict["lm_head"] = matrix(vocab_size, n_embd, d, g); 
	for (int i = 0; i < n_layer; i++) {
		state_dict["layer{" + to_string(i) + "}.attn_wq"] = matrix(n_embd, n_embd, d, g);
		state_dict["layer{" + to_string(i) + "}.attn_wk"] = matrix(n_embd, n_embd, d, g);
		state_dict["layer{" + to_string(i) + "}.attn_wv"] = matrix(n_embd, n_embd, d, g);
		state_dict["layer{" + to_string(i) + "}.attn_wo"] = matrix(n_embd, n_embd, d, g);
		state_dict["layer{" + to_string(i) + "}.mlp_fc1"] = matrix(4 * n_embd, n_embd, d, g);
		state_dict["layer{" + to_string(i) + "}.mlp_fc2"] = matrix(n_embd, 4 * n_embd, d, g);
	}
	vector<Value::Ptr> params;
	for (auto& pair : state_dict) {
		auto& vv = pair.second;
		for (int i = 0; i < vv.size(); i++) for (int j = 0; j < vv[i].size(); j++) params.push_back(vv[i][j]);
	}
	cout << "Num params: " << params.size() << endl;

	// Let there be Adam, the blessed optimizer and its buffers
	float learning_rate = 0.01;
	float beta1 = 0.85;
	float beta2 = 0.99;
	float eps_adam = 1e-8;
	vector<float> m(params.size(), 0.0); // first moment buffer
	vector<float> v(params.size(), 0.0); // second moment buffer
	
	// Repeat in sequence
	int num_steps = 1000; // number of training steps
	for (int step = 0; step < num_steps; step++) {
		//  Take single document, tokenize it, surround it with BOS special token on both sides
		string doc = docs[step % docs.size()];
		vector<int> tokens = { BOS };
		for (char ch : doc) {
			tokens.push_back(ch_to_token[ch]);
		}
		tokens.push_back(BOS);
		int n = min({block_size, static_cast<int>(tokens.size()) - 1});

		// Forward the token sequence through the model, building up the computation graph all the way to the loss
		vector<vector<vector<Value::Ptr>>> keys(n_layer, vector<vector<Value::Ptr>>());	
		vector<vector<vector<Value::Ptr>>> values(n_layer, vector<vector<Value::Ptr>>());

		vector<Value::Ptr> losses;
		Value::Ptr sum_losses = Value::make(0);
		for (int pos_id = 0; pos_id < n; pos_id++) {
			int token_id = tokens[pos_id];
			int target_id = tokens[pos_id + 1];
			vector<Value::Ptr> logits = gpt(token_id, pos_id, &keys, &values, &state_dict, n_layer, n_head, head_dim);
			vector<Value::Ptr> probs = softmax(logits);
			Value::Ptr neg1 = Value::make(-1);
			Value::Ptr logprob = Value::log(probs[target_id]);
			Value::Ptr loss_t = neg1 * logprob;
			losses.push_back(loss_t);
			sum_losses = sum_losses + loss_t;
		}
		Value::Ptr loss = Value::make(1.0/n) * sum_losses;

		// Backward the loss, calculating the gradients with respect to all model parameters
		loss->backward();

		// Adam optimizer update: update the model parameters based on the corresponding gradients
		float lr_t = learning_rate * (1 - step / num_steps); // linear learning rate decay
		for (int i = 0; i < params.size(); i++) {
			float grad = params[i]->grad;
			m[i] = beta1 * m[i] + (1 - beta1) * grad;
			v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
			float m_hat = m[i] / (1 - pow(beta1, step + 1));
			float v_hat = v[i] / (1 - pow(beta2, step + 1));
			params[i]->data -= lr_t * m_hat / (pow(v_hat, 0.5) + eps_adam); 
			params[i]->grad = 0;
		}

		if ((step+1) % 100 == 0) cout << "Step " << (step+1) << " / " << num_steps << " | loss = " << setprecision(6) << loss->data << endl;
	}

	// Inference: may the model babble back to us
	float temperature = 0.5; // in (0, 1], control the "creativity" of generated text, low to high
	cout << endl << "--- inference (new, hallucinated names) ---" << endl;
	for (int sample_idx = 0; sample_idx < 20; sample_idx++) {
		vector<vector<vector<Value::Ptr>>> keys(n_layer, vector<vector<Value::Ptr>>());	
		vector<vector<vector<Value::Ptr>>> values(n_layer, vector<vector<Value::Ptr>>());
		int token_id = BOS;
		string sample = "";
		for (int pos_id = 0; pos_id < block_size; pos_id++) {
			vector<Value::Ptr> logits = gpt(token_id, pos_id, &keys, &values, &state_dict, n_layer, n_head, head_dim);
			vector<Value::Ptr> tempered_logits;
			for (auto& val : logits) tempered_logits.push_back(val / Value::make(temperature));
			vector<Value::Ptr> probs = softmax(tempered_logits);
			vector<float> weights;
			for (auto& prob : probs) weights.push_back(prob->data);
			std::discrete_distribution<int> distribution(weights.begin(), weights.end());
			int token_id = distribution(g);
			if (token_id == BOS)
				break;
			sample += token_to_ch[token_id];
		}
		cout << "Sample " << sample_idx << " : " << sample << endl;
	}
	return 0;
}
