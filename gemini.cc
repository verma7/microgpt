#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <set>
#include <map>
#include <algorithm>
#include <random>
#include <numeric>
#include <functional>

// ----------------------------------------------------------------------------
// Autograd Engine
// ----------------------------------------------------------------------------

struct ValueData;

class Value {
public:
    std::shared_ptr<ValueData> ptr;

    Value(double data = 0.0);
    Value(std::shared_ptr<ValueData> ptr) : ptr(ptr) {}

    double data() const;
    double& grad();

    Value operator+(const Value& other) const;
    Value operator*(const Value& other) const;
    Value operator-(const Value& other) const;
    Value operator/(const Value& other) const;
    Value pow(const double other) const;
    Value exp() const;
    Value log() const;
    Value relu() const;

    void backward();
};

struct ValueData {
    double data;
    double grad;
    std::vector<std::shared_ptr<ValueData>> _prev;
    std::string _op;
    std::function<void()> _backward;

    ValueData(double data, std::vector<std::shared_ptr<ValueData>> prev = {}, std::string op = "")
        : data(data), grad(0.0), _prev(prev), _op(op), _backward([](){}) {}
};

Value::Value(double data) : ptr(std::make_shared<ValueData>(data)) {}

double Value::data() const { return ptr->data; }
double& Value::grad() { return ptr->grad; }

Value Value::operator+(const Value& other) const {
    auto out = std::make_shared<ValueData>(this->data() + other.data(), 
               std::vector<std::shared_ptr<ValueData>>{this->ptr, other.ptr}, "+");
    out->_backward = [out, _this=this->ptr, _other=other.ptr]() {
        _this->grad += 1.0 * out->grad;
        _other->grad += 1.0 * out->grad;
    };
    return Value(out);
}

Value Value::operator*(const Value& other) const {
    auto out = std::make_shared<ValueData>(this->data() * other.data(), 
               std::vector<std::shared_ptr<ValueData>>{this->ptr, other.ptr}, "*");
    out->_backward = [out, _this=this->ptr, _other=other.ptr]() {
        _this->grad += _other->data * out->grad;
        _other->grad += _this->data * out->grad;
    };
    return Value(out);
}

Value Value::operator-(const Value& other) const {
    return *this + (other * Value(-1.0));
}

Value Value::pow(const double other) const {
    auto out = std::make_shared<ValueData>(std::pow(this->data(), other), 
               std::vector<std::shared_ptr<ValueData>>{this->ptr}, "^");
    out->_backward = [out, _this=this->ptr, other]() {
        _this->grad += (other * std::pow(_this->data, other - 1)) * out->grad;
    };
    return Value(out);
}

Value Value::operator/(const Value& other) const {
    return *this * other.pow(-1.0);
}

Value Value::exp() const {
    auto out = std::make_shared<ValueData>(std::exp(this->data()), 
               std::vector<std::shared_ptr<ValueData>>{this->ptr}, "exp");
    out->_backward = [out, _this=this->ptr]() {
        _this->grad += out->data * out->grad;
    };
    return Value(out);
}

Value Value::log() const {
    auto out = std::make_shared<ValueData>(std::log(this->data()), 
               std::vector<std::shared_ptr<ValueData>>{this->ptr}, "log");
    out->_backward = [out, _this=this->ptr]() {
        _this->grad += (1.0 / _this->data) * out->grad;
    };
    return Value(out);
}

Value Value::relu() const {
    double val = this->data() > 0 ? this->data() : 0.0;
    auto out = std::make_shared<ValueData>(val, 
               std::vector<std::shared_ptr<ValueData>>{this->ptr}, "relu");
    out->_backward = [out, _this=this->ptr]() {
        _this->grad += (_this->data > 0 ? 1.0 : 0.0) * out->grad;
    };
    return Value(out);
}

void Value::backward() {
    std::vector<std::shared_ptr<ValueData>> topo;
    std::set<std::shared_ptr<ValueData>> visited;
    
    std::function<void(std::shared_ptr<ValueData>)> build_topo = [&](std::shared_ptr<ValueData> v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (auto& child : v->_prev) build_topo(child);
            topo.push_back(v);
        }
    };
    
    build_topo(this->ptr);
    this->grad() = 1.0;
    
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}

// ----------------------------------------------------------------------------
// Utilities & Basic Layers
// ----------------------------------------------------------------------------

std::vector<Value> linear(const std::vector<Value>& x, const std::vector<std::vector<Value>>& w) {
    std::vector<Value> out(w.size(), Value(0.0));
    for (size_t i = 0; i < w.size(); ++i) {
        Value sum(0.0);
        for (size_t j = 0; j < x.size(); ++j) {
            sum = sum + w[i][j] * x[j];
        }
        out[i] = sum;
    }
    return out;
}

std::vector<Value> softmax(const std::vector<Value>& logits) {
    double max_val = logits[0].data();
    for (const auto& val : logits) {
        if (val.data() > max_val) max_val = val.data();
    }
    
    std::vector<Value> exps;
    Value total(0.0);
    for (const auto& val : logits) {
        Value e = (val - Value(max_val)).exp();
        exps.push_back(e);
        total = total + e;
    }
    
    std::vector<Value> out;
    for (const auto& e : exps) out.push_back(e / total);
    return out;
}

std::vector<Value> rmsnorm(const std::vector<Value>& x) {
    Value ms(0.0);
    for (const auto& xi : x) ms = ms + (xi * xi);
    ms = ms / Value(x.size());
    Value scale = (ms + Value(1e-5)).pow(-0.5);
    
    std::vector<Value> out;
    for (const auto& xi : x) out.push_back(xi * scale);
    return out;
}

Value rand_weight() {
    static std::mt19937 gen(42);
    static std::normal_distribution<double> d(0.0, 0.08);
    return Value(d(gen));
}

std::vector<std::vector<Value>> make_matrix(size_t rows, size_t cols) {
    std::vector<std::vector<Value>> mat(rows, std::vector<Value>(cols));
    for(size_t i=0; i<rows; ++i)
        for(size_t j=0; j<cols; ++j)
            mat[i][j] = rand_weight();
    return mat;
}

// Samples a token index based on the probability distribution
int sample(const std::vector<Value>& probs) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(gen);
    double cdf = 0.0;
    
    for (size_t i = 0; i < probs.size(); ++i) {
        cdf += probs[i].data();
        if (r <= cdf) return i;
    }
    return probs.size() - 1; // Fallback
}

// ----------------------------------------------------------------------------
// The GPT Architecture 
// ----------------------------------------------------------------------------

std::vector<Value> gpt(
    int token_id, 
    int pos_id, 
    std::vector<std::vector<std::vector<Value>>>& keys,
    std::vector<std::vector<std::vector<Value>>>& values,
    int n_layer, int n_head, int head_dim,
    std::map<std::string, std::vector<std::vector<Value>>>& state_dict
) {
    auto tok_emb = state_dict["wte"][token_id];
    auto pos_emb = state_dict["wpe"][pos_id];
    
    std::vector<Value> x(tok_emb.size());
    for(size_t i=0; i<x.size(); ++i) x[i] = tok_emb[i] + pos_emb[i];
    
    x = rmsnorm(x); 
    
    for (int li = 0; li < n_layer; ++li) {
        auto x_residual = x;
        x = rmsnorm(x);
        
        auto q = linear(x, state_dict["layer" + std::to_string(li) + ".attn_wq"]);
        auto k = linear(x, state_dict["layer" + std::to_string(li) + ".attn_wk"]);
        auto v = linear(x, state_dict["layer" + std::to_string(li) + ".attn_wv"]);
        
        // Append current token's keys and values to the KV cache
        keys[li].push_back(k);
        values[li].push_back(v);
        
        std::vector<Value> x_attn;
        for (int h = 0; h < n_head; ++h) {
            int hs = h * head_dim;
            std::vector<Value> q_h(q.begin() + hs, q.begin() + hs + head_dim);
            
            std::vector<std::vector<Value>> k_h, v_h;
            for (const auto& ki : keys[li]) k_h.push_back({ki.begin() + hs, ki.begin() + hs + head_dim});
            for (const auto& vi : values[li]) v_h.push_back({vi.begin() + hs, vi.begin() + hs + head_dim});
            
            std::vector<Value> attn_logits;
            for (size_t t = 0; t < k_h.size(); ++t) {
                Value dot(0.0);
                for (int j = 0; j < head_dim; ++j) {
                    dot = dot + q_h[j] * k_h[t][j];
                }
                attn_logits.push_back(dot / Value(std::sqrt(head_dim)));
            }
            
            auto attn_weights = softmax(attn_logits);
            
            std::vector<Value> head_out(head_dim, Value(0.0));
            for (int j = 0; j < head_dim; ++j) {
                Value sum(0.0);
                for (size_t t = 0; t < v_h.size(); ++t) {
                    sum = sum + attn_weights[t] * v_h[t][j];
                }
                head_out[j] = sum;
            }
            
            x_attn.insert(x_attn.end(), head_out.begin(), head_out.end());
        }
        
        x = linear(x_attn, state_dict["layer" + std::to_string(li) + ".attn_wo"]);
        for(size_t i=0; i<x.size(); ++i) x[i] = x[i] + x_residual[i];
        
        x_residual = x;
        x = rmsnorm(x);
        
        x = linear(x, state_dict["layer" + std::to_string(li) + ".mlp_fc1"]);
        for(size_t i=0; i<x.size(); ++i) x[i] = x[i].relu();
        
        x = linear(x, state_dict["layer" + std::to_string(li) + ".mlp_fc2"]);
        for(size_t i=0; i<x.size(); ++i) x[i] = x[i] + x_residual[i];
    }
    
    return linear(x, state_dict["lm_head"]);
}

// ----------------------------------------------------------------------------
// Main: Training & Inference
// ----------------------------------------------------------------------------

int main() {
// Let there be a Dataset `docs` of documents (e.g. a list of names)
        std::vector<std::string> docs;
        std::string doc;
        while (std::cin >> doc) docs.push_back(doc);

        std::mt19937 g(42);  // Let there be order among chaos
        std::shuffle(docs.begin(), docs.end(), g);
        std::cout << "Num docs: " << docs.size() << std::endl;

        // Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
        std::set<char> uchars;
        for (const auto& doc : docs) for (char ch : doc) uchars.insert(ch);
        int BOS = uchars.size();
        int vocab_size = uchars.size() + 1;
        std::cout << "Vocab size: " << vocab_size << std::endl;
   
	// Char to Token mapping and vice-versa
        std::unordered_map<char, int> char_to_id;
        std::unordered_map<int, char> id_to_char;
        int tok = 0;
        for (char ch : uchars) {
                char_to_id[ch] = tok;
                id_to_char[tok] = ch;
                tok++;
        }
    
    // Model hyperparams
    int n_layer = 1;
    int n_embd = 16;
    int block_size = 64; // Increased to allow slightly longer generation
    int n_head = 4;
    int head_dim = n_embd / n_head;

    // State dict init
    std::map<std::string, std::vector<std::vector<Value>>> state_dict;
    state_dict["wte"] = make_matrix(vocab_size, n_embd);
    state_dict["wpe"] = make_matrix(block_size, n_embd);
    state_dict["lm_head"] = make_matrix(vocab_size, n_embd);

    for (int i = 0; i < n_layer; ++i) {
        std::string prefix = "layer" + std::to_string(i) + ".";
        state_dict[prefix + "attn_wq"] = make_matrix(n_embd, n_embd);
        state_dict[prefix + "attn_wk"] = make_matrix(n_embd, n_embd);
        state_dict[prefix + "attn_wv"] = make_matrix(n_embd, n_embd);
        state_dict[prefix + "attn_wo"] = make_matrix(n_embd, n_embd);
        state_dict[prefix + "mlp_fc1"] = make_matrix(4 * n_embd, n_embd);
        state_dict[prefix + "mlp_fc2"] = make_matrix(n_embd, 4 * n_embd);
    }

    std::vector<Value*> parameters;
    for (auto& [name, mat] : state_dict) {
        for (auto& row : mat) {
            for (auto& v : row) parameters.push_back(&v);
        }
    }

    // --- Training Loop ---
    double learning_rate = 0.05; // Slightly bumped for faster overfitting visually
    std::cout << "--- Training Phase ---" << std::endl;

    for (int step = 0; step < 3000; ++step) {
	//  Take single document, tokenize it, surround it with BOS special token on both sides
        std::string doc = docs[step % docs.size()];
        std::vector<int> ctx = {BOS};
        for(char c : doc) ctx.push_back(char_to_id[c]);
       	ctx.push_back(BOS);
 
        int n = std::min(block_size, (int)ctx.size() - 1);
        std::vector<std::vector<std::vector<Value>>> keys(n_layer), values(n_layer);
        
        Value loss(0.0);
        
        for (int t = 0; t < n; ++t) {
            int token_id = ctx[t];
            int next_id = ctx[t + 1];
            
            std::vector<Value> logits = gpt(token_id, t, keys, values, n_layer, n_head, head_dim, state_dict);
            std::vector<Value> probs = softmax(logits);
            
            loss = loss - probs[next_id].log();
        }
        
        loss = loss / Value((double)n);
        
        for (auto* p : parameters) p->grad() = 0.0;
        loss.backward();
        
        for (auto* p : parameters) {
            p->ptr->data -= learning_rate * p->grad();
        }
        
        if (step % 100 == 0) {
            std::cout << "Step " << step << " | Loss: " << loss.data() << std::endl;
        }
    }

    // --- Inference / Generation Loop ---
    std::cout << "\n--- Generation Phase ---" << std::endl;
    std::cout << "Generating text: ";
    
    // We need fresh, empty KV caches for generation
    std::vector<std::vector<std::vector<Value>>> gen_keys(n_layer), gen_values(n_layer);
   
    for (int sample_idx = 0; sample_idx < 20; sample_idx++) { 
	int current_token = BOS;
    	int max_new_tokens = 30; // Number of characters to generate
    
         std::cout << "Sample " << sample_idx << " : " ;
   	 for (int t = 0; t < max_new_tokens; ++t) {
        	// Run forward pass. The KV cache is automatically updated inside `gpt`
	        std::vector<Value> logits = gpt(current_token, t, gen_keys, gen_values, n_layer, n_head, head_dim, state_dict);
        	std::vector<Value> probs = softmax(logits);
        
	        // Pick next token
        	current_token = sample(probs);
        
	        // Print generated character
        	if (id_to_char.find(current_token) != id_to_char.end()) {
			std::cout << id_to_char[current_token] << std::flush;
	        } else {
        	    break; // Hit a token outside standard characters (like BOS again)
       		}
	}
	std::cout << std::endl;
    }
    std::cout << "\n\nDone." << std::endl;

    return 0;
}
