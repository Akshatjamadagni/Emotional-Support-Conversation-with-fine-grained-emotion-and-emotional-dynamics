# Causal Encoder Module  

This module implements key components of a causal encoder, focusing on cross-attention mechanisms and query vector computation for representing and analyzing causal relationships in a model's context.  

### Components  

#### 1. **CrossAttentionModule**  

This module implements a multi-head cross-attention mechanism with residual connections and layer normalization.  

**Key Features:**  
- Multi-head attention for capturing relationships between the input tensors.  
- Residual connections for stabilizing training.  
- Layer normalization for better convergence.  

**Class Definition:**  

```python
class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        ...
    def forward(self, key, query, context):
        ...
```

**Parameters:**  
- `hidden_dim`: Dimensionality of the hidden representations.  

**Input:**  
- `key`, `query`, `context` tensors for the attention mechanism.  

**Output:**  
- Cross-attended tensor with residual connections and normalization applied.  

---

#### 2. **ComprehensiveEffectRepresentation**  

This module uses two instances of `CrossAttentionModule` to represent causal effects comprehensively. It processes intra- and inter-causal representations using cross-attention mechanisms.  

**Class Definition:**  

```python
class ComprehensiveEffectRepresentation(nn.Module):
    def __init__(self, hidden_dim):
        ...
    def forward(self, KEC_intra, KEC_inter, Hc, Hq):
        ...
```

**Parameters:**  
- `hidden_dim`: Dimensionality of the hidden representations.  

**Input:**  
- `KEC_intra`: Intra-emotion causal representation tensor.  
- `KEC_inter`: Inter-emotion causal representation tensor.  
- `Hc`: Cause-aware context representation tensor.  
- `Hq`: Query-aware context representation tensor.  

**Output:**  
- `KEC_hat`: Updated causal effects representation.  
- `KES_hat`: Updated strategy-aware effects representation.  

---

#### 3. **Query Vector Computation**  

This function computes a query vector by mean-pooling the strategy history and context representations, then concatenating them.  

**Function Definition:**  

```python
def compute_query_vector(Hs, Hc):
    ...
```

**Parameters:**  
- `Hs`: Strategy history tensor `[sequence_length, batch_size, hidden_dim]`.  
- `Hc`: Context representation tensor `[sequence_length, batch_size, hidden_dim]`.  

**Output:**  
- Query vector of shape `[batch_size, 2 * hidden_dim]`.  

---

### Example Usage  

1. **CrossAttentionModule Example:**  

```python
hidden_dim = 512
cross_attention_module = ComprehensiveEffectRepresentation(hidden_dim)

K_EC_intra = torch.randn(10, 32, hidden_dim)
K_EC_inter = torch.randn(10, 32, hidden_dim)
H_c = torch.randn(10, 32, hidden_dim)
H_q = torch.randn(10, 32, hidden_dim)

K_EC_hat, K_ES_hat = cross_attention_module(K_EC_intra, K_EC_inter, H_c, H_q)
print(KEC_hat.shape, KES_hat.shape)
```

2. **Query Vector Computation Example:**  

```python
Hs = torch.randn(10, 32, 512)
Hc = torch.randn(10, 32, 512)

h = compute_query_vector(Hs, Hc)
print(h.shape)  # Output: [32, 1024]
```

---

### Dependencies  

- Python 3.7 or higher  
- PyTorch 1.8 or higher  

Install PyTorch using:  

```
!pip install torch
```

--- 
