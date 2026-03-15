import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_normalized = (x - mean)/np.sqrt(var + eps)
    return gamma * x_normalized + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here

    # Initialization
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads  

    # Q, K, V projection
    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)
    
    
    def split_heads(x):
        x = x.reshape(batch, -1, num_heads, d_k)
        return x.transpose(0, 2, 1, 3)

    # Head splitter
    Q_heads = split_heads(Q_proj)
    K_heads = split_heads(K_proj)
    V_heads = split_heads(V_proj)

    # attention weights 
    scaled_dot_product = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attention_weights = softmax(scaled_dot_product, axis=-1)
    

    # weighted value
    heads_output = np.matmul(attention_weights, V_heads)

    # concat

    concat_heads = heads_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    output = np.matmul(concat_heads, W_o)

    return output

    

    

    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    y = np.matmul(x, W1) + b1
    z = np.maximum(0, y)
    y2 = np.matmul(z, W2) + b2
    return y2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here

    MHA = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)

    mha_block_output = layer_norm(x + MHA, gamma1, beta1)

    FFN = feed_forward(mha_block_output, W1, b1, W2, b2)

    output = layer_norm(mha_block_output + FFN, gamma2, beta2)

    return output