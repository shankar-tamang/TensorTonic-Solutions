import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model//num_heads

    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_k)

    def split_heads(x):
        x = x.reshape(batch_size, -1, num_heads, d_k)
        return x.transpose(0, 2, 1, 3)

    Q_heads = split_heads(Q_proj)
    K_heads = split_heads(K_proj)
    V_heads = split_heads(V_proj)

    scaled_dot_product = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2))/np.sqrt(d_k)

    attention_weights = softmax(scaled_dot_product, axis=-1)

    head_outputs = np.matmul(attention_weights, V_heads)

    concat_heads = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    output = np.matmul(concat_heads, W_o)

    return output
    