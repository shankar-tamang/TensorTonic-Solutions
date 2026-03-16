import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    """
    Convert image to patch embeddings.
    """
    # YOUR CODE HERE
    B, H, W, C = image.shape
    p = patch_size

    assert H % patch_size == 0 and W % patch_size == 0
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    N = num_patches_h * num_patches_w

    patches = image.reshape(
                            B, 
                            num_patches_h, 
                            p, 
                            num_patches_w, 
                            p, 
                            C
    )

    patches = patches.transpose(0, 1, 3, 2, 4, 5)

    patches = patches.reshape(B, N, p*p*C)

    W_proj = np.random.randn(p*p*C, embed_dim)

    embeddings = patches @ W_proj

    return embeddings

    