import numpy as np

def compute_geometric_features(audio_emb, text_emb):
    """
    The 'Geometric Injection' module.
    Calculates manual interaction metrics between aligned Audio and Text tensors.
    
    Args:
        audio_emb (np.array): Shape (N, D)
        text_emb (np.array): Shape (N, D) - Must match audio dims or be projected
        
    Returns:
        np.array: Shape (N, 4) containing [CosSim, AngDist, L1, L2]
    """
    # Ensure inputs are numpy arrays
    a = np.array(audio_emb)
    t = np.array(text_emb)
    
    # Handle dimension mismatch via padding if necessary (simple heuristic)
    if a.shape[1] != t.shape[1]:
        min_dim = min(a.shape[1], t.shape[1])
        a = a[:, :min_dim]
        t = t[:, :min_dim]

    # 1. Cosine Similarity (Element-wise for paired data)
    # Dot product / (Norm A * Norm B)
    dot_product = np.sum(a * t, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_t = np.linalg.norm(t, axis=1)
    # Avoid divide by zero
    cos_sim = dot_product / (norm_a * norm_t + 1e-8)
    
    # 2. Angular Distance
    # Formula: 1 - arccos(cos_sim) / pi
    # Clip cos_sim to [-1, 1] to prevent NaN in arccos
    ang_dist = 1 - (np.arccos(np.clip(cos_sim, -1.0, 1.0)) / np.pi)
    
    # 3. L1 Norm (Manhattan Distance) of the difference vector
    l1_dist = np.sum(np.abs(a - t), axis=1)
    
    # 4. L2 Norm (Euclidean Distance) of the difference vector
    l2_dist = np.linalg.norm(a - t, axis=1)
    
    # Stack into a (N, 4) matrix
    return np.stack([cos_sim, ang_dist, l1_dist, l2_dist], axis=1)
