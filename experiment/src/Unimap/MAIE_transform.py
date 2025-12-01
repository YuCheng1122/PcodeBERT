import numpy as np
import os
from typing import Tuple
from gensim.models import KeyedVectors
from tqdm import tqdm

WORD2VEC_MODEL = "/home/tommy/Project/PcodeBERT/experiment/model/Unimap/MAIE"


def maie_transform(
    src_emb: np.ndarray,
    trg_emb: np.ndarray,
    vocab_cutoff: int = 4000,
    use_svd_smoothing: bool = False,
    svd_components: int = 100,
    bidirectional: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MAIE (Multi-Architecture Instruction Embedding) Transform
    Creates initial dictionary by matching instruction distributions between architectures
    
    Args:
        src_emb: Source architecture embeddings [vocab_size, dim]
        trg_emb: Target architecture embeddings [vocab_size, dim]
        vocab_cutoff: Number of most frequent instructions to use (default: 4000)
        use_svd_smoothing: Whether to apply SVD smoothing on self-similarity matrices
        svd_components: Number of SVD components to keep for smoothing
        bidirectional: Whether to use bidirectional matching for better quality
    
    Returns:
        src_indices: Source instruction indices
        trg_indices: Corresponding target instruction indices
    """
    
    # Step 1: Slicing - extract top vocab_cutoff instructions
    # Shape: X [vocab_cutoff, dim], Z [vocab_cutoff, dim]
    X = src_emb[:vocab_cutoff]
    Z = trg_emb[:vocab_cutoff]
    
    print(f"Step 1: Sliced embeddings - X shape: {X.shape}, Z shape: {Z.shape}")
    
    # Step 2: Compute self-similarity matrices
    # S_X shape: [vocab_cutoff, vocab_cutoff], S_Z shape: [vocab_cutoff, vocab_cutoff]
    S_X = np.dot(X, X.T)  # Source self-similarity
    S_Z = np.dot(Z, Z.T)  # Target self-similarity
    
    print(f"Step 2: Self-similarity - S_X shape: {S_X.shape}, S_Z shape: {S_Z.shape}")
    
    # Optional: SVD smoothing for better quality
    if use_svd_smoothing:
        S_X = apply_svd_smoothing(S_X, svd_components)
        S_Z = apply_svd_smoothing(S_Z, svd_components)
        print(f"Applied SVD smoothing with {svd_components} components")
    
    # Step 3: Sorting (CRITICAL) - sort each row to extract similarity distribution
    # This removes position-specific information and focuses on distribution patterns
    # Shape remains: [vocab_cutoff, vocab_cutoff]
    S_X_sorted = np.sort(S_X, axis=1)  # Sort each row independently
    S_Z_sorted = np.sort(S_Z, axis=1)  # Sort each row independently
    
    print(f"Step 3: Sorted rows - S_X_sorted shape: {S_X_sorted.shape}, S_Z_sorted shape: {S_Z_sorted.shape}")
    
    # Step 4: Normalization
    # Apply L2 normalization to each row
    S_X_normalized = normalize_rows(S_X_sorted, mean_center=True)
    S_Z_normalized = normalize_rows(S_Z_sorted, mean_center=True)
    
    print(f"Step 4: Normalized - S_X_normalized shape: {S_X_normalized.shape}, S_Z_normalized shape: {S_Z_normalized.shape}")
    
    # Step 5: Matching - compute similarity between sorted distributions
    # Sim shape: [vocab_cutoff, vocab_cutoff]
    # Sim[i,j] represents similarity between src instruction i and trg instruction j
    Sim = np.dot(S_X_normalized, S_Z_normalized.T)
    
    print(f"Step 5: Matching - Similarity matrix shape: {Sim.shape}")
    
    # Step 6: Generate dictionary
    if bidirectional:
        # Bidirectional matching for better quality
        src_indices, trg_indices = bidirectional_matching(Sim)
    else:
        # Simple unidirectional matching
        src_indices, trg_indices = unidirectional_matching(Sim)
    
    print(f"Step 6: Generated dictionary with {len(src_indices)} pairs")
    
    return src_indices, trg_indices


def apply_svd_smoothing(similarity_matrix: np.ndarray, n_components: int) -> np.ndarray:
    """
    Apply SVD smoothing to improve self-similarity matrix quality
    Reduces noise by keeping only top singular values
    
    Args:
        similarity_matrix: Self-similarity matrix [N, N]
        n_components: Number of SVD components to keep
    
    Returns:
        Smoothed similarity matrix [N, N]
    """
    # Perform SVD decomposition
    u, s, vt = np.linalg.svd(similarity_matrix)
    
    # Keep only top n_components
    u_reduced = u[:, :n_components]  # [N, n_components]
    s_reduced = s[:n_components]      # [n_components]
    
    # Reconstruct smoothed matrix: (U * S) * U^T
    # This preserves symmetry and filters out noise
    smoothed = np.dot(u_reduced * s_reduced, u_reduced.T)  # [N, N]
    
    return smoothed


def normalize_rows(matrix: np.ndarray, mean_center: bool = True) -> np.ndarray:
    """
    Normalize each row of the matrix (L2 normalization with optional mean centering)
    
    Args:
        matrix: Input matrix [N, M]
        mean_center: Whether to subtract mean before normalization
    
    Returns:
        Row-normalized matrix [N, M]
    """
    # Mean centering (optional but recommended)
    if mean_center:
        matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
    
    # L2 normalization for each row
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero
    row_norms = np.maximum(row_norms, 1e-10)
    normalized = matrix / row_norms
    
    return normalized


def unidirectional_matching(similarity_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple unidirectional matching: for each source, find best target
    
    Args:
        similarity_matrix: Similarity scores [src_size, trg_size]
    
    Returns:
        src_indices: Source indices
        trg_indices: Matched target indices
    """
    # For each source instruction, find the best matching target
    best_matches = np.argmax(similarity_matrix, axis=1)  # [src_size]
    
    # Create index arrays
    src_indices = np.arange(len(best_matches))
    trg_indices = best_matches
    
    return src_indices, trg_indices


def bidirectional_matching(similarity_matrix: np.ndarray, threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bidirectional matching: mutual best matches for higher quality
    
    Args:
        similarity_matrix: Similarity scores [src_size, trg_size]
        threshold: Minimum similarity threshold for valid matches
    
    Returns:
        src_indices: Source indices with mutual matches
        trg_indices: Matched target indices
    """
    # Forward matching: src -> trg
    forward_matches = np.argmax(similarity_matrix, axis=1)  # [src_size]
    
    # Backward matching: trg -> src
    backward_matches = np.argmax(similarity_matrix, axis=0)  # [trg_size]
    
    # Find mutual matches (bidirectional consistency)
    src_indices = []
    trg_indices = []
    
    for src_idx in range(len(forward_matches)):
        trg_idx = forward_matches[src_idx]
        # Check if it's a mutual match
        if backward_matches[trg_idx] == src_idx:
            # Optional: check similarity threshold
            if similarity_matrix[src_idx, trg_idx] > threshold:
                src_indices.append(src_idx)
                trg_indices.append(int(trg_idx))
    
    return np.array(src_indices), np.array(trg_indices)


def orthogonal_mapping(X: np.ndarray, Z: np.ndarray, 
                       src_indices: np.ndarray, trg_indices: np.ndarray) -> np.ndarray:
    """
    Compute optimal linear transformation matrix T (Orthogonal Mapping)
    Solves: min || X[src] * T - Z[trg] ||_F using Procrustes problem
    
    Args:
        X: Source embeddings [vocab_size, dim]
        Z: Target embeddings [vocab_size, dim]
        src_indices: Source instruction indices
        trg_indices: Target instruction indices
    
    Returns:
        W: Orthogonal transformation matrix [dim, dim]
    """
    # Extract aligned embeddings using the dictionary
    X_aligned = X[src_indices]  # [dict_size, dim]
    Z_aligned = Z[trg_indices]  # [dict_size, dim]
    
    # Solve Procrustes problem using SVD
    # We want to find W such that X_aligned @ W ≈ Z_aligned
    # Solution: W = V @ U^T where Z^T @ X = U @ S @ V^T
    u, s, vt = np.linalg.svd(np.dot(Z_aligned.T, X_aligned))
    W = np.dot(vt.T, u.T)  # [dim, dim]
    
    return W


def csls_retrieval(X_mapped: np.ndarray, Z: np.ndarray, k: int = 10) -> np.ndarray:
    """
    CSLS (Cross-domain Similarity Local Scaling)
    Mitigates hubness problem for more accurate matching
    
    Args:
        X_mapped: Mapped source embeddings [src_size, dim]
        Z: Target embeddings [trg_size, dim]
        k: Number of neighbors for local scaling
    
    Returns:
        csls_scores: CSLS similarity scores [src_size, trg_size]
    """
    # Compute cosine similarity
    # Normalize embeddings first
    X_norm = X_mapped / np.linalg.norm(X_mapped, axis=1, keepdims=True)
    Z_norm = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    
    # Cosine similarity matrix [src_size, trg_size]
    cosine_sim = np.dot(X_norm, Z_norm.T)
    
    # Compute mean similarity to k nearest neighbors
    # For X_mapped (source side)
    nearest_src = np.partition(cosine_sim, -k, axis=1)[:, -k:]  # [src_size, k]
    mean_src = np.mean(nearest_src, axis=1, keepdims=True)  # [src_size, 1]
    
    # For Z (target side)
    nearest_trg = np.partition(cosine_sim.T, -k, axis=1)[:, -k:]  # [trg_size, k]
    mean_trg = np.mean(nearest_trg, axis=1, keepdims=True)  # [trg_size, 1]
    
    # CSLS score: 2 * cos(x,y) - mean_cos(x) - mean_cos(y)
    # This penalizes hubs (points with high average similarity)
    csls_scores = 2 * cosine_sim - mean_src - mean_trg.T
    
    return csls_scores


def self_learning_loop(
    src_emb: np.ndarray,
    trg_emb: np.ndarray,
    initial_src_indices: np.ndarray,
    initial_trg_indices: np.ndarray,
    max_iters: int = 500,           # Max iterations
    convergence_threshold: float = 1e-6, # Convergence threshold
    stochastic_initial: float = 0.1, # Initial keep probability
    stochastic_multiplier: float = 2.0, # Multiplier for probability increase
    stochastic_interval: int = 50    # Iterations before increasing probability
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Self-Learning Loop (Iterative Refinement)
    Refines the mapping and dictionary iteratively until convergence.
    
    Returns:
        final_src_indices, final_trg_indices, final_W
    """
    print("\n" + "="*30)
    print("Starting Self-Learning Loop")
    print("="*30)
    
    # Copy indices to avoid modifying external variables
    src_indices = initial_src_indices.copy()
    trg_indices = initial_trg_indices.copy()
    
    # Initialize variables
    keep_prob = stochastic_initial
    best_objective = -100.0
    last_improvement = 0
    
    # Store best results
    best_W = None
    best_src_indices = None
    best_trg_indices = None
    
    # Create progress bar for iterations
    pbar = tqdm(range(1, max_iters + 1), desc="Self-Learning", ncols=120)
    
    for it in pbar:
        # Step 1: Update Mapping - compute optimal transformation matrix W
        W = orthogonal_mapping(src_emb, trg_emb, src_indices, trg_indices)
        
        # Step 2: Update Dictionary
        # Map source to target space
        src_emb_mapped = np.dot(src_emb, W)
        
        # Compute CSLS scores
        csls_scores = csls_retrieval(src_emb_mapped, trg_emb, k=10)
        
        # Stochastic Dictionary Induction - randomly drop some pairs
        if keep_prob < 1.0:
            # Generate random mask
            dropout_mask = np.random.rand(*csls_scores.shape) >= keep_prob
            # Temporarily modify scores for selection
            scores_for_selection = csls_scores.copy()
            scores_for_selection[dropout_mask] = -np.inf
        else:
            scores_for_selection = csls_scores

        # Find best target for each source (Forward Matching)
        new_trg_indices = np.argmax(scores_for_selection, axis=1)
        new_src_indices = np.arange(len(new_trg_indices))
        
        # Step 3: Convergence Check
        # Calculate objective: mean best similarity
        current_best_scores = csls_scores[new_src_indices, new_trg_indices]
        objective = np.mean(current_best_scores)
        
        # Check for improvement
        if objective - best_objective >= convergence_threshold:
            last_improvement = it
            best_objective = objective
            # Save current best state
            best_W = W.copy()
            best_src_indices = new_src_indices.copy()
            best_trg_indices = new_trg_indices.copy()
        
        # Update progress bar with current metrics
        pbar.set_postfix({
            'obj': f'{objective:.4f}',
            'best': f'{best_objective:.4f}',
            'keep_p': f'{keep_prob:.2f}',
            'last_imp': last_improvement
        })
        
        # Step 4: Update Hyperparameters
        # Increase keep_prob if no improvement for a while
        if it - last_improvement > stochastic_interval:
            if keep_prob >= 1.0:
                pbar.close()
                print(f"\nConverged at iteration {it}. Best Objective: {best_objective:.6f}")
                break # Converged
                
            keep_prob = min(1.0, stochastic_multiplier * keep_prob)
            last_improvement = it
            tqdm.write(f"Iter {it}: Increasing keep_prob to {keep_prob:.2f} (Objective: {objective:.6f})")
            
        # Update indices for next iteration
        src_indices = new_src_indices
        trg_indices = new_trg_indices
    
    # Close progress bar if loop completes normally
    if it == max_iters:
        pbar.close()
        print(f"\nReached max iterations ({max_iters}). Best Objective: {best_objective:.6f}")

    return best_src_indices, best_trg_indices, best_W


def save_results(transform_matrix: np.ndarray, 
                 src_indices: np.ndarray, 
                 trg_indices: np.ndarray, 
                 output_dir: str) -> None:
    """
    Save transformation matrix and dictionary indices to disk
    
    Args:
        transform_matrix: Orthogonal transformation matrix [dim, dim]
        src_indices: Source instruction indices
        trg_indices: Target instruction indices
        output_dir: Directory to save the results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save transformation matrix
    t_path = os.path.join(output_dir, "transformation_matrix_W.npy")
    np.save(t_path, transform_matrix)
    print(f"Saved transformation matrix to: {t_path}")
    
    # Save dictionary indices
    d_indices_path = os.path.join(output_dir, "dictionary_indices.npz")
    np.savez(d_indices_path, src=src_indices, trg=trg_indices)
    print(f"Saved dictionary indices to: {d_indices_path}")


# Example usage function
def run_maie_transform(src_arch: str = "x86_64", trg_arch: str = "arm_32"):
    """
    Example function to run MAIE transform between two architectures
    
    Args:
        src_arch: Source architecture name
        trg_arch: Target architecture name
    """
    print(f"Loading real embeddings for {src_arch} and {trg_arch}...")
    
    # 1. Load real Word2Vec models
    # Load KeyedVectors from saved files
    src_kv = KeyedVectors.load(f"{WORD2VEC_MODEL}/skipgram_{src_arch}_vectors.kv")
    trg_kv = KeyedVectors.load(f"{WORD2VEC_MODEL}/skipgram_{trg_arch}_vectors.kv")
    
    # 2. Get vector matrices
    # Gensim's vectors attribute is already a numpy array
    src_emb = src_kv.vectors.astype(np.float32)
    trg_emb = trg_kv.vectors.astype(np.float32)
    
    # 3. Get corresponding vocabularies for readable output
    src_vocab = src_kv.index_to_key
    trg_vocab = trg_kv.index_to_key
    
    print(f"Loaded shapes: src {src_emb.shape}, trg {trg_emb.shape}")
    print(f"Vocabulary sizes: src {len(src_vocab)}, trg {len(trg_vocab)}")
    
    # 1. Initial dictionary generation
    print("--- Phase 1: Initial Dictionary Generation ---")
    with tqdm(total=1, desc="Initial Dictionary", ncols=100) as pbar:
        init_src_idx, init_trg_idx = maie_transform(
            src_emb, 
            trg_emb,
            vocab_cutoff=4000,
            use_svd_smoothing=True,
            svd_components=100,
            bidirectional=True
        )
        pbar.update(1)
    
    # Compute initial orthogonal mapping
    print("Computing initial orthogonal mapping...")
    with tqdm(total=1, desc="Orthogonal Mapping", ncols=100) as pbar:
        W_init = orthogonal_mapping(src_emb, trg_emb, init_src_idx, init_trg_idx)
        pbar.update(1)
        pbar.update(1)
    print(f"Initial orthogonal mapping matrix shape: {W_init.shape}")
    
    # 2. Self-Learning Loop
    # Key part - iteratively refine the mapping
    print("--- Phase 2: Self-Learning ---")
    final_src_idx, final_trg_idx, final_W = self_learning_loop(
        src_emb, 
        trg_emb, 
        init_src_idx, 
        init_trg_idx
    )
    
    # 3. Validation and output
    # Map source embeddings to target space using the FINAL W
    src_emb_mapped = np.dot(src_emb, final_W)
    
    # Calculate final CSLS scores for observation
    final_csls = csls_retrieval(src_emb_mapped[:1000], trg_emb[:1000], k=10)
    print(f"Final CSLS scores shape: {final_csls.shape}")
    
    return final_src_idx, final_trg_idx, final_W


if __name__ == "__main__":
    # Test the implementation
    try:
        src_idx, trg_idx, transform_matrix = run_maie_transform()
        print(f"\nSuccess! Generated {len(src_idx)} instruction pairs")
        print(f"Transform matrix shape: {transform_matrix.shape}")
        
        # Save results to output directory
        output_dir = "/home/tommy/Project/PcodeBERT/experiment/outputs/MAIE"
        save_results(transform_matrix, src_idx, trg_idx, output_dir)
        print(f"\nAll results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
