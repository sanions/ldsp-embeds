import numpy as np
from scipy.stats import gaussian_kde
from utils import *
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os

def calculate_kl_divergence(dist1, dist2, num_points=1000):
    """
    Calculate KL divergence between two distributions using kernel density estimation.
    
    Args:
        dist1: First distribution values
        dist2: Second distribution values
        num_points: Number of points to use for distribution estimation
    
    Returns:
        KL divergence value
    """
    # Estimate the PDFs using kernel density estimation
    kde1 = gaussian_kde(dist1)
    kde2 = gaussian_kde(dist2)
    
    # Create evaluation points
    x_min = min(dist1.min(), dist2.min())
    x_max = max(dist1.max(), dist2.max())
    x_eval = np.linspace(x_min, x_max, num_points)
    
    # Evaluate PDFs
    pdf1 = kde1(x_eval)
    pdf2 = kde2(x_eval)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    pdf1 = pdf1 + epsilon
    pdf2 = pdf2 + epsilon
    
    # Calculate KL divergence: KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
    kl_div = np.sum(pdf1 * np.log(pdf1 / pdf2)) * (x_max - x_min) / num_points
    
    return kl_div

def calculate_embedding_kl_divergence(embeddings1, embeddings2):
    """
    Calculate KL divergence for each dimension of the embeddings.
    
    Args:
        embeddings1: Array of embeddings for first sentences [num_sentences, embedding_dim]
        embeddings2: Array of embeddings for second sentences [num_sentences, embedding_dim]
    
    Returns:
        Array of KL divergence values for each dimension
    """
    embedding_dim = embeddings1.shape[1]
    kl_divergences = np.zeros(embedding_dim)
    
    for dim in range(embedding_dim):
        dist1 = embeddings1[:, dim]
        dist2 = embeddings2[:, dim]
        kl_divergences[dim] = calculate_kl_divergence(dist1, dist2)
    
    return kl_divergences

def save_kl_divergence_results(kl_values, dimensions, embeddings_csv):
    """
    Save KL divergence results and create visualization comparing with t-test results.
    
    Args:
        kl_values: Array of KL divergence values
        dimensions: Array of corresponding dimension indices
        embeddings_csv: Path to the original embeddings CSV file
    """
    # Create results directory
    results_directory = get_results_directory(embeddings_csv, "kl_divergence_analysis")
    
    # Save KL divergence results
    results_df = pd.DataFrame({
        'dimension': dimensions,
        'kl_divergence': kl_values
    })
    results_df.to_csv(os.path.join(results_directory, "kl_divergence_results.csv"), index=False)
    
    # Load t-test results for comparison
    t_test_path = os.path.join(
        os.path.dirname(results_directory),
        "t_test_analysis",
        "t_test_results.csv"
    )
    t_test_df = pd.read_csv(t_test_path)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(-np.log10(t_test_df['p_value']), kl_values, alpha=0.5)
    plt.xlabel('-log10(t-test p-value)')
    plt.ylabel('KL Divergence')
    plt.title('Comparison of KL Divergence vs T-test Significance')
    
    # Add dimension labels for top 5 most interesting points
    interest_score = kl_values * (-np.log10(t_test_df['p_value']))
    top_indices = np.argsort(interest_score)[-5:]
    
    for idx in top_indices:
        plt.annotate(f'Dim {dimensions[idx]}', 
                    (-np.log10(t_test_df['p_value'].iloc[idx]), kl_values[idx]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "kl_vs_ttest_comparison.png"))
    plt.close()

if __name__ == "__main__": 
    embedding_filepaths = get_embeddings_filepaths()
    for embedding_csv in tqdm(embedding_filepaths):
        df = read_embeddings_df(embedding_csv)
        e1, e2 = df['Sentence1_embedding'], df['Sentence2_embedding']
        e1 = np.array([np.array(d) for d in e1])
        e2 = np.array([np.array(d) for d in e2])

        kl = calculate_embedding_kl_divergence(e1, e2)
        kl_values = np.sort(kl)[::-1]
        most_influential_dimensions = np.argsort(kl)[::-1]
        
        save_kl_divergence_results(kl, most_influential_dimensions, embedding_csv)