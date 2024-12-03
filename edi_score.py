import numpy as np
import pandas as pd
from utils import *
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
# from skopt import gp_minimize
# from skopt.space import Real


def calculate_edi_scores(embed_fpath, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Combine t-test p-values, mutual information scores, logistic regression coefficients,
    and PCA contributions into a single score (embedding dimension importance score).
    
    Args:
        embed_fpath (string): filename for embeddings
        weights (tuple): weights for t-test, MI, logistic, and PCA scores respectively
                        (should sum to 1)
    
    Returns:
        array-like: Combined scores for each dimension
    """
    
    # Load results
    t_test_results_df = pd.read_csv(os.path.join(get_results_directory(embed_fpath, "t_test_analysis"), "t_test_results.csv"))
    mi_results_df = pd.read_csv(os.path.join(get_results_directory(embed_fpath, "mutual_information"), "mutual_information_all.csv"))
    log_results_df = pd.read_csv(os.path.join(get_results_directory(embed_fpath, "logistic_classifier"), "classifier_weights.csv"))
    pca_results_df = pd.read_csv(os.path.join(get_results_directory(embed_fpath, "pca_analysis"), "pca_all.csv"))

    # Convert inputs to numpy arrays
    t_test_pvals = np.array(t_test_results_df['p_value'])
    mi_scores = np.array(mi_results_df['Mutual_Information'])
    log_coeffs = np.array(log_results_df['Weight'])
    pca_contribs = np.array(pca_results_df['Contribution'])
    
    # Calculate -log(p) for t-test scores
    epsilon = 1e-15
    t_test_scores = -np.log(t_test_pvals + epsilon)
    
    # Take absolute value of logistic coefficients
    log_scores = np.abs(log_coeffs)
    
    # Normalize each score type to [0,1] range
    t_test_scores = (t_test_scores - t_test_scores.min()) / (t_test_scores.max() - t_test_scores.min())
    mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
    log_scores = (log_scores - log_scores.min()) / (log_scores.max() - log_scores.min())
    pca_scores = (pca_contribs - pca_contribs.min()) / (pca_contribs.max() - pca_contribs.min())
    
    # Combine scores using weighted average
    combined_scores = (
        weights[0] * t_test_scores +
        weights[1] * mi_scores +
        weights[2] * log_scores +
        weights[3] * pca_scores
    )
    
    return combined_scores

def save_edi_scores(edi_scores, results_directory): 
    edi_df = pd.DataFrame({
        'Dimension': [i + 1 for i in range(len(edi_scores))],
        'EDI Score': edi_scores
    })

    edi_df.to_csv(os.path.join(results_directory, "edi_score.csv"), index=False)

    top_20_df = edi_df.nlargest(20, 'EDI Score')
    top_20_df.to_csv(os.path.join(results_directory, "top_20_edi_scores.csv"), index=False)

if __name__ == "__main__": 

    embedding_filepaths = get_embeddings_filepaths()

    for embeddings_csv in tqdm(embedding_filepaths):
        edi_scores = calculate_edi_scores(embeddings_csv)
        results_directory = get_results_directory(embeddings_csv, "edi_scores")
        save_edi_scores(edi_scores, results_directory)