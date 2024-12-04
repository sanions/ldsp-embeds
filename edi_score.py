import numpy as np
import pandas as pd
from utils import *
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
from skopt import gp_minimize
from skopt.space import Real


def calculate_edi_scores(embed_fpath, weights=(0.34, 0.33, 0.33)):
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
    rfe_results_df = pd.read_csv(os.path.join(get_results_directory(embed_fpath, "rfe_analysis"), "rfe_results.csv"))

    # Convert inputs to numpy arrays
    t_test_pvals = np.array(t_test_results_df['p_value'])
    mi_scores = np.array(mi_results_df['Mutual_Information'])

    rfe_importance = np.zeros(len(t_test_pvals))
    rfe_importance[rfe_results_df['Feature']] = rfe_results_df['Importance']
    
    # Calculate -log(p) for t-test scores
    epsilon = 1e-15
    t_test_scores = -np.log(t_test_pvals + epsilon)
    
    # Normalize each score type to [0,1] range
    t_test_scores = (t_test_scores - t_test_scores.min()) / (t_test_scores.max() - t_test_scores.min())
    mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
    rfe_scores = (rfe_importance - rfe_importance.min()) / (rfe_importance.max() - rfe_importance.min())
    
    # Combine scores using weighted average
    combined_scores = (
        weights[0] * t_test_scores +
        weights[1] * mi_scores +
        weights[2] * rfe_scores
    )
    
    return combined_scores

def objective_function(weights):

    w1, w2 = weights
    w3 = 1 - (w1 + w2)
    if w3 < 0:  # Invalid set if w3 is negative
        return 0
    
    weights = (w1, w2, w3)

    acc = 0
    embedding_filepaths = get_embeddings_filepaths()

    for fname in embedding_filepaths:
        embeddings_df = read_embeddings_df(fname)

        sentence1_df = pd.DataFrame({'embedding': embeddings_df['Sentence1_embedding'].tolist(), 'label': 0})
        sentence2_df = pd.DataFrame({'embedding': embeddings_df['Sentence2_embedding'].tolist(), 'label': 1})
        df = pd.concat([sentence1_df, sentence2_df], ignore_index=True)

        edi_scores = calculate_edi_scores(fname, weights)

        top_20_indices = np.argsort(edi_scores)[-20:]

        # Filter embeddings to keep only top 20 dimensions
        df['embedding'] = df['embedding'].apply(lambda x: np.array(x)[top_20_indices])

        X = np.array(df['embedding'].tolist())
        labels = np.array(df['label'].tolist())
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc += accuracy_score(y_test, y_pred)
    
    avg_acc = acc/len(embedding_filepaths)

    print(weights, avg_acc)

    return -avg_acc  # Maximize accuracy

def save_edi_scores(edi_scores, results_directory): 
    edi_df = pd.DataFrame({
        'Dimension': [i for i in range(len(edi_scores))],
        'EDI Score': edi_scores
    })

    edi_df.to_csv(os.path.join(results_directory, "edi_score.csv"), index=False)

    top_20_df = edi_df.nlargest(20, 'EDI Score')
    top_20_df.to_csv(os.path.join(results_directory, "top_20_edi_scores.csv"), index=False)

if __name__ == "__main__": 

    optimized_weights = (0.6675638190850008, 0.09635150783424912, 0.23608467308074998)

    # # Define search space for the first 3 weights
    # search_space = [Real(0, 1, name=f"w{i}") for i in range(2)]

    # # Optimize
    # result = gp_minimize(objective_function, search_space, n_calls=100)
    # optimized_weights = result.x + [1 - sum(result.x)]
    # print("Best weights:", optimized_weights)

    embedding_filepaths = get_embeddings_filepaths()

    for embeddings_csv in tqdm(embedding_filepaths):
        edi_scores = calculate_edi_scores(embeddings_csv, optimized_weights)
        results_directory = get_results_directory(embeddings_csv, "edi_scores")
        save_edi_scores(edi_scores, results_directory)


    

    