import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean

def evaluate_embeddings(true_embeddings, generated_embeddings, baseline_embeddings=None, metric='cosine'):
    """
    Evaluate the quality of generated embeddings using cosine similarity or euclidean distance
    
    Args:
        true_embeddings: Array of true sentence2 embeddings
        generated_embeddings: Array of generated embeddings
        baseline_embeddings: Array of baseline embeddings (sentence1) for comparison
        metric: 'cosine' or 'euclidean' (default: 'cosine')
    """
    true_embeddings = np.array([np.array(e) for e in true_embeddings])
    generated_embeddings = np.array([np.array(e) for e in generated_embeddings])
    
    if metric == 'cosine':
        similarities = [cosine_similarity(t.reshape(1, -1), g.reshape(1, -1))[0][0] 
                       for t, g in zip(true_embeddings, generated_embeddings)]
    else:  # euclidean
        similarities = [euclidean(t, g) for t, g in zip(true_embeddings, generated_embeddings)]
    
    # Calculate baseline similarities if provided
    if baseline_embeddings is not None:
        baseline_embeddings = np.array([np.array(e) for e in baseline_embeddings])
        if metric == 'cosine':
            baseline_similarities = [cosine_similarity(t.reshape(1, -1), b.reshape(1, -1))[0][0] 
                                   for t, b in zip(true_embeddings, baseline_embeddings)]
        else:  # euclidean
            baseline_similarities = [euclidean(t, b) for t, b in zip(true_embeddings, baseline_embeddings)]
        return similarities, baseline_similarities
    
    return similarities

def plot_comparison_boxplots(similarities_df, results_dir, metric='cosine'): 
    lp = results_dir.split('/')[1]
    results_dir = os.path.join(results_dir, "plots")
    plt.figure(figsize=(15, 6))
    
    columns = ['baseline_similarity', 'sampling_similarity', 'mean_shift_similarity', 
               'regression_similarity', 'transformative_loss_similarity', 
               'contrastive_loss_similarity', 'contrastive_cosine_similarity']

    # Define readable labels for each method
    labels = {
        'baseline_similarity': 'Baseline',
        'sampling_similarity': 'Sampling',
        'mean_shift_similarity': 'Mean Shift',
        'regression_similarity': 'Regression',
        'transformative_loss_similarity': 'EDI + Cosine Loss',
        'contrastive_loss_similarity': 'Constrastive MSE Loss',
        'contrastive_cosine_similarity': 'Contrastive Cosine Loss'
    }

    # Create a copy of the data with renamed columns
    plot_data = similarities_df[columns].copy()
    plot_data.columns = [labels[col] for col in columns]

    
    sns.boxplot(data=plot_data)
    metric_name = 'Cosine Similarity' if metric == 'cosine' else 'Euclidean Distance'
    # plt.title(f'Comparison of Embedding {metric_name} Across Methods: {lp}')
    # plt.xlabel('Method')
    # plt.ylabel(metric_name)
    plt.xticks([])
    
    # Invert y-axis for euclidean distance to show better (lower) values at top
    if metric == 'euclidean':
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{metric}_comparison.svg"))
    plt.savefig(os.path.join(results_dir, f"{metric}_comparison.png"))
    plt.close()

def create_mse_table(similarities_df, results_dir, all_results):
    """
    Create a table comparing MSE across methods and linguistic properties
    """
    columns = ['baseline_similarity', 'sampling_similarity', 'mean_shift_similarity', 
               'regression_similarity', 'transformative_loss_similarity', 
               'contrastive_loss_similarity', 'contrastive_cosine_similarity']

    labels = {
        'baseline_similarity': 'Baseline',
        'sampling_similarity': 'Sampling',
        'mean_shift_similarity': 'Mean Shift',
        'regression_similarity': 'Regression',
        'transformative_loss_similarity': 'EDI + Cosine Loss',
        'contrastive_loss_similarity': 'Constrastive MSE Loss',
        'contrastive_cosine_similarity': 'Contrastive Cosine Loss'
    }
    
    # Calculate MSE for current linguistic property
    mse_scores = {labels[col]: np.mean(np.square(similarities_df[col])) for col in columns}
    
    # Get linguistic property from results directory
    lp = results_dir.split('/')[1]
    
    # Add to results list
    all_results.append({
        'linguistic_property': lp,
        **mse_scores
    })
    
    # If this is the last linguistic property, create and save the table
    if len(all_results) == len(get_embeddings_filepaths()):
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        results_df.set_index('linguistic_property', inplace=True)
        
        # Format numbers to 4 decimal places
        results_df = results_df.round(4)
        
        # Save as CSV
        results_df.to_csv("results/mse_comparison_table.csv")
    

if __name__ == "__main__":
    embedding_filepaths = get_embeddings_filepaths()
    
    all_results = []
    
    for embeddings_csv in tqdm(embedding_filepaths):
        # Load generated test embeddings
        results_dir = get_results_directory(embeddings_csv, "generation")
        tables_dir = os.path.join(results_dir, "tables")
        test_df = read_embeddings_df(os.path.join(tables_dir, "generated_embeddings.csv"))

        contrastive_df = read_embeddings_df(os.path.join(tables_dir, "generated_embeddings_contrastive.csv"))
        cosine_df = read_embeddings_df(os.path.join(tables_dir, "generated_embeddings_contrastive_cosine.csv"))

        test_df = pd.concat([test_df, contrastive_df['contrastive_loss_embedding'], cosine_df['contrastive_cosine_embedding']], axis=1)
        
        # Evaluate mean-shifted embeddings with baseline comparison
        mean_shift_similarities, baseline_similarities = evaluate_embeddings(
            test_df['Sentence2_embedding'],
            test_df['mean_shift_embedding'],
            test_df['Sentence1_embedding']
        )

        # Evaluate regression embeddings 
        regression_similarities = evaluate_embeddings(
            test_df['Sentence2_embedding'],
            test_df['regression_embedding']
        )

        # Evaluate sampling-based embeddings 
        sampling_similarities = evaluate_embeddings(
            test_df['Sentence2_embedding'],
            test_df['sampling_embedding']
        )

        # Evaluate custom loss embeddings 
        custom_similarities = evaluate_embeddings(
            test_df['Sentence2_embedding'],
            test_df['custom_loss_embedding']
        )

        # Evaluate contrastive loss-based embeddings with baseline comparison
        contrastive_similarities = evaluate_embeddings(
            test_df['Sentence2_embedding'],
            test_df['contrastive_loss_embedding']
        )

        # Evaluate contrastive cosine loss-based embeddings with baseline comparison
        cosine_similarities = evaluate_embeddings(
            test_df['Sentence2_embedding'],
            test_df['contrastive_cosine_embedding']
        )

        # Save similarities
        similarities_df = pd.DataFrame({
            'baseline_similarity': baseline_similarities,
            'mean_shift_similarity': mean_shift_similarities,
            'regression_similarity': regression_similarities,
            'sampling_similarity': sampling_similarities,
            'transformative_loss_similarity': custom_similarities,
            'contrastive_loss_similarity': contrastive_similarities,
            'contrastive_cosine_similarity': cosine_similarities,
            'Sentence1': test_df['Sentence1'],
            'Sentence2': test_df['Sentence2']
        })

        plot_comparison_boxplots(similarities_df, results_dir)
        
        # Save detailed similarities
        similarities_df.to_csv(os.path.join(tables_dir, "embedding_similarities.csv"), index=False)

        # Evaluate with Euclidean distance
        euclidean_similarities_df = pd.DataFrame({
            'baseline_similarity': evaluate_embeddings(test_df['Sentence2_embedding'], 
                                                    test_df['Sentence1_embedding'], metric='euclidean'),
            'mean_shift_similarity': evaluate_embeddings(test_df['Sentence2_embedding'], 
                                                       test_df['mean_shift_embedding'], metric='euclidean'),
            'regression_similarity': evaluate_embeddings(test_df['Sentence2_embedding'], 
                                                      test_df['regression_embedding'], metric='euclidean'),
            'sampling_similarity': evaluate_embeddings(test_df['Sentence2_embedding'], 
                                                    test_df['sampling_embedding'], metric='euclidean'),
            'transformative_loss_similarity': evaluate_embeddings(test_df['Sentence2_embedding'], 
                                                               test_df['custom_loss_embedding'], metric='euclidean'),
            'contrastive_loss_similarity': evaluate_embeddings(test_df['Sentence2_embedding'], 
                                                            test_df['contrastive_loss_embedding'], metric='euclidean'),
            'contrastive_cosine_similarity': evaluate_embeddings(test_df['Sentence2_embedding'], 
                                                              test_df['contrastive_cosine_embedding'], metric='euclidean'),
            'Sentence1': test_df['Sentence1'],
            'Sentence2': test_df['Sentence2']
        })
        
        # Plot and save Euclidean distance comparison
        plot_comparison_boxplots(euclidean_similarities_df, results_dir, metric='euclidean')
        euclidean_similarities_df.to_csv(os.path.join(tables_dir, "embedding_euclidean_distances.csv"), index=False)

        # Add MSE scores to results table
        create_mse_table(euclidean_similarities_df, results_dir, all_results)

