import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_embeddings(true_embeddings, generated_embeddings, baseline_embeddings=None):
    """
    Evaluate the quality of generated embeddings using cosine similarity
    
    Args:
        true_embeddings: Array of true sentence2 embeddings
        generated_embeddings: Array of generated embeddings
        baseline_embeddings: Array of baseline embeddings (sentence1) for comparison
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    # Convert embeddings to numpy arrays if they aren't already
    true_embeddings = np.array([np.array(e) for e in true_embeddings])
    generated_embeddings = np.array([np.array(e) for e in generated_embeddings])
    
    # Calculate cosine similarity for generated embeddings
    similarities = [cosine_similarity(t.reshape(1, -1), g.reshape(1, -1))[0][0] 
                   for t, g in zip(true_embeddings, generated_embeddings)]
    
    metrics = {
        'mean_similarity': np.mean(similarities),
        'median_similarity': np.median(similarities),
        'std_similarity': np.std(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
        'num_pairs': len(similarities)
    }
    
    # Calculate baseline similarities if provided
    if baseline_embeddings is not None:
        baseline_embeddings = np.array([np.array(e) for e in baseline_embeddings])
        baseline_similarities = [cosine_similarity(t.reshape(1, -1), b.reshape(1, -1))[0][0] 
                               for t, b in zip(true_embeddings, baseline_embeddings)]
        
        metrics.update({
            'baseline_mean_similarity': np.mean(baseline_similarities),
            'baseline_median_similarity': np.median(baseline_similarities),
            'baseline_std_similarity': np.std(baseline_similarities),
            'baseline_min_similarity': np.min(baseline_similarities),
            'baseline_max_similarity': np.max(baseline_similarities),
            'improvement_over_baseline': np.mean(similarities) - np.mean(baseline_similarities)
        })
        return metrics, similarities, baseline_similarities
    
    return metrics, similarities

if __name__ == "__main__":
    embedding_filepaths = get_embeddings_filepaths()
    
    all_results = []
    
    for embeddings_csv in tqdm(embedding_filepaths):
        # Load generated test embeddings
        results_dir = get_results_directory(embeddings_csv, "generation")
        test_df = read_embeddings_df(os.path.join(results_dir, "generated_embeddings.csv"))
        
        # Evaluate mean-shifted embeddings with baseline comparison
        mean_shift_metrics, mean_shift_similarities, baseline_similarities = evaluate_embeddings(
            test_df['Sentence2_embedding'],
            test_df['mean_shift_embedding'],
            test_df['Sentence1_embedding']
        )

        # Evaluate mean-shifted embeddings with baseline comparison
        regression_metrics, regression_similarities, _ = evaluate_embeddings(
            test_df['Sentence2_embedding'],
            test_df['regression_embedding'],
            test_df['Sentence1_embedding']
        )

        # Combine metrics with prefixes to distinguish between methods
        metrics = {}
        for key, value in mean_shift_metrics.items():
            if key.startswith('baseline'):
                metrics[key] = value  # Keep baseline metrics as is (only need once)
            else:
                metrics[f'mean_shift_{key}'] = value
        
        for key, value in regression_metrics.items():
            if not key.startswith('baseline'):  # Skip baseline metrics as we already have them
                metrics[f'regression_{key}'] = value
        
        # Add dataset info to metrics
        metrics['dataset'] = os.path.basename(embeddings_csv)
        all_results.append(metrics)
        
        # Save detailed similarities for this dataset
        similarities_df = pd.DataFrame({
            'baseline_similarity': baseline_similarities,
            'mean_shift_similarity': mean_shift_similarities,
            'regression_similarity': regression_similarities,
            'Sentence1': test_df['Sentence1'],
            'Sentence2': test_df['Sentence2']
        })
        
        # Create box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=similarities_df[['baseline_similarity', 'mean_shift_similarity', 'regression_similarity']])
        plt.title('Comparison of Embedding Similarities Across Methods')
        plt.xlabel('Method')
        plt.ylabel('Cosine Similarity')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "similarity_comparison_boxplot.png"))
        plt.close()
        
        # Save detailed similarities
        similarities_df.to_csv(os.path.join(results_dir, "embedding_similarities.csv"), index=False)

        break
    
    # Save overall results
    results_df = pd.DataFrame(all_results)
    print("\nOverall Results:")
    print(results_df.to_string(index=False))
    results_df.to_csv("generation_evaluation_results.csv", index=False)
