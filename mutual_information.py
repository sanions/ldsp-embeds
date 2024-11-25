import numpy as np
from sklearn.metrics import mutual_info_score
import pandas as pd
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_mutual_information(embeddings, labels):

    n_samples, n_dimensions = embeddings.shape
    mutual_informations = np.zeros(n_dimensions)

    for dim in range(n_dimensions):

        embedding_dim = embeddings[:, dim]

        bins = np.quantile(embedding_dim, np.linspace(0, 1, 10)) # 10 bins
        discretized_embedding = np.digitize(embedding_dim, bins)

        mutual_informations[dim] = mutual_info_score(labels, discretized_embedding)

    return mutual_informations


def process_mutual_information(embeddings_df):

    all_embeddings = np.concatenate([np.array(embeddings_df['Sentence1_embedding'].to_list()),
                                 np.array(embeddings_df['Sentence2_embedding'].to_list())], axis=0)
    
    all_labels = np.concatenate([np.zeros(len(embeddings_df)), np.ones(len(embeddings_df))], axis=0)

    all_embeddings = all_embeddings.reshape(len(all_embeddings), -1)

    # Calculate mutual information
    mutual_informations = calculate_mutual_information(all_embeddings, all_labels)

    # Find the most important dimensions
    most_important_dimensions = np.argsort(mutual_informations)[::-1]  # Descending order
    top_10_dimensions = most_important_dimensions[:10]
    top_10_mutual_informations = mutual_informations[top_10_dimensions]

    # Choose the top N dimensions
    
    top_N_dimensions = most_important_dimensions[:N]
    top_N_mutual_informations = mutual_informations[top_N_dimensions]

    return mutual_informations, top_10_dimensions, top_10_mutual_informations, top_N_dimensions, top_N_mutual_informations



def save_results_and_plot(mutual_informations, top_10_dimensions, top_10_mutual_informations, top_N_dimensions, top_N_mutual_informations, results_directory):

    txt_filepath = os.path.join(results_directory, "mutual_information_results.txt")
    
    with open(txt_filepath, "w") as f:
        f.write("Top 10 Dimensions with Highest Mutual Information:\n")
        for dim, mi in zip(top_10_dimensions, top_10_mutual_informations):
            f.write(f"Dimension {dim + 1}: {mi}\n")
        f.write("\n")

        f.write("All Mutual Information Values:\n")
        for i, mi in enumerate(mutual_informations):
            f.write(f"Dimension {i + 1}: {mi}\n")

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(N), top_N_mutual_informations)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Mutual Information")
    plt.title(f"Top {N} Dimensions with Highest Mutual Information")
    plt.xticks(np.arange(N), top_N_dimensions + 1) 
    plt.tight_layout()

    plt.savefig(os.path.join(results_directory, "topN_mutual_information.png"))
    plt.close()

if __name__ == "__main__":

    embeddings_filepaths = get_embeddings_filepaths()
    N = 20

    for embeddings_csv in tqdm(embeddings_filepaths):
       embeddings_df = read_embeddings_df(embeddings_csv)

       mutual_informations, top_10_dimensions, top_10_mutual_informations, top_N_dimensions, top_N_mutual_informations = process_mutual_information(embeddings_df)
       results_directory = get_results_directory(embeddings_csv, "mutual_information")
       save_results_and_plot(mutual_informations, top_10_dimensions, top_10_mutual_informations, top_N_dimensions, top_N_mutual_informations, results_directory)


    print("Mutual Information Calculations Complete")
        


       

    




