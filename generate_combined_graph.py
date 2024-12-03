import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utils import *


def create_combined_graph(mutual_info_df, ttest_results_df, clf_weights_df, pca_contribs_df, results_directory, N=20):
   
    mutual_informations = mutual_info_df['Mutual_Information'].values

    top_N_ttest_dimensions = ttest_results_df.nsmallest(N, 'p_value')['dimension'].values

    top_N_clf_dimensions = clf_weights_df.nlargest(N, 'Weight')['Dimension'].values

    top_N_pca_contribs = pca_contribs_df.nlargest(N, 'Contribution')['Dimension'].values

    plt.figure(figsize=(12, 6))

    plt.bar(np.arange(len(mutual_informations)), mutual_informations, label='Mutual Information', alpha=0.5)

    threshold = np.sort(mutual_informations)[-N]
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Top {N} Threshold')

    plt.scatter(top_N_ttest_dimensions, mutual_informations[top_N_ttest_dimensions], color='red', marker='s', label=f'Top {N} (t-test)', s=50, alpha=0.4)
    plt.scatter(top_N_clf_dimensions, mutual_informations[top_N_clf_dimensions], color='blue', label=f'Top {N} (Logistic Reg.)', s=50, alpha=0.4)
    plt.scatter(top_N_pca_contribs, mutual_informations[top_N_pca_contribs], color='green', marker='^', label=f'Top {N} (PCA)', s=50, alpha=0.8)

    plt.xlabel("Embedding Dimension")
    plt.ylabel("Mutual Information")
    plt.title(f"Mutual Information of Embedding Dimensions")
    plt.legend()

    graph_filepath = os.path.join(results_directory, "combined_graph.png")
    plt.tight_layout()
    plt.savefig(graph_filepath)
    plt.close()

    print(f"Combined graph saved at: {graph_filepath}")


if __name__ == "__main__":

    embedding_filepaths = get_embeddings_filepaths()

    for embeddings_csv in tqdm(embedding_filepaths):
        results_directory = get_results_directory(embeddings_csv, "combined_analysis")

        mutual_info_df = pd.read_csv(os.path.join(get_results_directory(embeddings_csv, "mutual_information"), "mutual_information_all.csv"))
        ttest_results_df = pd.read_csv(os.path.join(get_results_directory(embeddings_csv, "t_test_analysis"), "t_test_results.csv"))
        clf_weights_df = pd.read_csv(os.path.join(get_results_directory(embeddings_csv, "logistic_classifier"), "classifier_weights.csv"))
        pca_contribs_df = pd.read_csv(os.path.join(get_results_directory(embeddings_csv, "pca_analysis"), "pca_all.csv"))

        create_combined_graph(mutual_info_df, ttest_results_df, clf_weights_df, pca_contribs_df, results_directory, N=20)

