import numpy as np
from sklearn.metrics import mutual_info_score
import pandas as pd
from utils import *
from tqdm import tqdm

def calculate_mutual_information(embeddings, labels):

    n_samples, n_dimensions = embeddings.shape
    mutual_informations = np.zeros(n_dimensions)

    for dim in range(n_dimensions):

        embedding_dim = embeddings[:, dim]

        bins = np.quantile(embedding_dim, np.linspace(0, 1, 10)) # 10 bins
        discretized_embedding = np.digitize(embedding_dim, bins)

        mutual_informations[dim] = mutual_info_score(labels, discretized_embedding)

    return mutual_informations


def process_mutual_information(dataset_csv):

    df = pd.read_csv(dataset_csv)

    all_embeddings = np.concatenate([np.array(df['Sentence1_embedding'].to_list()),
                                 np.array(df['Sentence2_embedding'].to_list())], axis=0)
    
    all_labels = np.concatenate([np.zeros(len(df)), np.ones(len(df))], axis=0)

    all_embeddings = all_embeddings.reshape(len(all_embeddings), -1)

    # Calculate mutual information
    mutual_informations = calculate_mutual_information(all_embeddings, all_labels)


if __name__ == "__main__":

    dataset_filepaths = get_dataset_filepaths()

    for dataset_csv in tqdm(dataset_filepaths):
    #    process_mutual_information(dataset_csv)
        os.makedirs(get_results_filepath(dataset_csv, "mi"))


       

    




