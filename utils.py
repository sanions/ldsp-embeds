import pandas as pd
import os
import numpy as np

def get_dataset_filepaths():
    directory = "./datasets"
    csv_filenames = []

    for file_name in os.listdir(directory):
        if not file_name.endswith(".csv"):
            continue

        csv_filenames.append(os.path.join(directory, file_name))

    return csv_filenames 

def get_embeddings_filepaths():
    directory = "./embeddings"
    return [os.path.join(directory, fn) for fn in os.listdir(directory) if "gender" not in fn and "modality" not in fn and "spatial" not in fn and "subject" not in fn]

def get_linguistic_property(embeddings_csv):
    return embeddings_csv.split("_")[0]


def get_results_directory(embeddings_csv, metric):
    p = os.path.join("results", get_linguistic_property(os.path.basename(embeddings_csv)), metric)

    if not os.path.exists(p):
        os.makedirs(p)
    
    return p


def read_embeddings_df(embeddings_csv):

    embeddings_df = pd.read_csv(embeddings_csv)
    
    for col in embeddings_df.columns:
        if col.endswith("_embedding"):
            embeddings_df[col] = embeddings_df[col].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    
    return embeddings_df


def get_edi_scores(embeddings_csv):
    """
    Get the EDI scores for the current linguistic property from the saved results.
    
    Returns:
        numpy.ndarray: Array of EDI scores for each dimension
    """
    
    # Construct path to EDI scores
    edi_scores_path = os.path.join(
        get_results_directory(embeddings_csv, "edi_scores"),
        "edi_score.csv"
    )
    
    # Load and return EDI scores
    edi_df = pd.read_csv(edi_scores_path)
    return edi_df['EDI Score'].values