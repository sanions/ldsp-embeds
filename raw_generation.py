from utils import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def get_dimension_shifts(train_df):
    """
    Calculate the mean shifts for each dimension based on the training embeddings
    """
    e1 = np.array([np.array(d) for d in train_df['Sentence1_embedding']])
    e2 = np.array([np.array(d) for d in train_df['Sentence2_embedding']])
    
    # Calculate mean difference for each dimension
    mean_shifts = np.mean(e2, axis=0) - np.mean(e1, axis=0)
    return mean_shifts

def generate_shifted_embedding(embedding, dimension_shifts):
    """
    Generate new embedding by shifting all dimensions
    """
    embedding = np.array(embedding)
    return embedding + dimension_shifts

def get_dimension_transformations(train_df):
    """
    Calculate regression-based transformation for each important dimension using training data
    
    Args:
        train_df: DataFrame containing training data with Sentence1_embedding and Sentence2_embedding
    """
    # Convert embeddings to numpy arrays
    e1_train = np.stack(train_df['Sentence1_embedding'])
    e2_train = np.stack(train_df['Sentence2_embedding'])
    
    transformations = {}
    n_dimensions = e1_train.shape[1]
    
    for dim in range(n_dimensions):
        # Reshape for sklearn
        X_train = e1_train[:, dim].reshape(-1, 1)
        y_train = e2_train[:, dim]
        
        # Fit linear regression on training data
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        transformations[dim] = model
    
    return transformations


def generate_regression_embedding(embedding, transformations):
    """
    Generate new embedding by applying regression transformations to important dimensions
    """
    embedding = np.array(embedding)
    for dim, model in transformations.items():
        # Apply transformation to dimension
        embedding[dim] = model.predict(np.array([embedding[dim]]).reshape(-1, 1))[0]
    return embedding


if __name__ == "__main__":
    embedding_filepaths = get_embeddings_filepaths()
    
    for embeddings_csv in tqdm(embedding_filepaths):
        # Load data
        df = read_embeddings_df(embeddings_csv)
        
        # Split data into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Calculate shifts using only training data
        shifts = get_dimension_shifts(train_df)
        transformations = get_dimension_transformations(train_df)
        
        # Generate new embeddings for test set by shifting important dimensions
        test_df['mean_shift_embedding'] = test_df['Sentence1_embedding'].apply(
            lambda x: generate_shifted_embedding(x, shifts)
        )

        test_df['regression_embedding'] = test_df['Sentence1_embedding'].apply(
            lambda x: generate_regression_embedding(x, transformations)
        )
        
        # Save results
        results_dir = get_results_directory(embeddings_csv, "generation")
        output_path = os.path.join(results_dir, "generated_embeddings.csv")
        test_df.to_csv(output_path, index=False)
        

    print("Embedding generation complete")