from utils import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from custom_loss_generation import train_edi_generator, generate_s2_embedding


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

def get_dimension_distributions(train_df):
    """
    Calculate the mean and std for each dimension of Sentence2 embeddings
    """
    e2 = np.stack(train_df['Sentence2_embedding'])
    means = np.mean(e2, axis=0)
    stds = np.std(e2, axis=0)
    return means, stds

def generate_sampling_embedding(embedding, edi_scores, s2_means, s2_stds, threshold=0.75):
    """
    Generate new embedding by:
    - Keeping dimensions with EDI < threshold unchanged
    - Sampling from Sentence2 distribution for dimensions with EDI >= threshold
    
    Args:
        embedding: Original embedding to transform
        edi_scores: Array of EDI scores for each dimension
        s2_means: Mean values for each dimension in Sentence2
        s2_stds: Standard deviations for each dimension in Sentence2
        threshold: EDI score threshold (default: 0.75)
    """
    embedding = np.array(embedding)
    high_edi_dims = edi_scores >= threshold
    
    # Generate random samples for high EDI dimensions
    n_dims = len(high_edi_dims)
    random_samples = np.random.normal(
        loc=s2_means[high_edi_dims],
        scale=s2_stds[high_edi_dims],
        size=np.sum(high_edi_dims)
    )
    
    # Create new embedding
    new_embedding = embedding.copy()
    new_embedding[high_edi_dims] = random_samples
    return new_embedding

if __name__ == "__main__":
    embedding_filepaths = get_embeddings_filepaths()
    
    for embeddings_csv in tqdm(embedding_filepaths):
        # Load data
        df = read_embeddings_df(embeddings_csv)
        
        # Split data into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # # Calculate transformations using only training data
        # shifts = get_dimension_shifts(train_df)
        # transformations = get_dimension_transformations(train_df)
        # s2_means, s2_stds = get_dimension_distributions(train_df)
        
        # Load EDI scores
        edi_scores = get_edi_scores(embeddings_csv)
        
        # Train EDI generator
        train_e1 = np.stack(train_df['Sentence1_embedding'])
        train_e2 = np.stack(train_df['Sentence2_embedding'])

        generator = train_edi_generator(train_e1, train_e2, edi_scores, n_epochs=200)
        
        # # Generate new embeddings for test set using different methods
        # test_df['mean_shift_embedding'] = test_df['Sentence1_embedding'].apply(
        #     lambda x: generate_shifted_embedding(x, shifts)
        # )
        
        # test_df['regression_embedding'] = test_df['Sentence1_embedding'].apply(
        #     lambda x: generate_regression_embedding(x, transformations)
        # )
        
        # test_df['sampling_embedding'] = test_df['Sentence1_embedding'].apply(
        #     lambda x: generate_sampling_embedding(x, edi_scores, s2_means, s2_stds)
        # )
        
        # test_df['custom_loss_embedding'] = test_df['Sentence1_embedding'].apply(
        #     lambda x: generate_s2_embedding(x, generator)
        # )   

        # test_df['contrastive_loss_embedding'] = test_df['Sentence1_embedding'].apply(
        #     lambda x: generate_s2_embedding(x, generator)
        # )  

        test_df['contrastive_cosine_embedding'] = test_df['Sentence1_embedding'].apply(
            lambda x: generate_s2_embedding(x, generator)
        )       
        # Save results
        results_dir = get_results_directory(embeddings_csv, "generation")
        output_path = os.path.join(results_dir, "generated_embeddings_contrastive_cosine.csv")
        test_df.to_csv(output_path, index=False)


    print("Embedding generation complete")