import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class EDIWeightedTransformationLoss:
    def __init__(self, edi_scores, scaling_method='sigmoid'):
        """
        Initialize a loss function that weights dimension changes based on EDI scores
        
        Args:
            edi_scores (torch.Tensor or np.ndarray): Embedding Dimension Importance scores
            scaling_method (str): Method to scale EDI scores ('sigmoid', 'softmax', 'power')
        """
        # Convert to torch tensor if not already
        self.edi_scores = torch.tensor(edi_scores, dtype=torch.float32)
        
        # Choose scaling method
        self.scaling_method = scaling_method
    

    def _scale_edi_weights(self):
        """
        Scale EDI scores using different methods
        
        Returns:
            torch.Tensor: Scaled weights for each dimension
        """
        if self.scaling_method == 'sigmoid':
            # Sigmoid scaling: pushes values to extremes
            return torch.sigmoid(5 * (self.edi_scores - 0.5))
        
        elif self.scaling_method == 'softmax':
            # Softmax scaling: relative importance
            return F.softmax(self.edi_scores * 10, dim=0)
        
        elif self.scaling_method == 'power':
            # Power law scaling: emphasizes high scores
            return torch.pow(self.edi_scores, 2)
        
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def compute_transformation_loss(self, 
                                    original_embedding, 
                                    transformed_embedding, 
                                    similarity_type='cosine'):
        """
        Compute a loss that penalizes changes to dimensions based on their EDI scores
        
        Args:
            original_embedding (torch.Tensor): Original sentence 1 embedding
            transformed_embedding (torch.Tensor): Transformed sentence 2 embedding
            similarity_type (str): Type of similarity measure to use
        
        Returns:
            torch.Tensor: Computed loss
        """
        # Get dimension-specific weights based on EDI scores
        edi_weights = self._scale_edi_weights()
        
        # Compute per-dimension change
        dimension_changes = torch.abs(original_embedding - transformed_embedding)

        # Weight dimension changes by EDI scores, invert the weights so high EDI scores lead to less penalty
        weighted_changes = dimension_changes * (1 - edi_weights)
        
        # Compute overall transformation loss
        if similarity_type == 'cosine':
            # Cosine similarity-based loss
            cosine_similarity = F.cosine_similarity(
                original_embedding.unsqueeze(0), 
                transformed_embedding.unsqueeze(0)
            ).mean()
            
            # Combination of weighted changes and cosine similarity
            transformation_loss = (
                weighted_changes.mean() * (1 - cosine_similarity)
            )
        
        elif similarity_type == 'mse':
            # Mean Squared Error with EDI-weighted dimensions
            transformation_loss = (
                (weighted_changes ** 2).mean()
            )
        
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
        
        return transformation_loss
    
    
    def compute_contrastive_loss(self, 
                               original_embedding, 
                               transformed_embedding, 
                               negative_embedding,
                               margin=1.0):
        """
        Compute contrastive loss to maximize distance from negative examples
        while minimizing distance from the original embedding
        
        Args:
            original_embedding (torch.Tensor): Original sentence embedding
            transformed_embedding (torch.Tensor): Transformed sentence embedding
            negative_embedding (torch.Tensor): Random different sentence embedding
            margin (float): Margin for negative examples
            
        Returns:
            torch.Tensor: Contrastive loss
        """

        transformed_embedding = transformed_embedding.unsqueeze(0)

        # Positive pair loss (minimize distance to original)
        pos_distance = (1 - F.cosine_similarity(transformed_embedding, original_embedding.unsqueeze(0))).mean()
        
        # Negative pair loss (maximize distance to negative, with margin)
        neg_distance = (1 - F.cosine_similarity(transformed_embedding, negative_embedding.unsqueeze(0))).mean()
        neg_loss = torch.clamp(margin - neg_distance, min=0.0)
        
        return pos_distance + neg_loss
    
    def combined_loss(
        self, 
        original_embedding, 
        transformed_embedding, 
        negative_embedding=None,
        transformation_weight=1.0,
        contrastive_weight=0.5
    ):
        """
        Combine transformation and contrastive losses
        
        Args:
            original_embedding (torch.Tensor): Original sentence 1 embedding
            transformed_embedding (torch.Tensor): Transformed sentence 2 embedding
            negative_embedding (torch.Tensor, optional): Random different sentence embedding
            transformation_weight (float): Weight for transformation loss
            contrastive_weight (float): Weight for contrastive loss
        
        Returns:
            torch.Tensor: Combined loss
        """
        # Compute transformation loss
        transformation_loss = self.compute_transformation_loss(
            original_embedding, 
            transformed_embedding
        )
        
        total_loss = transformation_weight * transformation_loss
        
        # Add contrastive loss if negative example is provided
        if negative_embedding is not None:
            contrastive_loss = self.compute_contrastive_loss(
                original_embedding,
                transformed_embedding,
                negative_embedding
            )
            total_loss += contrastive_weight * contrastive_loss
            
    
        return total_loss


def train_edi_generator(train_embeddings1, train_embeddings2, edi_scores, n_epochs=50, batch_size=32, lr=0.001):
    """
    Train a generator model using EDI-weighted loss
    
    Args:
        train_embeddings1 (np.ndarray): Array of sentence 1 embeddings
        train_embeddings2 (np.ndarray): Array of sentence 2 embeddings
        edi_scores (np.ndarray): Array of EDI scores
        n_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        lr (float): Learning rate
    
    Returns:
        nn.Module: Trained generator model
    """
    # Convert to torch tensors
    train_e1 = torch.tensor(train_embeddings1, dtype=torch.float32)
    train_e2 = torch.tensor(train_embeddings2, dtype=torch.float32)
    
    # Initialize generator
    input_dim = train_embeddings1.shape[1]
    generator = nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ReLU(),
        nn.Linear(input_dim, input_dim)
    )
    
    # Initialize loss and optimizer
    edi_loss = EDIWeightedTransformationLoss(edi_scores, scaling_method='sigmoid')
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    
    # Training loop
    generator.train()
    n_samples = len(train_e1)
    
    for epoch in range(n_epochs):
        total_loss = 0
        for i in range(0, n_samples, batch_size):
            batch_e1 = train_e1[i:i+batch_size]
            batch_e2 = train_e2[i:i+batch_size]
            
            # Generate negative examples by shuffling the batch
            negative_indices = torch.randperm(len(batch_e1))
            batch_negative = batch_e1[negative_indices]
            
            optimizer.zero_grad()
            
            # Generate transformed embeddings
            transformed_embeddings = generator(batch_e1)
            
            # Compute batch loss with negative examples
            batch_loss = torch.stack([
                edi_loss.combined_loss(e1, e2, neg)
                for e1, e2, neg in zip(transformed_embeddings, batch_e2, batch_negative)
            ]).mean()
            
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.4f}")
    
    return generator


def generate_s2_embedding(s1_embedding, generator):
    """
    Generate new embedding using trained EDI generator
    
    Args:
        embedding (np.ndarray): Input embedding to transform
        generator (nn.Module): Trained generator model
    
    Returns:
        np.ndarray: Transformed embedding
    """
    # Convert to torch tensor
    embedding_tensor = torch.tensor(s1_embedding, dtype=torch.float32)
    
    # Generate transformed embedding
    generator.eval()
    with torch.no_grad():
        transformed_embedding = generator(embedding_tensor)
    
    return transformed_embedding.numpy()