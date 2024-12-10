from utils import *
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def create_edi_comparison_table(embedding_dim):
    # Define the linguistic properties and their file paths
    properties = {
        'Control': 'results/control/edi_scores/edi_score.csv',
        'Negation': 'results/negation/edi_scores/edi_score.csv',
        'Intensifier': 'results/intensifier/edi_scores/edi_score.csv',
        'Tense': 'results/tense/edi_scores/edi_score.csv',
        'Voice': 'results/voice/edi_scores/edi_score.csv',
        'Polarity': 'results/polarity/edi_scores/edi_score.csv',
        'Quantity': 'results/quantity/edi_scores/edi_score.csv',
        'Factuality': 'results/factuality/edi_scores/edi_score.csv',
        'Definiteness': 'results/definiteness/edi_scores/edi_score.csv',
        'Synonym': 'results/synonym/edi_scores/edi_score.csv'
    }
    
    # Initialize an empty DataFrame to store all scores
    all_scores = pd.DataFrame(columns=range(embedding_dim))
    
    # Read each file and add it to the DataFrame
    for property_name, filepath in properties.items():
        df = pd.read_csv(filepath)
        # Set property name as the row name for these scores
        all_scores.loc[property_name] = df['EDI Score'].values
        
    # Rename columns to be dimension numbers
    all_scores.columns = [f'Dim_{i}' for i in range(len(all_scores.columns))]
    
    # Save to CSV
    output_path = 'results/combined_edi_scores.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_scores.to_csv(output_path)
    
    return all_scores

def create_highlighted_heatmap(df):
    # Create a figure with a larger size
    plt.figure(figsize=(20, 8))
    
    # Create the base heatmap
    sns.heatmap(df, 
                cmap='YlOrRd',
                xticklabels=False,  # Hide x-axis labels for clarity
                yticklabels=True,
                cbar_kws={'label': 'EDI Score'})
    
    plt.title('EDI Scores Across Dimensions')
    plt.xlabel('Dimensions (0-767)')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/edi_scores_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_colored_top_values(df, threshold=0.675):
    # Create a figure with a larger size
    plt.figure(figsize=(20, 14))
    
    # Define distinct colors for each property
    colors = [
        '#FF5252',  # Red
        '#FF7B52',  # Red-Orange
        '#FFB347',  # Orange
        '#FFEB3B',  # Yellow
        '#9CCC65',  # Yellow-Green
        '#66BB6A',  # Green
        '#4DB6AC',  # Blue-Green
        '#4FC3F7',  # Light Blue
        '#5C6BC0',  # Blue
        '#7E57C2',  # Blue-Violet
        '#D81B60'   # Pink-Violet
    ]
    
    # Create the base white plot
    plt.imshow(np.zeros_like(df), cmap='binary', aspect='auto')
    
    # For each row (property), highlight values above threshold
    for idx, (property_name, row) in enumerate(df.iterrows()):
        # Get indices and values above threshold
        high_value_idx = row[row > threshold].index
        high_values = row[row > threshold]
        
        # Convert string indices to integers
        x_positions = [int(col.split('_')[1]) for col in high_value_idx]
        y_positions = [idx] * len(x_positions)
        
        # Plot colored dots for high values
        plt.scatter(x_positions, y_positions, c=[colors[idx]], 
                   s=100, label=f'{property_name} ({len(high_values)} dims)')
    
    # Customize the plot
    plt.yticks(range(len(df)), df.index)
    plt.xlabel('Dimensions (0-767)')
    plt.title(f'EDI Scores Above {threshold} for Each Linguistic Property')
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.01, 1), 
              loc='upper left', 
              fontsize=12,
              markerscale=2,
              borderpad=1,
              labelspacing=1)
    
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('results/colored_top_edi_scores.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    

def create_colored_grid(df, threshold=0.675):
    # Create a figure with a larger size
    plt.figure(figsize=(20, 12))
    
    # Define distinct colors for each property
    colors = [
        '#FF5252',  # Red
        '#FF7B52',  # Red-Orange
        '#FFB347',  # Orange
        '#FFEB3B',  # Yellow
        '#9CCC65',  # Yellow-Green
        '#66BB6A',  # Green
        '#4DB6AC',  # Blue-Green
        '#4FC3F7',  # Light Blue
        '#5C6BC0',  # Blue
        '#7E57C2',  # Blue-Violet
        '#AB47BC',  # Violet
        '#D81B60'   # Pink-Violet
    ]
    
    # Create a white background
    plt.pcolor(np.zeros((len(df), len(df.columns))), 
               cmap='binary', 
               edgecolors='lightgray',
               linewidths=0.1)
    
    # For each row (property), highlight values above threshold
    for idx, (property_name, row) in enumerate(df.iterrows()):
        # Get indices where values exceed threshold
        high_value_idx = row[row > threshold].index
        
        # Convert string indices to integers
        x_positions = [int(col.split('_')[1]) for col in high_value_idx]
        
        # Plot colored cells for high values
        for x in x_positions:
            plt.fill([x, x+1, x+1, x], 
                    [idx, idx, idx+1, idx+1], 
                    color=colors[idx], 
                    alpha=0.7)
    
    # Customize the plot
    plt.yticks(np.arange(0.5, len(df), 1), df.index)
    plt.xlabel('Dimensions (0-767)')
    plt.title(f'EDI Scores Above {threshold} for Each Linguistic Property')
    
    # Create custom legend patches
    
    legend_elements = [Patch(facecolor=colors[i], 
                           label=f"{prop} ({len(df.iloc[i][df.iloc[i] > threshold])} dims)", 
                           alpha=0.7) 
                      for i, prop in enumerate(df.index)]
    plt.legend(handles=legend_elements, 
              bbox_to_anchor=(1.05, 1), 
              fontsize=16,
              markerscale=2,
              borderpad=1,
              loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('results/colored_grid_edi_scores.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def create_individual_colored_grid_rows(df, threshold=0.8):
    # Define distinct colors for each property
    colors = {
        'Control': '#EC9D91',      
        'Negation': '#FF85FF',     
        'Intensifier': '#00AB8E',  
        'Tense': '#61D836',        
        'Voice': '#00A2FF',        
        'Polarity': '#9437FF',     
        'Quantity': '#FFD932',     
        'Factuality': '#7A81FF',   
        'Definiteness': '#0433FF', 
        'Synonym': '#F27200'       
    }
    
    # Process each property individually
    for property_name, row in df.iterrows():
        # Create a figure for this property
        plt.figure(figsize=(20, 2))
        
        # Create a white background
        plt.pcolor(np.zeros((1, len(df.columns))), 
                  cmap='binary', 
                  edgecolors='lightgray',
                  linewidths=0.1)
        
        # Get indices where values exceed threshold
        high_value_idx = row[row > threshold].index
        x_positions = [int(col.split('_')[1]) for col in high_value_idx]
        
        # Plot colored cells for high values
        for x in x_positions:
            plt.fill([x, x+1, x+1, x], 
                    [0, 0, 1, 1], 
                    color=colors[property_name], 
                    alpha=0.7, 
                    linewidth=2)
        
        # Customize the plot
        plt.yticks([])
        plt.xticks(fontsize=22) 
        # plt.xlabel('Dimensions (1-768)')
        # plt.title(f'{property_name} EDI Scores Above {threshold} ({len(x_positions)} dimensions)')
        
        # Create save directory if it doesn't exist
        save_dir = f'results/{property_name.lower()}/'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the plot
        plt.savefig(f'{save_dir}/dimension_highlights.svg', 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()


if __name__ == "__main__":

    # df = create_edi_comparison_table(768)
    # print(f"Created table with shape: {df.shape}")
    # print("\nFirst few columns of the table:")
    # print(df.iloc[:, :5])

    df = pd.read_csv('results/combined_edi_scores.csv', index_col=0)

    # create_highlighted_heatmap(df)
    # print("Created heatmap at 'results/edi_scores_heatmap.png'")

    # create_colored_top_values(df, threshold=0.8)
    # print("Created visualization at 'results/colored_top_edi_scores.png'")

    # create_colored_grid(df, threshold=0.8)
    # print("Created visualization at 'results/colored_grid_edi_scores.png'")

    create_individual_colored_grid_rows(df, threshold=0.8)



# if __name__ == "__main__":
#     embedding_filepaths = get_embeddings_filepaths()
#     for embeddings_csv in embedding_filepaths:
#         edi_df = pd.read_csv(os.path.join(get_results_directory(embeddings_csv, "edi_scores"), "edi_score.csv"))

