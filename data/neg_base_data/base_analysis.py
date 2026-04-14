"""Plot sequence-length distributions for generated negative peptide datasets."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def generate_peptide_length_distribution_plot(csv_file_path, output_pic_dir):
    """Create a length-distribution plot for one negative dataset CSV file."""
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Assume peptide sequences are stored in the 'Sequence' column
    peptides = df['Sequence'].astype(str)

    # Calculate the length of each peptide
    peptide_lengths = peptides.apply(len)

    # Calculate the total number of peptides
    total_count = len(peptide_lengths)

    # Create a DataFrame to store peptide lengths
    length_df = pd.DataFrame({'peptide_length': peptide_lengths})

    # Handle inf and NaN values
    length_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    length_df.dropna(inplace=True)

    # Calculate the frequency of each length and convert it to percentages
    length_counts = length_df['peptide_length'].value_counts(normalize=True) * 100
    length_counts = length_counts.sort_index()



    # Build the output file path
    base_name = os.path.basename(csv_file_path)
    file_name, _ = os.path.splitext(base_name)


    # Draw the percentage histogram for the generated negative peptides.
    plt.figure(figsize=(10, 6))
    counts, bins = np.histogram(length_df['peptide_length'], bins=30)
    counts = counts / counts.sum() * 100
    plt.hist(bins[:-1], bins, weights=counts, edgecolor='k')
    plt.title(f'{file_name} (Total: {total_count})')
    plt.xlabel('Peptide Length')
    plt.ylabel('Percentage (%)')


    output_file_path = os.path.join(output_pic_dir, f'{file_name}.png')

    # Save the figure at 300 DPI
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_all_csv_files_in_folder(folder_path, output_pic_dir):
    """Generate plots for every negative-dataset CSV file in a folder."""
    # Ensure the output directory exists.
    os.makedirs(output_pic_dir, exist_ok=True)
    
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            print(file_name)
            csv_file_path = os.path.join(folder_path, file_name)
            generate_peptide_length_distribution_plot(csv_file_path, output_pic_dir)
            print(f'Processed {csv_file_path}')

# Example usage for the default negative-dataset analysis workflow.
folder_path = 'data/neg_base_data/csv'
output_pic_dir = 'data/neg_base_data/pic'
process_all_csv_files_in_folder(folder_path, output_pic_dir)
