"""Analyze peptide-length distributions and export plots plus summary tables."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def generate_peptide_length_distribution_plot(csv_file_path, output_pic_dir, output_csv_dir):
    """Create one length-distribution plot and one summary CSV for a peptide dataset."""
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    df = df[df['seq'].apply(len) <= 50]
    # Assume peptide sequences are stored in the 'seq' column
    peptides = df['seq']

    # Calculate the length of each peptide
    peptide_lengths = peptides.apply(len)

    # Create a DataFrame to store peptide lengths
    length_df = pd.DataFrame({'peptide_length': peptide_lengths})

    # Handle inf and NaN values
    length_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    length_df.dropna(inplace=True)

    # Calculate the frequency of each length and convert it to percentages
    length_counts = length_df['peptide_length'].value_counts(normalize=True) * 100
    length_counts = length_counts.sort_index()

    # Create the output DataFrame
    output_data = pd.DataFrame({
        'Peptide Length': length_counts.index,
        'Percentage': length_counts.values,
    })

    # Build the output file path
    base_name = os.path.basename(csv_file_path)
    file_name, _ = os.path.splitext(base_name)

    # Save the lengths and percentages to a CSV file
    output_csv_path = os.path.join(output_csv_dir, f'{file_name}.csv')
    output_data.to_csv(output_csv_path, index=False)

    # Draw a percentage histogram with one bar per observed peptide length.
    plt.figure(figsize=(12, 6))
    
    # Use bar() to draw one bar for each length
    plt.bar(length_counts.index, length_counts.values, color='wheat', edgecolor='black')

    # Set the plot range
    plt.xlim([0, 50])  # Set the x-axis range from 0 to 50
    plt.xlabel('Peptide Length', fontsize=24)
    plt.ylabel('Percentage (%)', fontsize=24)

    # Set the tick label font sizes on both axes
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Force integer ticks on the y-axis
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))

    # Configure the axis spines
    ax = plt.gca()
    # ax.set_ylim(0, 10)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    # Set tick width to 2
    ax.tick_params(axis='both', width=2, length=6)

    # Save the figure
    output_file_path = os.path.join(output_pic_dir, f'{file_name}')
    plt.savefig(output_file_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_file_path + '.svg', bbox_inches='tight')  # Save the vector version as SVG
    plt.close()

def process_all_csv_files_in_folder(folder_path, output_pic_dir, output_csv_dir):
    """Generate length summaries for every CSV file in the input folder."""
    # Ensure the output directories exist.
    os.makedirs(output_pic_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)
    
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, file_name)
            generate_peptide_length_distribution_plot(csv_file_path, output_pic_dir, output_csv_dir)
            print(f'Processed {csv_file_path}')

# Example usage for the default base-dataset analysis workflow.

name = 'base'
folder_path = f'data/{name}'
output_pic_dir = f'data/base_analysis/{name}/pic'
output_csv_dir = f'data/base_analysis/{name}/csv'
process_all_csv_files_in_folder(folder_path, output_pic_dir, output_csv_dir)
