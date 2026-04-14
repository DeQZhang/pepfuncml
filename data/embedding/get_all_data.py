"""Merge positive and negative peptide datasets into labeled training tables."""

import os
import pandas as pd

def merge_csv_files(folder1, folder2, output_folder):
    """Merge matching CSV files from two folders and assign binary labels."""
    os.makedirs(output_folder, exist_ok=True)
    
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    

    common_files = files1.intersection(files2)

    
    for file_name in common_files:
        if file_name.endswith('.csv'):
            file1_path = os.path.join(folder1, file_name)
            file2_path = os.path.join(folder2, file_name)
            
            # Read the positive dataset and assign the positive class label.
            df1 = pd.read_csv(file1_path)
            
            df1['Label'] = '1'
            
            # Read the negative dataset, align the column name, and assign the negative class label.
            df2 = pd.read_csv(file2_path)
            df2 = df2.rename(columns={'Sequence': 'seq'}) 
            
            df2['Label'] = '0'
            
                # Merge the two labeled datasets into one training table.
            merged_df = pd.concat([df1, df2], ignore_index=True)

                # Drop duplicate sequences so the same peptide appears only once.
            merged_df = merged_df.drop_duplicates(subset='seq', keep='first')
            
                # Export the merged dataset using the shared filename.
            output_file_path = os.path.join(output_folder, file_name)
            merged_df.to_csv(output_file_path, index=False)
            print(f'Merged {file_name} and saved to {output_file_path}')

if __name__ == '__main__':
        folder1 = 'data/embance_data'  # Replace with the actual path of the first folder
        folder2 = 'data/neg_base_data/csv'  # Replace with the actual path of the second folder
        output_folder = 'data/embedding/dataset_low_90'  # Replace with the output folder path
    merge_csv_files(folder1, folder2, output_folder)
