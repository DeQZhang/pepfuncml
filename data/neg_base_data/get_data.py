"""Sample negative peptide fragments from a FASTA source while controlling similarity."""

import os
import pandas as pd
import numpy as np
import random
import Levenshtein  # Import Levenshtein to calculate string similarity

def extract_sequences_from_fasta(fasta_file, length_distribution, num_sequences, base_sequences):
    """Extract sequence fragments that match a target length profile and stay below a similarity threshold."""
    with open(fasta_file, 'r') as f:
        fasta_content = f.read()
    
    sequences = []
    for entry in fasta_content.split('>')[1:]:
        lines = entry.split('\n')
        header = lines[0]
        sequence = ''.join(lines[1:])
        sequences.append(sequence)
    
    extracted_sequences = set()
    total_attempts = 0
    max_attempts = num_sequences * 10  # Prevent infinite loops by capping the number of attempts.
    
    while len(extracted_sequences) < num_sequences and total_attempts < max_attempts:
        for length, percentage in length_distribution.items():
            num_to_extract = int(num_sequences * (percentage / 100))
            filtered_sequences = [seq for seq in sequences if len(seq) >= length]
            
            for _ in range(num_to_extract):
                if not filtered_sequences:
                    break
                seq = random.choice(filtered_sequences)
                start_index = random.randint(0, len(seq) - length)
                extracted_seq = seq[start_index:start_index + length]
                
                # Reject fragments that are too similar to any positive reference sequence.
                if not any(
                    Levenshtein.ratio(extracted_seq, base_seq) >= 0.9 for base_seq in base_sequences
                ):
                    extracted_sequences.add(extracted_seq)
                
                total_attempts += 1
                if len(extracted_sequences) >= num_sequences:
                    break
    
    return list(extracted_sequences)

def main(csv_folder, fasta_file, output_folder):
    """Generate one negative dataset per CSV length-distribution summary."""
    os.makedirs(output_folder, exist_ok=True)
    
    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(csv_folder, csv_file)
            df = pd.read_csv(file_path)
            
            # Match the negative-sample count to twice the size of the corresponding positive dataset.
            df1 = pd.read_csv(os.path.join('data/embance_data', csv_file))
            num_sequences = len(df1)* 2
            
            # Reuse the observed length distribution from the analysis step.
            length_distribution = dict(zip(df['Peptide Length'], df['Percentage']))
            
            # Use positive sequences as references for similarity filtering.
            base_sequences = df1['seq'].tolist()  # Adjust the column name if your data uses a different one
            
            # Randomly sample negative fragments from the FASTA source under the defined constraints.
            extracted_sequences = extract_sequences_from_fasta(fasta_file, length_distribution, num_sequences, base_sequences)
            
            # Save the generated negative dataset using the same filename as the source summary table.
            output_file_path = os.path.join(output_folder, csv_file)
            output_df = pd.DataFrame({
                'Sequence Number': [f'sequence_{i+1}' for i in range(len(extracted_sequences))],
                'Sequence': extracted_sequences
            })
            output_df.to_csv(output_file_path, index=False)

if __name__ == '__main__':
    csv_folder = 'data/base_analysis/embance_data/csv'  # Replace with the actual CSV folder path
    fasta_file = '../animal_proteins.fasta'  # Replace with the actual FASTA file path
    output_folder = 'data/neg_base_data/csv'  # Replace with the actual output folder path
    main(csv_folder, fasta_file, output_folder)
