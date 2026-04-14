"""Balance peptide datasets by generating conservative amino-acid substitutions.

This module expands underrepresented sequence-length groups so that each source CSV
has a more even distribution of peptide lengths before downstream model training.
"""

import os
import pandas as pd
import random

# Define the dictionary of similar amino acids
similar_amino_acids = {
    'A': ['V', 'L', 'I'],
    'V': ['A', 'L', 'I'],
    'L': ['A', 'V', 'I'],
    'I': ['A', 'V', 'L'],
    'S': ['T'],
    'T': ['S'],
    'D': ['E'],
    'E': ['D'],
    'K': ['R'],
    'R': ['K'],
    'F': ['Y', 'W'],
    'Y': ['F', 'W'],
    'W': ['F', 'Y'],
    'G': ['A'],
    'P': ['A'],
    'Q': ['N'],
    'N': ['Q'],
    'H': ['K', 'R'],
    'M': ['L', 'I'],
    'C': ['S'],
}

def mutate_sequence(sequence, n):
    """Generate up to ``n`` single-site conservative mutations for one sequence."""
    if len(sequence) <= 10:
        return []

    all_mutated_sequences = set()
    
    for index_to_mutate in range(len(sequence)):
        original_amino_acid = sequence[index_to_mutate]
        
        if original_amino_acid in similar_amino_acids:
            possible_mutations = similar_amino_acids[original_amino_acid]
            
            for new_amino_acid in possible_mutations:
                mutated_sequence = (sequence[:index_to_mutate] +
                                    new_amino_acid +
                                    sequence[index_to_mutate + 1:])
                
                all_mutated_sequences.add(mutated_sequence)

    # Cap the output size so each source sequence contributes a bounded number of variants.
    all_mutated_sequences = list(all_mutated_sequences)
    if len(all_mutated_sequences) > n:
        return random.sample(all_mutated_sequences, n)
    else:
        return all_mutated_sequences


def balance_amino_acid_data_in_folder(folder_path,out_folder_path):
    """Balance every CSV file in a folder and write the expanded datasets out."""
    # Process each peptide dataset independently so the output files mirror the input names.
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_file = os.path.join(folder_path, filename)
            print(f"Processing file: {csv_file}")
            
            df = pd.read_csv(csv_file)
            df = df[df['seq'].apply(len) <= 50]
            
            # Measure the sequence-length distribution that will be balanced.
            df['Length'] = df['seq'].apply(len)
            length_counts = df['Length'].value_counts()

            # Use the largest bucket as the balancing target for every other length.
            max_count = length_counts.max()

            # Create synthetic variants only for length groups that are currently underrepresented.
            new_sequences = []
            for length, count in length_counts.items():
                if count < max_count:
                    sequences_to_balance = df[df['Length'] == length]['seq']
                    needed_sequences = max_count - count
                    for seq in sequences_to_balance:
                        mutations_needed = min(needed_sequences, max_count - len(sequences_to_balance))
                        new_sequences += mutate_sequence(seq, mutations_needed)
                        needed_sequences -= mutations_needed
                        if needed_sequences <= 0:
                            break

            # Append the generated sequences and export a dataset with the original schema.
            new_df = pd.DataFrame(new_sequences, columns=['seq'])
            df_balanced = pd.concat([df, new_df], ignore_index=True)

            # Remove the temporary helper column before saving the final CSV.
            df_balanced.drop(columns=['Length'], inplace=True)
            output_file = os.path.join(out_folder_path, filename)
            df_balanced.to_csv(output_file, index=False)
            print(f"Saved balanced file: {output_file}")

# Example usage for the default balancing workflow.
folder_path = 'data/base'  # Replace with the folder path containing the CSV files
out_folder_path = 'data/embance_data'
balance_amino_acid_data_in_folder(folder_path,out_folder_path)
