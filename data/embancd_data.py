"""Augment peptide datasets by applying limited conservative multi-site mutations."""

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

def mutate_sequence(sequence, n, max_mutations):
    """Generate up to ``n`` unique mutants while limiting the number of edited positions."""
    if len(sequence) <= 10:
        return []

    mutated_sequences = set()

    # Favor simpler mutations first so synthetic samples stay close to the source sequence.
    for num_mutations in range(1, max_mutations + 1):
        while len(mutated_sequences) < n:
            indices_to_mutate = random.sample(range(len(sequence)), num_mutations)  # Randomly choose mutation sites
            mutated_sequence = list(sequence)  # Convert the string to a list for editing

            for index in indices_to_mutate:
                original_amino_acid = sequence[index]

                if original_amino_acid in similar_amino_acids:
                    possible_mutations = similar_amino_acids[original_amino_acid]
                    new_amino_acid = random.choice(possible_mutations)
                    mutated_sequence[index] = new_amino_acid  # Replace with the new amino acid

            mutated_sequences.add("".join(mutated_sequence))  # Convert the list back to a string

            if len(mutated_sequences) >= n:
                break  # Stop once the required number of mutated sequences is reached

        if len(mutated_sequences) >= n:
            break  # Stop if enough sequences have already been generated at this mutation count

    return list(mutated_sequences)

def balance_amino_acid_data_in_folder(folder_path,out_folder_path,max_mutations=3):
    """Balance sequence-length counts for every CSV file in a folder."""
    # Iterate through all CSV files in the folder.
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_file = os.path.join(folder_path, filename)
            print(f"Processing file: {csv_file}")
            
            df = pd.read_csv(csv_file)
            
            # Compute the current sequence-length distribution for this dataset.
            df['Length'] = df['seq'].apply(len)
            length_counts = df['Length'].value_counts()

            # Use the largest length bucket as the balancing target.
            max_count = length_counts.max()

            # Generate additional synthetic sequences only for underrepresented lengths.
            new_sequences = []
            for length, count in length_counts.items():
                if count < max_count:
                    sequences_to_balance = df[df['Length'] == length]['seq']
                    needed_sequences = max_count - count
                    for seq in sequences_to_balance:
                        mutations_needed = min(needed_sequences, max_count - len(sequences_to_balance))
                        new_sequences += mutate_sequence(seq, mutations_needed,max_mutations)
                        needed_sequences -= mutations_needed
                        if needed_sequences <= 0:
                            break

                # Append the generated sequences to the original dataset.
            new_df = pd.DataFrame(new_sequences, columns=['seq'])
            df_balanced = pd.concat([df, new_df], ignore_index=True)

                # Remove the temporary helper column before exporting the balanced dataset.
            df_balanced.drop(columns=['Length'], inplace=True)
            output_file = os.path.join(out_folder_path, filename)
            df_balanced.to_csv(output_file, index=False)
            print(f"Saved balanced file: {output_file}")

if __name__ == '__main__':
        # Example usage for the default augmentation pipeline.
        folder_path = 'data/base'  # Replace with the folder path containing the CSV files
    out_folder_path = 'data/embance_data'
    balance_amino_acid_data_in_folder(folder_path,out_folder_path)
