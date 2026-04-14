"""Build peptide feature vectors from ProtT5 embeddings and amino-acid descriptors."""

import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool, cpu_count


from collections import Counter

def shannon_entropy(sequence):
    """Compute the Shannon entropy of an amino acid sequence."""
    length = len(sequence)
    if length == 0:
        return 0 # Avoid errors for empty sequences.
    counts = Counter(sequence)
    probabilities = np.array([count / length for count in counts.values()])
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def inverse_normalized_entropy(sequence):
    """Compute inverse normalized Shannon entropy."""
    L = len(sequence)
    if L == 0:
        return 1 # Define the value of an empty sequence as 1.
    H = shannon_entropy(sequence)
    H_max = np.log2(20) # Use the theoretical maximum entropy.
    return 1 - (H / H_max) if H_max > 0 else 1 # Avoid division-by-zero errors.



def load_csv_data(csv_path):
    """Load CSV data into a dictionary mapping amino acid sequences to 1024-dimensional features."""
    df = pd.read_csv(csv_path)
    csv_data = {}
    for _, row in df.iterrows():
        sequence = row['seq']
        features = row.drop('seq').values  # Extract feature values from all columns except 'seq'.
        csv_data[sequence] = np.array(features)
    return csv_data

def get_sequence_features(sequence, csv_data):
    """Generate an n*1024 feature matrix from an amino acid sequence."""
    sequence_len = len(sequence)
    
    # Create an n*1024 matrix whose row count matches the sequence length.
    sequence_matrix = np.zeros((sequence_len, 1024))
    
    for i, amino_acid in enumerate(sequence):
        # Use the corresponding feature vector from the CSV data.
        if amino_acid in csv_data:
            sequence_matrix[i] = csv_data[amino_acid]  # 1024-dimensional features for this amino acid.
        else:
            sequence_matrix[i] = np.zeros(1024)  # Use a zero vector when no feature is available.
    
    return sequence_matrix

def feature_extract(sequences, model, tokenizer, csv_data):
    """Embed peptide sequences and blend ProtT5 features with handcrafted residue descriptors."""
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Replace rare or unsupported residue codes so tokenization remains compatible with ProtT5.
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    sequences = [' '.join(seq) for seq in sequences]

    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = embedding.last_hidden_state

    # Use mean pooling to convert variable-length embeddings into fixed-length vectors.
    pooled_embedding = torch.mean(last_hidden_state, dim=1).cpu().numpy()

    features = []
    for seq_num in range(len(pooled_embedding)):
        seq_emd = pooled_embedding[seq_num]
        sequence = sequences[seq_num].replace(' ', '')  # Remove spaces to recover the original sequence.
        
        # Build the n*1024 feature matrix.
        sequence_matrix = get_sequence_features(sequence, csv_data)
        
        # Apply mean pooling to turn the feature matrix into a fixed-length vector.
        sequence_matrix_torch = torch.tensor(sequence_matrix, dtype=torch.float32)
        pooled_features = torch.mean(sequence_matrix_torch, dim=0).numpy()

        # Use entropy as a mixing weight between language-model features and residue-level descriptors.
        w = inverse_normalized_entropy(sequence)
        # Combine the pooled feature vector with the original sequence embedding.
        adjusted_vector = seq_emd*(1-w) + pooled_features*w
        features.append(adjusted_vector)

    return features

def load_model_and_tokenizer():
    """Load the pretrained ProtT5 encoder and tokenizer used for sequence embeddings."""
    tokenizer = T5Tokenizer.from_pretrained("../embedder/Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("../embedder/Rostlab/prot_t5_xl_uniref50")
    return model, tokenizer

def process_file(args):
    """Convert one labeled peptide CSV file into a feature table suitable for model training."""
    file_name, folder_path, csv_data = args
    model, tokenizer = load_model_and_tokenizer()

    print(f"Processing {file_name}")
    file_path = os.path.join(folder_path, file_name)
    
    df = pd.read_csv(file_path)
    labels = df['Label'].values
    seqs = df['seq'].values
    
    # Process sequences in fixed-size batches to control memory usage during embedding.
    batch_size = 64
    n = len(seqs) // batch_size
    count = 0
    
    save_path = 'data/embedding/embedded_non_embance'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, file_name)
    
    # Write the CSV header only once.
    header_written = False
    
    for i in range(0, len(seqs), batch_size):
        count += 1
        print(f'Progress: {count}/{n}')
        batch_seqs = seqs[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        batch_vector = feature_extract(batch_seqs, model, tokenizer, csv_data)

        
        data = []
        for j in range(len(batch_vector)):
            result_dict = {}
            vector = batch_vector[j]
            label = batch_labels[j]
            for k in range(len(vector)):
                col_name = f'feature_{k+1}'
                result_dict[col_name] = vector[k]
            result_dict['label'] = label
            data.append(result_dict)
        
        result_df = pd.DataFrame(data)
        
        if not header_written:
            result_df.to_csv(save_file, index=False, mode='w')
            header_written = True
        else:
            result_df.to_csv(save_file, index=False, mode='a', header=False)

def process_files(folder_path, csv_data):
    """Parallelize feature extraction across all CSV files in the input folder."""
    files = os.listdir(folder_path)
    args = [(file_name, folder_path, csv_data) for file_name in files]
    
    with Pool(cpu_count()) as pool:
        pool.map(process_file, args)

if __name__ == '__main__':
    folder_path = 'data/embedding/dataset_non_embance'
    csv_path = 'data/aa_features_1024.csv'  # Path to the CSV file storing 1024-dimensional features.
    csv_data = load_csv_data(csv_path)  # Load the CSV feature data.
    process_files(folder_path, csv_data)
