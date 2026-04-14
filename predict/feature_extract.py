"""Extract ProtT5 embeddings for inference and optionally project them with PCA."""

import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
import pandas as pd
from multiprocessing import Pool, cpu_count
import joblib

def feature_extract(sequences, model, tokenizer, pca):
    """Convert peptide sequences into pooled ProtT5 embeddings or PCA-reduced vectors."""
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Replace uncommon residue symbols so the tokenizer receives valid tokens.
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
    if pca is not None:
        for seq_num in range(len(pooled_embedding)):
            seq_emd = pooled_embedding[seq_num]
            result = pca.transform([seq_emd])
            features.append(result)
    else:
        return pooled_embedding

    return features

def load_model_and_tokenizer():
    """Load the pretrained ProtT5 encoder used during training and inference."""
    tokenizer = T5Tokenizer.from_pretrained("../embedder/Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("../embedder/Rostlab/prot_t5_xl_uniref50")
    return model, tokenizer

def load_pca_model():
    """Load the PCA model used to reduce embedding dimensionality during inference."""
    file_path = "/home/zdq/gzFile/guanzhan/Pepefficacy-2/data/embedding/pca_model.pkl"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            pca = joblib.load(f)
    else:
        pca = None
    return pca

def save_pca_model(pca):
    """Persist the PCA model so later scripts can reuse the same projection."""
    file_path = "/home/zdq/gzFile/guanzhan/Pepefficacy-2/data/embedding/pca_model.pkl"
    with open(file_path, 'wb') as f:
        joblib.dump(pca, f)

def fit_pca_model(data):
    """Fit a PCA model on pooled embeddings and visualize explained variance."""
    pca = PCA(n_components=64)  # Choose an appropriate number of principal components.

    pca.fit(data)
    save_pca_model(pca)
    # Plot the explained variance ratio.
    labels = [f'PC{num+1}' for num in range(64)]
    df = pd.DataFrame({'Data': pca.explained_variance_ratio_,'Labels': labels})

    # Draw the bar chart with Seaborn.
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Labels', y='Data', data=df)

    # Add the title and axis labels.
    plt.title('explained_variance_ratio')
    plt.xlabel('Principal Components', fontsize=14)
    plt.ylabel('Explained Variance Ratio', fontsize=14)

    # Save the figure.
    plt.savefig('PCA.png')
    # Display the figure.
    plt.show()
    return pca

def process_file(args):
    """Embed one labeled peptide dataset and save the projected features as CSV."""
    file_name, folder_path = args
    model, tokenizer = load_model_and_tokenizer()
    pca = load_pca_model()

    print(f"Processing {file_name}")
    file_path = os.path.join(folder_path, file_name)
    
    df = pd.read_csv(file_path)
    labels = df['Label'].values
    seqs = df['seq'].values
    
    # Fit PCA on a sample only once if no saved projection model is available yet.
    if pca is None:
        sample_seq = seqs[:1000]
        sample_embedding = feature_extract(sample_seq, model, tokenizer, None)
        sample_df = pd.DataFrame(sample_embedding)
        pca = fit_pca_model(sample_df.values)
        
    # Process the dataset in batches to avoid excessive memory use during embedding.
    batch_size = 64
    n = len(seqs) // batch_size
    count = 0
    
    save_path = 'data/embedding/embedded'
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
        batch_vector = feature_extract(batch_seqs, model, tokenizer, pca)
        
        data = []
        for j in range(len(batch_vector)):
            result_dict = {}
            vector = batch_vector[j][0]
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

def process_files(folder_path):
    """Parallelize inference-time feature extraction across every dataset CSV file."""
    files = os.listdir(folder_path)
    args = [(file_name, folder_path) for file_name in files]
    
    with Pool(cpu_count()) as pool:
        pool.map(process_file, args)

if __name__ == '__main__':
    folder_path = 'data/embedding/dataset'
    process_files(folder_path)
