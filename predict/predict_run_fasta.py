"""Run trained peptide classifiers on all unique fragments generated from FASTA inputs."""

from joblib import load
import os
import pandas as pd
import sys


from predict.feature_extract import feature_extract, load_pca_model, load_model_and_tokenizer

def predict_run(X):
    """Run every trained classifier on the provided feature matrix."""
    # Load the trained models with joblib.
    folder_path = 'model/trained_model'
    models = os.listdir(folder_path)

    result = {}
    for model in models:
        model_filename = os.path.join(folder_path, model)
        load_model = load(model_filename)
        is_cite = load_model.predict(X)
        # Store both the hard prediction and the positive-class probability for later ranking.
        y_pred_proba = load_model.predict_proba(X)[:, 1]
        result[model.split(".")[0]] = is_cite
        result[model.split(".")[0]+'_proba'] = y_pred_proba
        
    return result

def generate_fragments(sequence, name):
    """Generate unique subsequences between lengths 10 and 50 for one FASTA entry."""
    fragments = []
    fragment_names = []
    seen_fragments = set()  # Keep track of unique fragments
    count = 0
    for length in range(10, 51):  # Iterate over fragment lengths from 10 to 50
        for i in range(len(sequence) - length + 1):
            fragment = sequence[i:i + length]
            if fragment not in seen_fragments:
                seen_fragments.add(fragment)
                count += 1
                fragments.append(fragment)
                fragment_names.append(name + f'{count}')
    return fragments, fragment_names

def read_fasta(file_path):
    """Read a FASTA file into a mapping from sequence identifiers to raw sequences."""
    sequences = {}
    with open(file_path, 'r') as fasta_file:
        sequence_id = ""
        sequence_data = ""

        for line in fasta_file:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id:
                    sequences[sequence_id] = sequence_data
                sequence_id = line[1:]  # Remove the '>' character
                sequence_data = ""
            else:
                sequence_data += line

        if sequence_id:  # Add the last sequence to the dictionary
            sequences[sequence_id] = sequence_data

    return sequences

def process_in_batches(fragments, model, tokenizer, pca, batch_size=100):
    """Convert sequence fragments into model-ready feature vectors in batches."""
    data = []
    count = 0 
    for start in range(0, len(fragments), batch_size):
        count+=1
        print('Progress:',f'{count}/{len(fragments)//batch_size+1}')
        batch_fragments = fragments[start:start + batch_size]
        # Extract embeddings for one batch at a time to keep memory usage bounded.
        seqs_vector = feature_extract(batch_fragments, model, tokenizer, pca)
        
        for i in range(len(seqs_vector)):
            result_dict = {}
            vector = seqs_vector[i]
            for j in range(len(vector)):
                col_name = f'feature_{j+1}'
                result_dict[col_name] = vector[j]
            result_dict['seg'] = batch_fragments[i]
            data.append(result_dict)
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    # Process each FASTA file name listed here and save one CSV per input sequence record.
    names = ['col1_jd']
    for name in names:
        fasta_sequences = read_fasta(f"predict/{name}.fasta")

        file_path = f'predict/result/{name}'
        if not os.path.exists(file_path):
            # Create the folder if it does not exist
            os.makedirs(file_path)
        for seq_id, seq in fasta_sequences.items():
            name = seq_id.split(" ")[0]
            fragments, fragment_names = generate_fragments(seq, name)

            model, tokenizer = load_model_and_tokenizer()
            pca = load_pca_model()

            # Process the fragments in batches
            df = process_in_batches(fragments, model, tokenizer, pca, batch_size=512)

            X = df.drop('seg', axis=1).values
            X = [feature[0] for feature in X]
            seg = df['seg'].values

            predict_result = predict_run(X)

            predict_result['seqs'] = seg
            predict_result['seq_names'] = fragment_names

            df_result = pd.DataFrame(predict_result)
            df_result.to_csv(f'{file_path}/{name}.csv', index=False)

