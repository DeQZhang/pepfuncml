"""Benchmark dimensionality-reduction methods before classifier training."""

import os
import glob
import pandas as pd
import csv
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

def evaluate_model(clf, X, y, cv=5):
    """Evaluate one classifier with cross-validated predictions and return common metrics."""
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    pr = precision_score(y, y_pred, average='binary')
    sp = recall_score(y, y_pred, pos_label=0)  # Specificity
    fs = f1_score(y, y_pred, average='binary')
    auc = roc_auc_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    return acc, pr, sp, fs, auc, mcc

def test_and_save_results(csv_file, output_folder):
    """Compare PCA, UMAP, and t-SNE on one embedded dataset and save the metrics."""
    df = pd.read_csv(csv_file)

    # Prepare the data
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Split once so every dimensionality-reduction method is timed on the same training subset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the dimensions to evaluate
    pca_umap_dimensions = [8, 16, 32, 64]
    tsne_dimensions = [2, 3]

    # Extract the CSV filename without the path or extension
    base_name = os.path.basename(csv_file).split('.')[0]

    # Define the output file path
    output_file = os.path.join(output_folder, f'{base_name}_results.csv')

    # Open the output file and write the results
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Method', 'Dimensions', 'Time', 'Accuracy', 'Precision', 'Specificity', 'F1 Score', 'AUC', 'MCC'])

        for dim in pca_umap_dimensions:
            # Measure both runtime and downstream classification performance for PCA.
            start_time = time.time()
            pca = PCA(n_components=dim)
            X_pca = pca.fit_transform(X_train)
            pca_time = time.time() - start_time
            clf_pca = SVC(probability=True, random_state=42)
            pca_metrics = evaluate_model(clf_pca, X_pca, y_train)
            writer.writerow(['PCA', dim, pca_time] + list(pca_metrics))

            # Repeat the same evaluation process for UMAP.
            start_time = time.time()
            reducer = umap.UMAP(n_components=dim, random_state=42)
            X_umap = reducer.fit_transform(X_train)
            umap_time = time.time() - start_time
            clf_umap = SVC(probability=True, random_state=42)
            umap_metrics = evaluate_model(clf_umap, X_umap, y_train)
            writer.writerow(['UMAP', dim, umap_time] + list(umap_metrics))

        for dim in tsne_dimensions:
            # t-SNE is evaluated on lower-dimensional settings because of its higher cost.
            start_time = time.time()
            tsne = TSNE(n_components=dim, random_state=42)
            X_tsne = tsne.fit_transform(X_train)
            tsne_time = time.time() - start_time
            clf_tsne = SVC(probability=True, random_state=42)
            tsne_metrics = evaluate_model(clf_tsne, X_tsne, y_train)
            writer.writerow(['t-SNE', dim, tsne_time] + list(tsne_metrics))

def process_folder(input_folder, output_folder):
    """Run the dimensionality-reduction benchmark for every CSV file in a folder."""
    # Get the paths of all CSV files in the folder.
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    # Process each CSV file
    for csv_file in csv_files:
        test_and_save_results(csv_file, output_folder)
        print(f"Processed file: {csv_file}")

if __name__ == "__main__":
    input_folder = 'data/embedding/embedded/'
    output_folder = 'data/embedding/results/'

    # Create the output folder if it does not exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_folder(input_folder, output_folder)
    print("All CSV files have been processed and the results were saved to the target folder.")
