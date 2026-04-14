
"""Train the selected classifier for each peptide task and export evaluation artifacts."""




# Confusion matrix and AUC plots
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from multiprocessing.dummy import Pool as ThreadPool

def train_with_best_params(file_name, pr_results):
    """Train one task-specific model using tuned parameters and collect PR-curve data."""
    folder_path = 'data/embedding/Dimensionality_Reduction'
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, index_col=None)
    
    params_filename = f'optim_parameters/parameters/{file_name.split(".")[0]}.{file_name.split(".")[1]}_params_best.json'
    with open(params_filename, 'r') as f:
        best_params = json.load(f)
    
    model_name = best_params.pop('model')
    accuracy = best_params.pop('accuracy')
    X = df.drop('label', axis=1).values
    Y = df['label'].values

    # Hold out a test split so the final confusion matrix and PR curve are evaluated consistently.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    models = {
        "LR": LogisticRegression(max_iter=1000),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
        "GBDC": GradientBoostingClassifier(),
        "AdB": AdaBoostClassifier(),
        "ET": ExtraTreesClassifier(),
        "SVC": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "GNB": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "MLP": MLPClassifier(max_iter=1000)
    }

    model = models[model_name]
    model.set_params(**best_params)

    # Fit the selected model with its tuned hyperparameters.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Get the probability of the positive class

    # Save the confusion matrix in CSV form so it can be reused in figures or reports.
    cm = confusion_matrix(y_test, y_pred)
    with open(f'model/test/{file_name.split(".")[0]}_confusion_matrix.csv', 'w', newline='') as cm_file:
        cm_writer = csv.writer(cm_file)
        cm_writer.writerow(['True Negative', 'False Positive', 'False Negative', 'True Positive'])
        cm_writer.writerows(cm.flatten().reshape(1, -1))  # Flatten and write the confusion matrix

    # Compute the precision-recall curve from predicted probabilities.
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    # Summarize the curve with its area under the curve value.
    pr_auc = auc(recall, precision)

    # Store everything needed to draw a combined plot across all peptide tasks.
    pr_results.append((file_name.split(".")[0], recall, precision, pr_auc))

def plot_precision_recall(pr_results):
    """Plot the precision-recall curves for all trained task-specific models."""
    fig, ax = plt.subplots(figsize=(6, 6))

    for file_name, recall, precision, pr_auc in pr_results:
        label = f'{file_name} (AUC = {pr_auc:.4f})'  # Use the filename and AUC value as the legend label
        plt.plot(recall, precision, lw=2, label=label)

    # Draw the Precision = Recall diagonal line from (1, 0) to (0, 1)
    plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')  # Keep this line out of the legend

    # Configure plot properties
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)

    # Leave the title disabled
    # plt.title('Precision-Recall Curve for All Models', fontsize=20)  # No title is needed

    # Increase the legend font size
    plt.legend(loc='lower left', fontsize=14)

    # Adjust the x-axis and y-axis tick label sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Set tick width to 2
    plt.tick_params(axis='both', width=2)

    # Hide the top and right spines
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Set tick width to 2
    ax.tick_params(axis='both', width=2, length=6)

    # Save the figure
    plt.savefig('model/test/all_models_precision_recall_curve.png')
    plt.savefig('model/test/all_models_precision_recall_curve.svg', format='svg')  # Save the vector version as SVG
    plt.show()

if __name__ == '__main__':
    folder_path = 'data/embedding/test'
    files = os.listdir(folder_path)

    pr_results = []  # Store Precision-Recall curve data for all models

    pool = ThreadPool(64)
    pool.map(lambda file_name: train_with_best_params(file_name, pr_results), files)
    pool.close()
    pool.join()

    # Plot the Precision-Recall curves for all models
    plot_precision_recall(pr_results)

