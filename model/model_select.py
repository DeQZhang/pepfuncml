"""Compare candidate classifiers for each peptide task and save selection results."""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


def model_select(file_name):
    """Train all candidate models for one dataset and record their test accuracy."""
    folder_path = 'data/embedding/Dimensionality_Reduction'
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, index_col=None)
    
    # Prepare the data
    X = df.drop('label', axis=1).values
    Y = df['label'].values

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize the candidate models
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
        "MLP": MLPClassifier(max_iter=1000),
        "XGB": XGBClassifier(eval_metric='logloss')
    }

    # Train and evaluate every candidate model on the same train/test split.
    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy

    # Save every model's score so aggregated comparisons can be built later.
    for model_name, accuracy in accuracies.items():
        result_df = pd.DataFrame({
            'File': [file_name],
            'Model': [model_name],
            'Accuracy': [accuracy]
        })
        result_file_path = f'model_results/{model_name}_results.csv'
        
        # Append if the file exists; otherwise create a new file
        if os.path.exists(result_file_path):
            result_df.to_csv(result_file_path, mode='a', header=False, index=False)
        else:
            result_df.to_csv(result_file_path, mode='w', header=True, index=False)

    # Build a DataFrame for plotting and highlighting the best classifier.
    df = pd.DataFrame({"Model": list(accuracies.keys()), "Accuracy": list(accuracies.values())})

    # Get the index of the maximum value
    max_index = df["Accuracy"].idxmax()

    # Highlight the best bar in pink and keep the others light blue
    colors = ['skyblue' if i != max_index else 'pink' for i in range(len(df))]

    # Draw the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Model", y="Accuracy", hue="Model", data=df, palette=colors, dodge=False)
    plt.xlabel("Models", fontsize=18)
    plt.xticks(rotation=45, fontsize=16)
    plt.ylabel("Accuracy", fontsize=18)
    plt.yticks(fontsize=16)
    plt.tick_params(axis='both', width=2)
    ax.set_ylim(0.7, 1)
    yticks = [round(i, 1) for i in plt.yticks()[0]]
    plt.yticks(yticks)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='both', width=2, length=6)

    plt.savefig(f'model/pic/model_select/{file_name.split(".")[0]}.png', dpi=300)
    plt.savefig(f'model/pic/model_select/{file_name.split(".")[0]}.svg', format='svg', bbox_inches='tight')

    # Print the per-model scores for quick inspection in the console.
    print(file_name.split(".")[0], df["Accuracy"].values)


if __name__ == '__main__':
    # Read the reduced feature files that correspond to each peptide task.
    folder_path = 'data/embedding/Dimensionality_Reduction'
    files = os.listdir(folder_path)

    # Store intermediate per-model accuracy tables in a dedicated folder.
    if not os.path.exists('model_results'):
        os.makedirs('model_results')

    # Parallelize model selection across peptide tasks.
    pool = Pool(64)

    # Process the files in parallel.
    pool.map(model_select, files)

    # Close the process pool.
    pool.close()
    pool.join()

    print("Each model result has been saved to a separate CSV file.")
