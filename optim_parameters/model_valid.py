"""Validate tuned task-specific models with cross-validation."""

import pandas as pd
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def train_with_best_params(file_name):
    """Load tuned parameters for one dataset and report cross-validation scores."""
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
    
    # Use shuffled K-fold cross-validation to estimate performance stability.
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Evaluate the tuned estimator across all folds and print the score vector.
    scores = cross_val_score(model, X, Y, cv=kf)

    print(file_name.split(".")[0],scores)

if __name__ == '__main__':
    # Validate every tuned task-specific model in parallel.
    folder_path = 'data/embedding/Dimensionality_Reduction'
    files = os.listdir(folder_path)

    pool = ThreadPool(64)
    pool.map(train_with_best_params, files)
    pool.close()
    pool.join()