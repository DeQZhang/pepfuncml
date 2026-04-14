"""Tune the selected model for each peptide task with Bayesian optimization."""

import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBClassifier
from multiprocessing.dummy import Pool as ThreadPool

def model_select(file_name):
    """Optimize one preselected model and save the best hyperparameters to JSON."""
    folder_path = 'data/embedding/Dimensionality_Reduction'
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, index_col=None)
    
    opt_model = pd.read_csv('model/model_table.csv')
    model_name = opt_model[opt_model['peptidase'] == file_name.split(".")[0]]['opt_model'].values[0]

    # Prepare the feature matrix and binary labels for hyperparameter search.
    X = df.drop('label', axis=1).values
    Y = df['label'].values

    # Keep a holdout test split so tuned parameters can be checked on unseen data.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Define one search space per model family.
    param_spaces = {
        "LR": {
            'C': Real(1e-3, 1e3, prior='log-uniform'),
            'solver': Categorical(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        },
        "DT": {
            'max_depth': Integer(1, 50),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 8),
            'criterion': Categorical(['gini', 'entropy'])
        },
        "RF": {
            'n_estimators': Integer(100, 500),
            'max_features': Categorical(['auto', 'sqrt', 'log2']),
            'max_depth': Integer(1, 50),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4)
        },
        "GBDC": {
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 9),
            'subsample': Real(0.7, 1.0)
        },
        "AdB": {
            'n_estimators': Integer(50, 300),
            'learning_rate': Real(0.01, 1.0, prior='log-uniform')
        },
        "SVC": {
            'C': Real(1e-3, 1e2, prior='log-uniform'),
            'gamma': Categorical(['scale', 'auto']),
            'kernel': Categorical(['rbf', 'linear', 'poly', 'sigmoid']),
            'degree': Integer(2, 5)
        },
        "KNN": {
            'n_neighbors': Integer(3, 21),
            'weights': Categorical(['uniform', 'distance']),
            'metric': Categorical(['euclidean', 'manhattan', 'chebyshev', 'minkowski'])
        },
        "GNB": {},  # Gaussian Naive Bayes has no tunable hyperparameters.
        "LDA": {
            'solver': Categorical(['svd', 'lsqr', 'eigen'])
        },
        "QDA": {
            'reg_param': Real(0.0, 0.5)
        },
        "MLP": {
            'activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),
            'solver': Categorical(['sgd', 'adam']),
            'alpha': Real(1e-4, 1e-2, prior='log-uniform'),
            'learning_rate': Categorical(['constant', 'invscaling', 'adaptive'])
        },
        "XGB": {
            'n_estimators': Integer(100, 300),
            'max_depth': Integer(3, 7),
            'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'subsample': Real(0.7, 1.0),
            'colsample_bytree': Real(0.7, 1.0),
            'gamma': Real(0, 0.3)
        }
    }

    # Recreate the estimator objects so the selected model can be optimized.
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
        "XGB": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    model = models[model_name]

    # Record baseline performance before running the Bayesian optimizer.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)

    best_params = {}
    best_params['model'] = model_name
    best_params['accuracy'] = best_accuracy

    if best_accuracy > 0.96:
        params_filename = f'optim_parameters/parameters/{file_name.split(".")[0]}.{file_name.split(".")[1]}_params.json'
        with open(params_filename, 'w') as f:
            json.dump(best_params, f)
        print(file_name.split(".")[0], best_accuracy)
        return


    # Run Bayesian optimization only when the baseline result is not already strong enough.
    param_space = param_spaces[model_name]
    if param_space:
        bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, cv=5, n_jobs=1, verbose=2, scoring='accuracy', n_iter=50, random_state=42)
        bayes_search.fit(X_train, y_train)

        # Retrieve the best estimator and its parameters from the search object.
        best_accuracy = bayes_search.best_score_
        best_model = bayes_search.best_estimator_
        best_params = bayes_search.best_params_

        # Evaluate the tuned estimator on the holdout test set before saving the result.
        y_pred = best_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred)
        best_params['accuracy'] = final_accuracy

        if final_accuracy > 0.96:
            params_filename = f'optim_parameters/parameters/{file_name.split(".")[0]}.{file_name.split(".")[1]}_params.json'
            with open(params_filename, 'w') as f:
                json.dump(best_params, f)
            print(file_name.split(".")[0], final_accuracy)
            return

    # Persist the best available parameter set even when it does not pass the early-return threshold.
    params_filename = f'optim_parameters/parameters/{file_name.split(".")[0]}.{file_name.split(".")[1]}_params_best.json'
    with open(params_filename, 'w') as f:
        json.dump(best_params, f)

    print(file_name.split(".")[0], best_accuracy)

if __name__ == '__main__':
    # Read all reduced feature tables before dispatching optimization jobs.
    folder_path = 'data/embedding/Dimensionality_Reduction'
    files = os.listdir(folder_path)

    # Use a thread pool to launch independent optimization jobs for each task.
    pool = ThreadPool(64)

    # Process files in parallel.
    pool.map(model_select, files)
    
    # Close the thread pool
    pool.close()
    pool.join()
