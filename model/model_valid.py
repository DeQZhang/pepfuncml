"""Validate the chosen model for each peptide task with cross-validation."""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from multiprocessing import Pool
from xgboost import XGBClassifier


def model_select(file_name):
    """Run cross-validation for the model selected in the model summary table."""
    folder_path = 'data/embedding/Dimensionality_Reduction'
    file_path=os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, index_col=None)
    

    opt_model = pd.read_csv('model/model_table.csv')

 
    model_name = opt_model[opt_model['peptidase'] == file_name.split(".")[0]]['opt_model'].values[0]

    # Prepare the full dataset because validation uses cross-validation rather than a holdout split.
    X = df.drop('label', axis=1).values
    Y = df['label'].values



    # Split the dataset
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    
    # Recreate the candidate model pool so the selected name can be resolved to an estimator.
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

    clf = models[model_name]
    # Use shuffled K-fold splits to estimate the selected model's stability.
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Evaluate the chosen model on all folds and print the resulting score vector.
    scores = cross_val_score(clf, X, Y, cv=kf)

    print(file_name.split(".")[0],scores)


    # # Train and evaluate the models
    # accuracies = {}
    # for name, model in models.items():
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     accuracy = accuracy_score(y_test, y_pred)
    #     accuracies[name] = accuracy
    #     pass

    # max_key = max(accuracies, key=accuracies.get)


    # # Build the DataFrame
    # df = pd.DataFrame({"Model": list(accuracies.keys()), "Accuracy": list(accuracies.values())})

    # # Draw the bar chart
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x="Model", y="Accuracy", data=df)
    # plt.title(f"{file_name.split(".")[0]+'.'+file_name.split(".")[1]}")
    # plt.xlabel("Models", fontsize=14)
    # plt.ylabel("Accuracy", fontsize=14)

    # plt.savefig(f'picture/model_select/{file_name.split(".")[0]+'.'+file_name.split(".")[1]+f'({max_key})'}.png')


if __name__ == '__main__':
    # Read all reduced feature tables before dispatching validation jobs.
    folder_path = 'data/embedding/Dimensionality_Reduction'
    files = os.listdir(folder_path)
    data = {}


    # Parallelize validation across tasks.
    pool = Pool(7)

    # Process sequence data in parallel.
    pool.map(model_select, files)
    
    # Close the process pool
    pool.close()
    pool.join()