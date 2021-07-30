import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

pd.options.mode.chained_assignment = None  # default='warn'


def fit_model(data, model):
    scen_labels = data.scenario.unique()    # Get scenario labels
    predictors = {}
    thresholds = np.arange(0.01, 0.51, 0.01)
    for label in scen_labels:
        print("Fitting", model, "for scenario", label)
        # Split data by scenario
        data_scen = data[data['scenario'] == label]
        data_scen.drop('scenario', inplace=True, axis=1)

        # Split training / test set
        X = data_scen.iloc[:, 0:-1]
        Y = data_scen.iloc[:, -1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        # Fit model
        if model == 'NN':
            clf = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000)
        elif model == 'LogR':
            clf = LogisticRegression(max_iter=2000)
        elif model == 'RF':
            clf = RandomForestClassifier()
        clf.fit(X_train, Y_train)

        # Find threshold with best F1-score:
        Y_pred_list = []
        for t in thresholds:
            Y_pred = (clf.predict_proba(X_test)[:, 1] >= t).astype(int)
            Y_pred_list.append((Y_pred, t))

        best_score = 0
        best_t = 0
        for Y_pred, t in Y_pred_list:
            current_score = round(metrics.f1_score(Y_test, Y_pred, pos_label=1), 2)
            if current_score > best_score:
                best_score = current_score
                best_t = round(t, 2)

        Y_pred = (clf.predict_proba(X_test)[:, 1] >= best_t).astype(int)
        print("Threshold", best_t)
        print(metrics.classification_report(Y_test, Y_pred))

        predictors[label] = {"Model": clf, "Threshold": best_t}

    return predictors
