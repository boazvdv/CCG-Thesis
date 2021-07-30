import pandas as pd
from joblib import dump, load

from Main_Functions.learning_functions import fit_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

C = 10
S = 5
gamma_percent = 0.8
model = 'RF'     # LogR / NN / RF

# Set file names
models = {'LogR': 'Logistic_Regression', 'NN': 'Neural_Network', 'RF': 'Random_Forest'}
GP = {0.9: 0, 0.8: 1}
data_name = "Data/Training_Data/Training_Data_GP{}.csv".format(GP[gamma_percent])
model_name = 'Models/{}/Model_GP{}.joblib'.format(models[model], GP[gamma_percent])

data = pd.read_csv(data_name, delimiter=';')
fit_new_model = False

# Fit model
if fit_new_model:
    predictors = fit_model(data, model)
    dump(predictors, model_name)
else:
    predictors = load(model_name)

X_test_list = []
Y_test_list = []

scen_labels = data.scenario.unique()
for label in scen_labels:
    data_scen = data[data['scenario'] == label]

    X = data_scen.iloc[:, 0:-1]
    Y = data_scen.iloc[:, -1]
    _, X_test_scen, _, Y_test_scen = train_test_split(X, Y, test_size=0.2, random_state=0)

    X_test_list.append(X_test_scen)
    Y_test_list.append(Y_test_scen)

X_test = pd.concat(X_test_list)
Y_test = pd.concat(Y_test_list)

Y_pred = []

i = 0
N = len(X_test)
for _, row in X_test.iterrows():
    scenario = row['scenario']

    row.drop('scenario', inplace=True)
    row = pd.DataFrame([row])

    clf = predictors[scenario]['Model']
    t = predictors[scenario]['Threshold']

    pred = (clf.predict_proba(row)[0, 1] >= t).astype(int)
    Y_pred.append(pred)

    if i % 100 == 0:
        print(i, '/', N)
    i += 1

print(metrics.classification_report(Y_test, Y_pred))