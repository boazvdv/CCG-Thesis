import pandas as pd
from joblib import dump, load

from Main_Functions.analyze_functions import analyze_results
from Main_Functions.compare_functions import compare_ML_to_normal
from Main_Functions.data_functions import generate_data
from Main_Functions.learning_functions import fit_model


# Define environment variables
C = 10  # Number of customers
S = 5   # Number of facilities
gamma_percent = 0.8
env_variables = (C, S, gamma_percent)

# Aim of program
model = 'RF'     # LogR / NN / RF
pipeline = 1       # 1 / 2

gen_data = 0        # Set to 1 to generate data
fit_new_model = 0   # Set to 1 to fit new model
run_algorithm = 0   # Set to 1 to run C&CG with/without ML
print_results = 1   # Print analysis of results

# Set file names
models = {'LogR': 'Logistic_Regression', 'NN': 'Neural_Network', 'RF': 'Random_Forest'}
GP = {0.9: 0, 0.8: 1}

data_name = "Data/Training_Data/Training_Data_GP{}.csv".format(GP[gamma_percent])
results_database_name = "Data/Results/{}/Pipeline_{}_GP{}.csv".format(models[model], pipeline, GP[gamma_percent])
model_name = 'Models/{}/Model_GP{}.joblib'.format(models[model], GP[gamma_percent])

# Generate data
if gen_data:
    num_instances_data = 5000
    generate_data(env_variables, num_instances_data, data_name)

# Fit model
if fit_new_model:
    data = pd.read_csv(data_name, delimiter=';')
    predictors = fit_model(data, model)
    dump(predictors, model_name)
else:
    predictors = load(model_name)

# Run algorithm with and without ML
if run_algorithm:
    num_instances_ML = 1000
    compare_ML_to_normal(env_variables, predictors, num_instances_ML, results_database_name, pipeline, model)

# Print stats about results
if print_results:
    analyze_results(results_database_name, env_variables, models[model], pipeline)
