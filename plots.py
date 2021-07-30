from joblib import load
from matplotlib import pyplot as plt
import numpy as np

from Environment.Env import Environment

# Define environment variables
C = 10  # Number of customers
S = 5   # Number of facilities
gamma_percent = 0.9
env_variables = (C, S, gamma_percent)

model = 'LogR'     # LogR / NN / RF

# Set file name
models = {'LogR': 'Logistic_Regression', 'NN': 'Neural_Network', 'RF': 'Random_Forest'}
GP = {0.9: 0, 0.8: 1}
model_name = 'Models/{}/Model_GP{}.joblib'.format(models[model], GP[gamma_percent])

predictors = load(model_name)

env = Environment(C, S, gamma_percent)
scenarios = env.vertices

if model == 'LogR':
    for i, val in enumerate(predictors.values()):
        print(i)
        scenario = scenarios[i]
        zeroes = np.where(scenario == 0)

        # Get model coefficients
        mod = val['Model']
        coef = mod.coef_[0]

        # Split demand / delta coefficients
        demand_coef = coef[np.arange(0, 20, 2)]
        delta_coef = coef[np.arange(1, 20, 2)]

        # Make plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        # title = 'Coefficients of ' + models[model] + ' model for scenario ' + str(i)
        # fig.suptitle(title)
        ax1.title.set_text('Demand coefficients')
        ax1.set_ylabel('Coefficient')
        ax1.set_xlabel('Customer')
        ax2.title.set_text('Delta coefficients')
        ax2.set_ylabel('Coefficient')
        ax2.set_xlabel('Customer')

        ax1.scatter(np.arange(10), demand_coef[np.arange(10)])
        ax1.scatter(zeroes, demand_coef[zeroes], color='r')
        ax2.scatter(np.arange(10), delta_coef[np.arange(10)])
        ax2.scatter(zeroes, delta_coef[zeroes], color='r')

        ax1.plot(np.arange(0, 10), np.zeros(10), color='grey', linestyle='dashed')
        ax2.plot(np.arange(0, 10), np.zeros(10), color='grey', linestyle='dashed')
        name = 'Figures/' + model + '_' + str(i) + '.png'
        plt.savefig(name)
        # plt.show()