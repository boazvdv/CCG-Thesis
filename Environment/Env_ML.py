from .Env import Environment
import pandas as pd
import numpy as np


class Environment_ML(Environment):
    def __init__(self, C, S, gamma_percent, inst_num=0, predictors={}, model=''):
        Environment.__init__(self, C, S, gamma_percent, inst_num)
        self.predictors = predictors
        self.active_set = []
        self.status = 'inactive'
        self.model = model
        self.scenarios_prioritized = get_important_scenarios(self)


# Get important scenarios using LR predictors
def get_important_scenarios(env):
    if env.predictors == {}:
        print("No predictors found")
        return []

    important_scenarios = []
    unimportant_scenarios = []

    # Get all vertices of uncertainty polytope
    scenarios = env.vertices
    predictors = env.predictors
    model = env.model

    # Gather demand Data
    customer_demand_delta = [[round(cus.demand, 2), round(cus.delta, 2)] for cus in env.customers.values()]
    row = []
    for dem_del in customer_demand_delta:
        row.append(dem_del[0])
        row.append(dem_del[1])
    row_data = pd.DataFrame([row])

    for i, scen in enumerate(scenarios):
        pred = predictors[i]["Model"]
        threshold = predictors[i]["Threshold"]
        if model == 'NN' or model == 'LogR' or model == 'RF':
            important = pred.predict_proba(row_data)[:, 1] > threshold
        elif model == 'LinR':
            important = pred.predict(row_data) > threshold

        if important:
            important_scenarios.append(scen)
        else:
            unimportant_scenarios.append(scen)

    return [np.array(important_scenarios), np.array(unimportant_scenarios)]
