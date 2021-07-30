import pandas as pd
from matplotlib import pyplot as plt


def analyze_results(results_database_name, env_variables, model, pipeline):
    C, S, gamma_percent = env_variables
    results = pd.read_csv(results_database_name, delimiter=';')
    num_instances = len(results)

    print("\nModel:", model)
    print("Gamma Percent:", gamma_percent)
    print("Pipeline:", pipeline)

    print("\nNumber of ML instances:", num_instances)
    print("Objective difference:", sum(results["obj_reg"] - results["obj_ML"]))
    print("Second-stage difference:", sum(results["eta_reg"] - results["eta_ML"]))

    print("\nAverage time (regular):", sum(results["total_time_reg"]) / num_instances)
    print("Average time (ML):", sum(results["total_time_ML"]) / num_instances)
    print("Average time difference:", sum(results["total_time_reg"] - results["total_time_ML"]) / num_instances)

    print("\nAverage # scenarios (regular):", round(sum(results["num_scenarios_reg"]) / num_instances, 3))
    print("Average # scenarios (ML):", round(sum(results["num_scenarios_ML"]) / num_instances, 3))
    print("Average # scenarios difference:", round(sum(results["num_scenarios_reg"] - results["num_scenarios_ML"]) / num_instances, 3))
