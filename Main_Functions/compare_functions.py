import csv
from Environment.Env_ML import Environment_ML
from CCG.algorithm import ccg_function_vertices


# Run CCG with and without ML
def compare_ML_to_normal(env_variables, predictors, num_instances_ML, results_database_name, pipeline, model):
    C, S, gamma_percent = env_variables

    for i in range(num_instances_ML):
        print(i, "/", num_instances_ML, "ML instances generated")
        # Run C&CG
        env = Environment_ML(C, S, gamma_percent, predictors=predictors, model=model)
        result = ccg_function_vertices(env)
        result_ML = ccg_function_vertices(env, pipeline=pipeline)

        row = []
        for res in [result, result_ML]:
            row.append(round(res["obj"], 2))
            row.append(round(res["eta"], 2))
            row.append(len(res["vertices"]))
            row.append(res["total_time"])

        with open(results_database_name, mode='a') as ccg_results_file:
            ccg_results_writer = csv.writer(ccg_results_file, delimiter=';')
            ccg_results_writer.writerow(row)
    return
