import csv

from CCG.algorithm import ccg_function_vertices
from CCG.delete_scenarios import delete_scenarios
from Environment.Env import Environment


# # MAIN FUNCTION
def generate_data(env_variables, num_instances_data, data_name):
    with open(data_name, mode='w') as ccg_data_file:
        ccg_data_writer = csv.writer(ccg_data_file, delimiter=';')
        C, S, gamma_percent = env_variables

        # Column names
        col_names = ['scenario']
        for i in range(C):
            col_names.append("demand_{}".format(str(i)))
            col_names.append("delta_{}".format(str(i)))
        col_names.append('label')
        ccg_data_writer.writerow(col_names)

        # Get all vertices of uncertainty polytope
        env = Environment(C, S, gamma_percent)
        scenarios = env.vertices

        # Data
        for i in range(num_instances_data):
            env = Environment(C, S, gamma_percent)
            customer_demand_delta = [[round(cus.demand, 2), round(cus.delta, 2)] for cus in env.customers.values()]

            result = ccg_function_vertices(env, vertices_only=True)
            result_deleted = delete_scenarios(result, env)
            important_scenarios = result_deleted['important_scenarios']

            # Check progress
            if i % 25 == 0:
                print(i, "/", num_instances_data, "data instances generated")

            for j, scenario in enumerate(scenarios):
                row = [j]
                for dem_del in customer_demand_delta:
                    row.append(dem_del[0])
                    row.append(dem_del[1])
                row.append(int(scenario_in_set(scenario, important_scenarios)))
                ccg_data_writer.writerow(row)
    return


# # HELPER FUNCTIONS
# Check if scenario in set
def scenario_in_set(scenario, scenario_set):
    for scenario_comp in scenario_set:
        comparison = scenario == scenario_comp
        equal = comparison.all()
        if equal:
            return True
    return False
