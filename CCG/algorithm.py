from .master_sub import master_problem, sub_problem
import numpy as np
import time


def ccg_function_vertices(env, vertices_only=False, pipeline=0):
    # Initialize
    start_time = time.time()    # Get start time of complete algorithm
    z_set = [env.init_uncertainty]  # Define initial uncertainty set (zeros)

    # Solve master problem once with vertices of uncertainty polytope to obtain initial min. cost and 'x' decision
    obj_init, eta_init, x_init, y_init = master_problem(env, env.vertices)

    # Immediately return result for CCG using all vertices of uncertainty polytope
    if vertices_only:
        end_time = time.time()
        total_time = end_time - start_time
        result_dict = {"x": x_init, "y": y_init, "eta": eta_init, "obj": obj_init, "vertices": env.vertices,
                       "total_time": total_time, "zeta": 'NA'}
        return result_dict

    it_time = dict()  # Dictionary for iteration time
    it = 0  # Iteration number
    zeta = 0    # Zeta for robustness
    obj_new, eta_new, x_new, y_new = master_problem(env, z_set)

    if pipeline and obj_init - obj_new > 1e-4:
        important_scenarios = env.scenarios_prioritized[0]
        unimportant_scenarios = env.scenarios_prioritized[1]

        # Solve MP using important scenarios before attempting subproblem
        if pipeline == 1:
            for scen in important_scenarios:
                z_set.append(scen)
                obj_new, eta_new, x_new, y_new = master_problem(env, z_set)
                if obj_init - obj_new <= 1e-4:
                    break
            env.vertices = unimportant_scenarios

        # Solve SP using important scenarios before looking at unimportant scenarios
        elif pipeline == 2:
            env.status = 'important'
            env.vertices = important_scenarios

    # Iteratively add scenarios, but stop if the new objective cost exceeds the initially found cost
    while obj_init - obj_new > 1e-4:
        it += 1  # Increment iteration number
        it_time[it-1] = time.time() - start_time  # Record iteration time

        # Solve master problem
        obj_new, eta_new, x_new, y_new = master_problem(env, z_set)

        # Solve subproblem with 'y' decision (i.e. search for violating scenario with current decision)
        zeta, z_sol = sub_problem(env, y_new)

        if zeta <= 1e-4 or obj_init - obj_new <= 1e-4:   # Stop if robust
            if pipeline == 2 and env.status == 'important':
                env.vertices = unimportant_scenarios
                env.status = 'unimportant'
            else:
                break
        else:
            z_set.append(z_sol)     # If not robust, add violating scenario to uncertainty set

    end_time = time.time()
    total_time = end_time - start_time
    z_set_return = np.array(z_set[1:])  # Set of scenarios to return to main

    result_dict = {"x": x_new, "y": y_new, "eta": eta_new, "obj": obj_new, "vertices": z_set_return,
                   "it_time": it_time, "total_time": total_time, "zeta": zeta}

    return result_dict
