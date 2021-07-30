import numpy as np

from .master_sub import master_problem


def delete_scenarios(result_dict, env):
    z_set = list(result_dict['vertices'])
    obj_new = result_dict['obj']

    important = []

    for i, scenario in enumerate(z_set):
        z_set_del = z_set[:i] + z_set[i+1:]
        obj_del, _, x_del, _ = master_problem(env, z_set_del)

        if abs(obj_del - obj_new) > 1e-4:
            important.append(i)

    z_set_important = []
    for i in important:
        z_set_important.append(z_set[i])
    z_set_important = np.array(z_set_important)
    obj_del, eta_del, x_del, y_del = master_problem(env, z_set_important)

    deleted_result_dict = {"x_del": x_del, "y_del": y_del, "obj_del": obj_del, "eta_del": eta_del,
                           "important_scenarios": z_set_important}

    return deleted_result_dict
