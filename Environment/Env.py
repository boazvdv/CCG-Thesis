import numpy as np
import pypoman as pm


class Customer:
    def __init__(self):
        self.demand = np.random.uniform(10, 500)    # Nominal demand
        self.delta = np.random.uniform(0.1, 0.5) * self.demand  # Deviation


class Facility:
    def __init__(self, capacity):
        self.fix_cost = np.random.uniform(1, 10)    # Fixed cost of opening facility
        self.var_cost = np.random.uniform(0.1, 1)   # Variable cost of producing items
        self.capacity = capacity    # Capacity of facility


class Environment:
    def __init__(self, C, S, gamma_percent, inst_num=0):
        self.z_hat = 1  # z_hat
        self.C = C  # Number of customers
        self.S = S  # Number of facilities
        self.gamma_percent = gamma_percent  # Uncertainty
        self.gamma = gamma_percent * self.C # Uncertainty
        self.inst_num = inst_num    # Instance number

        # Make customers dictionary
        customers = dict()
        for i in np.arange(self.C):
            customers[i] = Customer()
        self.customers = customers

        # Make facilities dictionary
        self.capacity_full = sum([self.customers[c].demand + self.z_hat * self.customers[c].delta for c in np.arange(C)])

        facilities = dict()
        for s in np.arange(self.S):
            facilities[s] = Facility(self.capacity_full)
        self.facilities = facilities

        # Make distances between facilities and customers
        trans_cost = dict()
        for s, facility in facilities.items():
            for c, customer in customers.items():
                trans_cost[s, c] = np.random.uniform(0, 10)
        self.trans_cost = trans_cost

        # 'Big M' for subproblem
        self.bigM = sum([self.customers[c].demand + self.z_hat * self.customers[c].delta for c in np.arange(C)])

        # Upper bound (Unused?)
        self.upper_bound = self.S * self.C * max([trans_cost[s, c] for s in np.arange(S) for c in np.arange(C)]) * self.bigM + \
                           self.capacity_full * max([self.facilities[s].var_cost for s in np.arange(S)]) + \
                           sum([self.facilities[s].fix_cost for s in np.arange(S)])

        # Vertices of uncertainty set
        self.vertices = vertex_fun(self)

        # Initial uncertainty, Shape = (C,)
        self.init_uncertainty = np.zeros(C, dtype=np.float)


def vertex_fun(env):
    C = env.C   # Number of customers
    num_consts = 1+2*C  # Number of constraints

    # Define constraints in halfspace representation for uncertainty set
    A = np.concatenate((np.ones([1, C]), np.eye(C), -1*np.eye(C)), axis=0) # num_consts rows, C columns
    b = np.concatenate((np.array([env.gamma]).reshape(1, 1), env.z_hat * np.ones([1, C]), np.zeros([1, C])),
                       axis=1).reshape(num_consts) # Array of num_consts values

    # Get vertices of polytope
    vertices_array = np.array(pm.compute_polytope_vertices(A, b))

    # Find only vertices for which the maximum uncertainty budget is used
    max_sum_vertices = max(np.sum(vertices_array, axis=1))
    needed_vertices = vertices_array[np.where(np.sum(vertices_array, axis=1) >= max_sum_vertices)]

    return needed_vertices
