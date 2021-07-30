import numpy as np
import gurobipy as gp
from gurobipy import GRB


def master_problem(env, z_set):
    S = env.S  # Number of facilities
    C = env.C  # Number of customers
    custs = env.customers  # Customers dictionary
    facs = env.facilities  # Facilities dictionary
    N = len(z_set)  # Current size of uncertainty set

    # Model
    mp = gp.Model("Master Problem")  # Define model
    mp.Params.OutputFlag = 0  # Disable Gurobi solver output

    # Variables
    eta = mp.addVar(lb=0, vtype=GRB.CONTINUOUS)  # Second-stage cost variable
    x = mp.addVars(S, name="x", vtype=GRB.BINARY)  # Boolean variable defining open/closed facility
    p = mp.addVars(S, lb=0, name="p", vtype=GRB.CONTINUOUS)  # Production level per facility (>= 0)

    y_index = [(s, c) for s in np.arange(S) for c in np.arange(C)]  # Indices for facilities-customers
    y = dict()  # Dictionary with second-stage transport decisions for each realization of uncertainty
    for i in np.arange(N):
        y[i] = mp.addVars(y_index, lb=0, name="y_{}".format(i), vtype=GRB.CONTINUOUS)  # >= 0

    # Objective function to be minimized
    mp.setObjective(gp.quicksum(facs[s].var_cost * p[s] + facs[s].fix_cost * x[s] for s in np.arange(S)) + eta,
                    GRB.MINIMIZE)

    # First-stage constraint - production level doesn't exceed facility capacity
    mp.addConstrs(p[s] <= facs[s].capacity * x[s] for s in np.arange(S))

    # Combination constraint - total items transported from facility doesn't exceed production level
    for i in np.arange(N):
        mp.addConstrs(gp.quicksum(y[i][s, c] for c in np.arange(C)) <= p[s] for s in np.arange(S))

    # Second-stage objective constraints - total transport cost
    for i in np.arange(N):
        mp.addConstr(eta >= gp.quicksum(env.trans_cost[s, c] * y[i][s, c] for s in np.arange(S) for c in np.arange(C)))

    # Uncertainty constraint - (uncertain) customer demand is always met
    for i, z in enumerate(z_set):
        mp.addConstrs(
            (gp.quicksum(y[i][s, c] for s in np.arange(S)) >= (custs[c].demand + z[c] * custs[c].delta)) for c in
            np.arange(C))

    # Solve model
    mp.optimize()  # Gurobi solver function

    try:
        x_sol = np.array([var.X for i, var in x.items()])  # Get 'x' solution variables for master problem
    except:
        print("Instance {}: INFEASIBLE MASTER PROBLEM".format(env.inst_num))  # Infeasibility exception

    p_sol = np.array([var.X for i, var in p.items()])  # Get 'p' solution variables for master problem

    y_sol = dict()  # Get 'y' solution variables for master problem, for each realization of uncertainty
    for j in np.arange(N):
        y_sol[j] = np.array([var.X for i, var in y[j].items()]).reshape(S, C)

    eta_sol = eta.X  # Get 'eta' solution (second-stage cost) for master problem
    theta = mp.objVal  # Get value of objective function (total cost)

    return theta, eta_sol, {"p": p_sol, "x": x_sol}, y_sol


def sub_problem(env, y_input):
    S = env.S  # Number of facilities
    C = env.C  # Number of customers
    custs = env.customers  # Customers dictionary
    N = len(y_input)  # Current size of uncertainty set
    z = env.vertices  # Vertices of uncertainty polytope

    # Model
    ic = gp.Model("Infeasible check")  # Define model
    ic.Params.OutputFlag = 0  # Disable Gurobi solver output
    ic.Params.IntFeasTol = 1e-9  # Reduce 'Integer Feasibility Tolerance' to avoid trickle flow problems

    # Variables
    z_bin = ic.addVars(len(z), vtype=GRB.BINARY)  # Boolean variable for violating scenario
    b_index = [(i, c) for i in np.arange(N) for c in np.arange(C)]  # Indices for b variable
    b = ic.addVars(b_index, vtype=GRB.BINARY)  # Boolean b variable
    zeta = ic.addVar(lb=-env.capacity_full, vtype=GRB.CONTINUOUS)  # Sum ysc over s is maximum the capacity!

    # Objective function to maximize
    ic.setObjective(zeta, GRB.MAXIMIZE)

    # Constraints
    for i, y in y_input.items():  # Constraint for finding violating scenario
        ic.addConstrs(zeta + env.bigM * b[i, c] <= -sum(y[s, c] for s in np.arange(S)) + custs[c].demand +
                      gp.quicksum(z[l][c] * z_bin[l] for l in np.arange(len(z))) * custs[c].delta + env.bigM
                      for c in np.arange(C))
    ic.addConstrs(gp.quicksum(b[i, c] for c in np.arange(C)) == 1 for i in np.arange(N))  # Constraint binary variable
    ic.addConstr(gp.quicksum(z_bin[l] for l in np.arange(len(z))) == 1)  # Constraint vertex variable

    # Solve model
    ic.optimize()

    z_bin_sol = np.where(np.array([var.X for i, var in z_bin.items()]))  # Indicate index of violating scenario
    z_sol = z[z_bin_sol][0].T  # Get violating scenario
    zeta_sol = zeta.X  # Zeta indicating robustness
    b_sol = np.array([var.X for i, var in b.items()])  # 'b' boolean variables

    return zeta_sol, z_sol
