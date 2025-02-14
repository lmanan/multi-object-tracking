import gurobipy as gp

import gurobipy as gp

try:
    # Create a basic environment to test license access
    env = gp.Env(empty=True)
    env.start()
    print("Gurobi license is accessible.")
except gp.GurobiError as e:
    print(f"Gurobi license error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")


