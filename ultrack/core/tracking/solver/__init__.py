from ultrack.core.tracking.solver.cvxpy_solver import CVXPySolver
from ultrack.core.tracking.solver.heuristic.heuristic_solver import HeuristicSolver

try:
    from ultrack.core.tracking.solver.gurobi_solver import GurobiSolver
except Exception as e:
    from rich import print

    print(e)
