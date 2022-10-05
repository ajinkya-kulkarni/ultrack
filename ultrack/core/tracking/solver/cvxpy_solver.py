import logging
from typing import List

import cvxpy as cp
import numpy as np
import pandas as pd
from cvxpy.expressions.expression import Expression
from numpy.typing import ArrayLike
from tqdm import tqdm

from ultrack.config.config import TrackingConfig
from ultrack.core.tracking.solver.base_solver import BaseSolver
from ultrack.core.tracking.solver.utils import indices_to_solution_dataframe

LOG = logging.getLogger(__name__)


class CVXPyNode:
    def __init__(
        self,
        appear_weight: float,
        disappear_weight: float,
        division_weight: float,
    ) -> None:
        """Helper class for objective oriented problem construction."""
        self.node_var = cp.Variable(boolean=True)
        self._appear_var = cp.Variable(boolean=True)
        self._disappear_var = cp.Variable(boolean=True)
        self._division_var = cp.Variable(boolean=True)
        self._appear_weight = cp.Parameter()
        self._appear_weight.value = appear_weight
        self._disappear_weight = cp.Parameter()
        self._disappear_weight.value = disappear_weight
        self._division_weight = cp.Parameter()
        self._division_weight.value = division_weight
        self._in_edges = []
        self._out_edges = []

    def add_in_edge(self, edge: cp.Variable) -> None:
        self._in_edges.append(edge)

    def add_out_edge(self, edge: cp.Variable) -> None:
        self._out_edges.append(edge)

    @property
    def constraints(self) -> List[Expression]:
        """Single-incoming node, flow conservation and division constraints for a single node."""
        return [
            # single incoming node
            cp.sum(self._in_edges) + self._appear_var == self.node_var,
            # flow conservation
            self.node_var + self._division_var
            == cp.sum(self._out_edges) + self._disappear_var,
            self.node_var >= self._division_var,  # existance division
        ]

    @property
    def objective(self) -> Expression:
        """Objective variable per node."""
        return (
            self._appear_weight * self._appear_var
            + self._disappear_weight * self._disappear_var
            + self._division_weight * self._division_var
        )


class CVXPySolver(BaseSolver):
    def __init__(
        self,
        config: TrackingConfig,
    ) -> None:
        """cvxpy tracking ILP solver

        Parameters
        ----------
        config : TrackingConfig
            Tracking configuration parameters.
        """

        self._config = config
        self.reset()

    def reset(self) -> None:
        """Sets model to an empty state."""
        self._nodes = {}
        self._edges = []
        self._edges_keys = []
        self._constraints = []
        self._weights = None
        self._problem = None

    def add_nodes(
        self, indices: ArrayLike, is_first_t: ArrayLike, is_last_t: ArrayLike
    ) -> None:
        """Add nodes slack variables to cvxpy model.

        Parameters
        ----------
        indices : ArrayLike
            Nodes indices.
        is_first_t : ArrayLike
            Boolean array indicating if it belongs to first time point and it won't receive appearance penalization.
        is_last_t : ArrayLike
            Boolean array indicating if it belongs to last time point and it won't receive disappearance penalization.
        """
        if len(self._nodes) > 0:
            raise ValueError("Nodes have already been added.")

        self._assert_same_length(
            indices=indices, is_first_t=is_first_t, is_last_t=is_last_t
        )

        LOG.info(f"# {np.sum(is_first_t)} nodes at starting `t`.")
        LOG.info(f"# {np.sum(is_last_t)} nodes at last `t`.")

        appear_weight = np.logical_not(is_first_t) * self._config.appear_weight
        disappear_weight = np.logical_not(is_last_t) * self._config.disappear_weight

        indices = indices.tolist()

        self._nodes = {
            index: CVXPyNode(a_w, d_w, self._config.division_weight)
            for index, a_w, d_w in tqdm(
                zip(indices, appear_weight, disappear_weight),
                "Adding nodes to solver",
                total=len(indices),
            )
        }

    def add_edges(
        self, sources: ArrayLike, targets: ArrayLike, weights: ArrayLike
    ) -> None:
        """Add edges to model and applies weights link function from config.

        Parameters
        ----------
        source : ArrayLike
            Array of integers indicating source indices.
        targets : ArrayLike
            Array of integers indicating target indices.
        weights : ArrayLike
            Array of weights, input to the link function.
        """
        if len(self._edges) > 0:
            raise ValueError("Edges have already been added.")

        self._assert_same_length(sources=sources, targets=targets, weights=weights)

        self._weights = self._config.apply_link_function(weights)

        LOG.info(f"transformed edge weights {self._weights}")

        self._edges = cp.Variable(len(self._weights), boolean=True)

        self._edges_keys = []
        for i, (s, t) in tqdm(
            enumerate(zip(sources, targets)),
            "Adding edges to solver",
            total=len(self._weights),
        ):
            edge = self._edges[i]
            self._nodes[t].add_in_edge(edge)
            self._nodes[s].add_out_edge(edge)
            self._edges_keys.append((s, t))

    def set_standard_constraints(self) -> None:
        """Resets constraints and add biologcal contraints from node class."""
        self._constraints = sum(
            (
                node.constraints
                for node in tqdm(
                    self._nodes.values(),
                    "Adding standard constraints",
                    total=len(self._nodes),
                )
            ),
            [],
        )

    def add_overlap_constraints(self, sources: ArrayLike, targets: ArrayLike) -> None:
        """Add constraints such that `source` and `target` can't be present in the same solution.

        Parameters
        ----------
        source : ArrayLike
            Source nodes indices.
        target : ArrayLike
            Target nodes indices.
        """
        self._constraints += [
            self._nodes[s].node_var + self._nodes[t].node_var <= 1
            for s, t in tqdm(
                zip(sources, targets), "Adding overlap constraints", total=len(targets)
            )
        ]

    def enforce_node_to_solution(self, indices: ArrayLike) -> None:
        """Constraints given nodes' variables to 1.

        Parameters
        ----------
        indices : ArrayLike
            Nodes indices.
        """
        self._constraints += [self._nodes[i].node_var >= 1 for i in indices]

    def optimize(self) -> float:
        """Optimizes cvxpy model."""
        self._problem = cp.Problem(
            cp.Maximize(
                cp.sum(self._weights[np.newaxis, ...] @ self._edges)
                + cp.sum([node.objective for node in self._nodes.values()])
            ),
            self._constraints,
        )
        # NOTE: see comments regarding ILP solvers in:
        # https://www.cvxpy.org/tutorial/advanced/index.html#mixed-integer-programs
        self._problem.solve(
            solver="ECOS_BB", mi_rel_eps=self._config.solution_gap, verbose=True
        )
        return self._problem.value

    def solution(self) -> pd.DataFrame:
        """Returns the solution as dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe indexed by nodes as indices and their parent (NA if orphan).
        """
        if self._problem is None:
            raise ValueError(
                "cvxpy solver must be optimized before returning solution."
            )

        nodes = [k for k, var in self._nodes.items() if var.node_var.value > 0.5]
        edges = np.asarray(
            [k for k, var in zip(self._edges_keys, self._edges) if var.value > 0.5]
        )

        return indices_to_solution_dataframe(nodes, edges)
