import logging

import cvxpy as cp
import numpy as np
import pandas as pd
from cvxpy.expressions.expression import Expression
from numpy.typing import ArrayLike
from scipy import sparse
from skimage.util._map_array import ArrayMap

from ultrack.config.config import TrackingConfig
from ultrack.core.tracking.solver.base_solver import BaseSolver
from ultrack.core.tracking.solver.utils import indices_to_solution_dataframe

LOG = logging.getLogger(__name__)


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
        self._n_slack_vars = 4
        self.reset()

    def reset(self) -> None:
        """Sets model to an empty state."""
        self._n_nodes = 0
        self._n_edges = 0
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
        if self._n_nodes > 0:
            raise ValueError("Nodes have already been added.")

        self._assert_same_length(
            indices=indices, is_first_t=is_first_t, is_last_t=is_last_t
        )

        LOG.info(f"# {np.sum(is_first_t)} nodes at starting `t`.")
        LOG.info(f"# {np.sum(is_last_t)} nodes at last `t`.")

        appear_weight = np.logical_not(is_first_t) * self._config.appear_weight
        disappear_weight = np.logical_not(is_last_t) * self._config.disappear_weight

        self._backward_map = np.array(indices, copy=True)
        self._forward_map = ArrayMap(np.asarray(indices), np.arange(len(indices)))
        self._n_nodes = len(self._backward_map)

        self._weights = np.concatenate(
            (
                np.zeros(self._n_nodes),
                appear_weight,
                disappear_weight,
                np.full(self._n_nodes, self._config.division_weight),
            ),
            axis=0,
            dtype=np.float32,
        )

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
        if self._n_edges > 0:
            raise ValueError("Edges have already been added.")

        self._assert_same_length(sources=sources, targets=targets, weights=weights)

        weights = self._config.apply_link_function(weights)

        LOG.info(f"transformed edge weights {weights}")
        self._weights = np.concatenate(
            (self._weights, weights),
            axis=0,
            dtype=np.float32,
        )
        self._weights = cp.Parameter(len(self._weights), value=self._weights)

        self._out_edges = self._forward_map[np.asarray(sources)]
        self._in_edges = self._forward_map[np.asarray(targets)]
        self._n_edges = len(self._out_edges)

        # n_nodes x 4 + n_edges
        size = self._n_slack_vars * self._n_nodes + self._n_edges

        # (nodes, appear, disappear, division, edges)
        self._variables = cp.Variable(size, boolean=True)
        self._shape = (self._n_nodes, size)

    def _sparse_nodes_ids_eye_matrix(self, offset: int = 0) -> sparse.csr_matrix:
        """Creates a sparse rectangular identity matrix of size (n_nodes, n_vars)

        Parameters
        ----------
        offset : int, optional
            Diagonal offset, it's multiplied by n_nodes, by default 0

        Returns
        -------
        sparse.csr_matrix
            Compressed row sparse matrix of shape (n_nodes, n_vars)
        """
        node_ids = np.arange(self._n_nodes)
        values = np.ones(self._n_nodes, dtype=np.float32)
        return sparse.csr_matrix(
            (values, (node_ids, offset * self._n_nodes + node_ids)),
            shape=self._shape,
        )

    def _single_in_edge_constraint(self) -> Expression:
        """Creates a contraint such that there's only a single in coming link per node."""
        in_edges = sparse.csr_matrix(
            (
                np.ones(self._n_edges, dtype=np.float32),
                (
                    self._in_edges,
                    self._n_slack_vars * self._n_nodes + np.arange(self._n_edges),
                ),
            ),
            shape=self._shape,
        )

        appear = self._sparse_nodes_ids_eye_matrix(offset=1)
        nodes = self._sparse_nodes_ids_eye_matrix()

        return (in_edges + appear - nodes) @ self._variables == 0

    def _flow_conservation_constraint(self) -> Expression:
        """Creates a constraint where the number of in coming and out going links plus slack variables must be equal."""
        nodes = self._sparse_nodes_ids_eye_matrix()
        disappear = self._sparse_nodes_ids_eye_matrix(offset=2)
        division = self._sparse_nodes_ids_eye_matrix(offset=3)
        out_edges = sparse.csr_matrix(
            (
                np.ones(self._n_edges, dtype=np.float32),
                (
                    self._out_edges,
                    self._n_slack_vars * self._n_nodes + np.arange(self._n_edges),
                ),
            ),
            shape=self._shape,
        )
        return (nodes + division - out_edges - disappear) @ self._variables == 0

    def _division_constraint(self) -> Expression:
        """Creates constraint where a node must exist for it to divide."""

        nodes = self._sparse_nodes_ids_eye_matrix()
        division = self._sparse_nodes_ids_eye_matrix(offset=3)

        return (nodes - division) @ self._variables >= 0

    def set_standard_constraints(self) -> None:
        """Resets constraints and add biologcal contraints from node class."""
        self._constraints = [
            self._single_in_edge_constraint(),
            self._flow_conservation_constraint(),
            self._division_constraint(),
        ]

    def add_overlap_constraints(self, sources: ArrayLike, targets: ArrayLike) -> None:
        """Add constraints such that `source` and `target` can't be present in the same solution.

        Parameters
        ----------
        source : ArrayLike
            Source nodes indices.
        target : ArrayLike
            Target nodes indices.
        """
        sources = self._forward_map[np.asarray(sources)]
        targets = self._forward_map[np.asarray(targets)]

        ones = np.ones(len(targets), dtype=np.float32)

        nodes = self._sparse_nodes_ids_eye_matrix()
        overlaps = sparse.csr_matrix((ones, (sources, targets)), shape=self._shape)

        self._constraints += [(nodes - overlaps) @ self._variables >= 0]

    def enforce_node_to_solution(self, indices: ArrayLike) -> None:
        """Constraints given nodes' variables to 1.

        Parameters
        ----------
        indices : ArrayLike
            Nodes indices.
        """
        ones = np.ones(len(indices), dtype=np.float32)
        selected = sparse.csr_matrix((ones, (indices, indices)), shape=self._shape)
        self._constraints += [selected @ self._variables >= 1]

    def optimize(self) -> float:
        """Optimizes cvxpy model."""
        self._problem = cp.Problem(
            cp.Maximize(self._weights.T @ self._variables),
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

        nodes = self._backward_map[self._variables.value[: self._n_nodes] > 0.5]

        selected_edges = (
            self._variables.value[self._n_slack_vars * self._n_nodes :] > 0.5
        )
        sources = self._backward_map[self._out_edges[selected_edges]]
        targets = self._backward_map[self._in_edges[selected_edges]]
        edges = np.stack(
            (sources, targets),
            axis=1,
        )
        return indices_to_solution_dataframe(nodes, edges)
