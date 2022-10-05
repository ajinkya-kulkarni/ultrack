import logging
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ultrack.core.database import NO_PARENT

LOG = logging.getLogger(__name__)


def indices_to_solution_dataframe(
    nodes: Sequence[int], edges: Union[Sequence[Tuple[int]], np.ndarray]
) -> pd.DataFrame:
    """
    Converts a list a nodes and a list of edges to a solution data frame.

    Parameters
    ----------
    nodes : Sequence[int]
        Indices of nodes on solution.
    edges : Union[Sequence[Tuple[int]], np.ndarray]
        List of edges on solution (source, target).

    Returns
    -------
    pd.DataFrame
        Dataframe indexed by nodes as indices and their parent (NA if orphan).
    """
    LOG.info(f"Solution nodes\n{nodes}")
    LOG.info(f"Solution edges\n{edges}")

    if len(nodes) == 0:
        raise ValueError("Something went wrong, nodes solution is empty.")

    nodes_df = pd.DataFrame(
        data=NO_PARENT,
        index=nodes,
        columns=["parent_id"],
    )

    if len(edges) == 0:
        raise ValueError("Something went wrong, edges solution is empty")

    edges = np.asarray(edges)

    edges = pd.DataFrame(
        data=edges[:, 0],
        index=edges[:, 1],
        columns=["parent_id"],
    )

    nodes_df.update(edges)

    return nodes_df
