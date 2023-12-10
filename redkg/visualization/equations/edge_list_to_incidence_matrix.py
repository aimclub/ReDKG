"""Convert edge list to incidence matrix module."""

from itertools import chain
from typing import Any, Union

import numpy as np


def edge_list_to_incidence_matrix(vertex_num: int, edge_list: Union[list[tuple], list[tuple[Any, ...]]]) -> np.ndarray:
    """Convert edge list to incidence matrix function."""
    vertex_indexes = list(chain(*edge_list))
    edge_indexes_base = [[idx] * len(e) for idx, e in enumerate(edge_list)]
    edge_indexes = list(chain(*edge_indexes_base))
    H = np.zeros((vertex_num, len(edge_list)))
    H[vertex_indexes, edge_indexes] = 1
    return H
