from dataclasses import dataclass

import numpy as np

from redkg.visualization.contracts.size_constructor_contract import TEdgeList


@dataclass
class DrawLineEdgesContract:
    vertex_coordinates: np.array
    vertex_size: list
    edge_list: TEdgeList
    show_arrow: bool
    edge_color: list
    edge_line_width: list