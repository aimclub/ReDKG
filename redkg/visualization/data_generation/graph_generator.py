"""GraphGenerator module."""

import random

from redkg.visualization.exceptions.exceptions_classes import ParamsValidationException


class GraphGenerator:
    """GraphGenerator base class."""

    def __init__(self, vertex_num: int, edge_num: int):
        """Graph generator constructor."""
        self.vertex_num = vertex_num
        self.edge_num = edge_num

        self._validate()

    def __call__(self) -> list:
        """Graph generator start point."""
        vertex_list = list(range(self.vertex_num))
        current_edge_num, edge_set = 0, set()

        while current_edge_num < self.edge_num:
            v = random.choice(vertex_list)
            w = random.choice(vertex_list)

            if v > w:
                v, w = w, v

            if v == w or (v, w) in edge_set:
                continue

            edge_set.add((v, w))

            current_edge_num += 1

        return list(edge_set)

    def _validate(self) -> None:
        vertex_num_gt_1 = self.vertex_num > 1
        numbers_are_proportional = self.edge_num < self.vertex_num * (self.vertex_num - 1) // 2

        if not vertex_num_gt_1 or not numbers_are_proportional:
            raise ParamsValidationException("Parameters are not valid")
