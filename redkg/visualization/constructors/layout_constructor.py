"""LayoutConstructor module."""

from typing import Any

import numpy as np

from redkg.visualization.config.parameters.defaults import Defaults
from redkg.visualization.contracts.core_model_contract import CoreModelContract
from redkg.visualization.contracts.layout_contract import LayoutContract
from redkg.visualization.equations.calculate_init_position import calculate_init_position
from redkg.visualization.equations.core_physical_model import CorePhysicalModel
from redkg.visualization.equations.edge_list_to_incidence_matrix import edge_list_to_incidence_matrix
from redkg.visualization.exceptions.exceptions_classes import ParamsValidationException


class LayoutConstructor:
    """Constructor (one action controller) for Graph layout."""

    def __call__(self, contract: LayoutContract) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Class entrypoint."""
        vertex_coord = calculate_init_position(contract.vertex_num, scale=Defaults.layout_scale_initial)

        self._validate(vertex_coord)

        centers = [np.array([0, 0])]

        core_model_contract: CoreModelContract = CoreModelContract(
            nums=contract.vertex_num,
            forces={
                Defaults.node_attraction_key: contract.pull_edge_strength,
                Defaults.node_repulsion_key: contract.push_vertex_strength,
                Defaults.edge_repulsion_key: contract.push_edge_strength,
                Defaults.center_of_gravity_key: contract.pull_center_strength,
            },
            centers=centers,
        )
        model: CorePhysicalModel = CorePhysicalModel(core_model_contract)

        vertex_coord = model.build(vertex_coord, edge_list_to_incidence_matrix(contract.vertex_num, contract.edge_list))

        vertex_coord = (vertex_coord - vertex_coord.min(0)) / (
            vertex_coord.max(0) - vertex_coord.min(0)
        ) * Defaults.vertex_coord_multiplier + Defaults.vertex_coord_modifier

        return vertex_coord

    @staticmethod
    def _validate(vertex_coord: Any) -> None:
        is_valid = vertex_coord.max() <= Defaults.vertex_coord_max and vertex_coord.min() >= Defaults.vertex_coord_min

        if not is_valid:
            raise ParamsValidationException("Parameters are not valid")
