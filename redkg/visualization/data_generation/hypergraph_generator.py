"""HypergraphGenerator module."""

import random
from typing import Any, Optional, Set

from redkg.visualization.config.parameters.generator_methods import GeneratorMethods
from redkg.visualization.equations.calculate_c_log_function import calculate_c_log_function
from redkg.visualization.exceptions.exceptions_classes import ParamsValidationException


class HypergraphGenerator:
    """HypergraphGenerator base class."""

    def __init__(
        self,
        vertex_num: int,
        edge_num: int,
        generation_method: str = GeneratorMethods.uniform,
        probability_k_list: Optional[list[float]] = None,
    ):
        """Hypergraph generator constructor."""
        self.vertex_num = vertex_num
        self.edge_num = edge_num
        self.method = generation_method
        self.probability_k_list: list[Any] = probability_k_list if probability_k_list is not None else []

        self._validate()

    def __generate_uniform(self, edge_degree_list: list) -> list:
        return [calculate_c_log_function(self.vertex_num, k) / (2**self.vertex_num - 1) for k in edge_degree_list]

    @staticmethod
    def __generate_low_order_first(edge_degree_list: list) -> list:
        probability_k_list = [3 ** (-k) for k in range(len(edge_degree_list))]
        sum_of_prob_k_list = sum(probability_k_list)
        probability_k_list = [probability_k / sum_of_prob_k_list for probability_k in probability_k_list]
        return probability_k_list

    @staticmethod
    def __generate_high_order_first(edge_degree_list: list) -> list:
        probability_k_list = [3 ** (-k) for k in range(len(edge_degree_list))]
        probability_k_list.reverse()
        sum_of_prob_k_list = sum(probability_k_list)
        probability_k_list = [probability_k / sum_of_prob_k_list for probability_k in probability_k_list]
        return probability_k_list

    @staticmethod
    def __generate_custom(probability_k_list: list) -> list:
        probability_k_list_sum = sum(probability_k_list)
        probability_k_list = [probability_k / probability_k_list_sum for probability_k in probability_k_list]
        return probability_k_list

    def __call__(self):
        """Hypergraph generator start point."""
        probability_k_list = self.probability_k_list
        degree_edge_list = list(range(2, self.vertex_num + 1))

        if self.method == GeneratorMethods.uniform:
            probability_k_list = self.__generate_uniform(degree_edge_list)

        elif self.method == GeneratorMethods.low_order_first:
            probability_k_list = self.__generate_low_order_first(degree_edge_list)

        elif self.method == GeneratorMethods.high_order_first:
            probability_k_list = self.__generate_high_order_first(degree_edge_list)

        elif self.method == GeneratorMethods.custom:
            self._validate_custom_generation_method()

            probability_k_list = self.__generate_custom(probability_k_list)

        edges: Set[tuple] = set()

        while len(edges) < self.edge_num:
            k = random.choices(degree_edge_list, weights=probability_k_list)[0]
            edge = random.sample(range(self.vertex_num), k)
            edge = tuple(sorted(edge))

            if edge not in edges:
                edges.add(edge)

        return list(edges)

    def _validate(self) -> None:
        vertex_num_gt_1 = self.vertex_num > 1
        edge_num_is_positive = self.edge_num > 0
        method_is_valid = self.method in GeneratorMethods().values

        if not vertex_num_gt_1 or not edge_num_is_positive or not method_is_valid:
            raise ParamsValidationException("Parameters are not valid")

    def _validate_custom_generation_method(self) -> None:
        prob_k_list_is_not_none = self.probability_k_list is not None
        prob_k_list_len_is_valid = len(self.probability_k_list) == self.vertex_num - 1

        if not prob_k_list_is_not_none or not prob_k_list_len_is_valid:
            raise ParamsValidationException("Parameters are not valid")
