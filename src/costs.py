from motile.costs import Costs, Weight
from motile.variables import EdgeSelected
import networkx as nx
import numpy as np
from typing import cast


class EdgeDistance(Costs):
    def __init__(
        self,
        position_attribute,
        weight,
        constant,
        mean_edge_distance=None,
        std_edge_distance=None,
        velocity_attribute="velocity",
        use_velocity=False,
    ):
        self.position_attribute = position_attribute
        self.velocity_attribute = velocity_attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.mean_edge_distance = mean_edge_distance
        self.std_edge_distance = std_edge_distance
        self.use_velocity = use_velocity

    def apply(self, solver):
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if type(key[1]) is tuple:  # must be a hyper edge ...
                (start,) = key[0]
                end1, end2 = key[1]
                pos_start = self.__get_node_position(solver.graph, start)
                pos_end1 = self.__get_node_position(solver.graph, end1)
                pos_end2 = self.__get_node_position(solver.graph, end2)
                if self.use_velocity:
                    pos_start += self.__get_node_velocity(solver.graph, start)
                    pos_end1 += self.__get_node_velocity(solver.graph, end1)
                    pos_end2 += self.__get_node_velocity(solver.graph, end2)
                feature = np.linalg.norm(pos_start - 0.5 * (pos_end1 + pos_end2))
                if (
                    self.mean_edge_distance is not None
                    and self.std_edge_distance is not None
                ):
                    feature = (feature - self.mean_edge_distance) / (
                        self.std_edge_distance
                    )
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)
            else:  # normal edge
                u, v = cast("tuple[int, int]", key)
                pos_start = self.__get_node_position(solver.graph, u)
                pos_end = self.__get_node_position(solver.graph, v)
                if self.use_velocity:
                    pos_start += self.__get_node_velocity(solver.graph, u)
                    pos_end += self.__get_node_velocity(solver.graph, v)
                feature = np.linalg.norm(pos_start - pos_end)
                if (
                    self.mean_edge_distance is not None
                    and self.std_edge_distance is not None
                ):
                    feature = (feature - self.mean_edge_distance) / (
                        self.std_edge_distance
                    )
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)

    def __get_node_position(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.position_attribute, tuple):
            return np.array([graph.nodes[node][p] for p in self.position_attribute])
        else:
            return np.array(graph.nodes[node][self.position_attribute])

    def __get_node_velocity(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.velocity_attribute, tuple):
            return np.array([graph.nodes[node][p] for p in self.velocity_attribute])
        else:
            return np.array(graph.nodes[node][self.velocity_attribute])


class NodeEmbeddingDistance(Costs):
    def __init__(
        self,
        node_embedding_attribute,
        weight,
        constant,
        mean_node_embedding_distance=None,
        std_node_embedding_distance=None,
    ):
        self.node_embedding_attribute = node_embedding_attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.mean_node_embedding_distance = mean_node_embedding_distance
        self.std_node_embedding_distance = std_node_embedding_distance

    def apply(self, solver):
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if type(key[1]) is tuple:  # must be a hyper edge ...
                (start,) = key[0]
                end1, end2 = key[1]
                embedding_start = self.__get_node_embedding(solver.graph, start)
                embedding_end1 = self.__get_node_embedding(solver.graph, end1)
                embedding_end2 = self.__get_node_embedding(solver.graph, end2)
                feature = np.linalg.norm(
                    embedding_start - 0.5 * (embedding_end1 + embedding_end2)
                )
                if (
                    self.mean_node_embedding_distance is not None
                    and self.std_node_embedding_distance is not None
                ):
                    feature = (feature - self.mean_node_embedding_distance) / (
                        self.std_node_embedding_distance
                    )
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)
            else:  # normal edge
                u, v = cast("tuple[int, int]", key)
                embedding_u = self.__get_node_embedding(solver.graph, u)
                embedding_v = self.__get_node_embedding(solver.graph, v)
                feature = np.linalg.norm(embedding_u - embedding_v)
                if (
                    self.mean_node_embedding_distance is not None
                    and self.std_node_embedding_distance is not None
                ):
                    feature = (feature - self.mean_node_embedding_distance) / (
                        self.std_node_embedding_distance
                    )
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)

    def __get_node_embedding(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.node_embedding_attribute, tuple):
            return np.array(
                [graph.nodes[node][p] for p in self.node_embedding_attribute]
            )
        else:
            return np.array(graph.nodes[node][self.node_embedding_attribute])
