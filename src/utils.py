from typing import Mapping, List
import numpy as np
import networkx as nx
import motile
from itertools import combinations
from motile_toolbox.candidate_graph.graph_attributes import NodeAttr
from motile.constraints.pin import Pin
from motile.constraints.max_children import MaxChildren
from motile.constraints.max_parents import MaxParents
from motile.costs.appear import Appear
from motile.costs.disappear import Disappear
from costs import EdgeDistance, NodeEmbeddingDistance


def get_recursion_limit(candidate_graph: nx.DiGraph) -> int:
    """get_recursion_limit.
    This function looks at the maximum outgoing edges present in the candidate
    graph and adjusts the recursion limit in motile.

    Parameters
    ----------
    candidate_graph : nx.DiGraph
        candidate_graph

    Returns
    -------
    int

    """
    max_in_edges = max_out_edges = 0
    for node in candidate_graph.nodes:
        num_next = len(candidate_graph.out_edges(node))
        if num_next > max_out_edges:
            max_out_edges = num_next

        num_prev = len(candidate_graph.in_edges(node))
        if num_prev > max_in_edges:
            max_in_edges = num_prev

    print("+" * 10)
    print(f"Maximum out edges is {max_out_edges}, max in edges {max_in_edges}.")
    temp_limit = np.maximum(max_in_edges, max_out_edges) + 500
    return temp_limit


def add_hyper_edges(candidate_graph: nx.DiGraph) -> nx.DiGraph:
    """add_hyper_edges.
    This function adds hyper edges in addition to regular edges.
    These hyper edges take the form ((u,), (v1, v2))
    where u is the node id of node at t and v1 and v2 are node ids of nodes at
    t+1.

    Parameters
    ----------
    candidate_graph : nx.DiGraph
        candidate_graph

    Returns
    -------
    nx.DiGraph

    """
    candidate_graph_copy = nx.DiGraph()
    candidate_graph_copy.add_nodes_from(candidate_graph.nodes(data=True))
    candidate_graph_copy.add_edges_from(candidate_graph.edges(data=True))
    nodes_original = list(candidate_graph_copy.nodes)
    for node in nodes_original:
        out_edges = candidate_graph_copy.out_edges(node)
        pairs = list(combinations(out_edges, 2))
        for pair in pairs:
            temporary_node = (
                str(pair[0][0]) + "_" + str(pair[0][1]) + "_" + str(pair[1][1])
            )
            candidate_graph_copy.add_node(temporary_node)
            candidate_graph_copy.add_edge(pair[0][0], temporary_node)
            candidate_graph_copy.add_edge(
                temporary_node,
                pair[0][1],
            )
            candidate_graph_copy.add_edge(
                temporary_node,
                pair[1][1],
            )
    return candidate_graph_copy


def save_ilp_result(solution_graph: nx.DiGraph, results_dir_name: str):
    """save_ilp_result.
    This function saves the ILP result as a text file.
    This text file has four columns:
    id at t, t, id at t+1, t+1

    Parameters
    ----------
    solution_graph : nx.DiGraph
        solution_graph is the ubset of the candidate graph which was selected
        after the optimmization.
    results_dir_name : str
        results_dir_name is the name of the directory where the ilp result is
        saved.
    """
    ilp_results_data = []
    for edge in solution_graph.edges:
        u, v = edge
        if isinstance(u, tuple):
            (u,) = u

        t_u, id_u = u.split("_")
        t_u, id_u = int(t_u), int(id_u)
        if isinstance(v, tuple):
            m, n = v

            t_m, id_m = m.split("_")
            t_m, id_m = int(t_m), int(id_m)

            t_n, id_n = n.split("_")
            t_n, id_n = int(t_n), int(id_n)

            ilp_results_data.append([id_u, t_u, id_m, t_m])
            ilp_results_data.append([id_u, t_u, id_n, t_n])
        else:
            t_v, id_v = v.split("_")
            t_v, id_v = int(t_v), int(id_v)

            ilp_results_data.append([id_u, t_u, id_v, t_v])

    np.savetxt(
        results_dir_name + "/jsons/ilp.csv",
        np.asarray(ilp_results_data),
        delimiter=" ",
        fmt=["%i", "%i", "%i", "%i"],
    )


def add_gt_edges_to_graph(
    groundtruth_graph: nx.DiGraph, gt_data: np.ndarray
) -> nx.DiGraph:
    """add_gt_edges_to_graph.
    This function creates a ground truth candidate graph.

    Parameters
    ----------
    groundtruth_graph : nx.DiGraph
        groundtruth_graph is the graph with only the edges which are
        known to be correct.
    gt_data : np.ndarray
        gt_data is the data read from man_track file.

    Returns
    -------
    nx.DiGraph

    """
    """gt data will have last column as parent id column."""
    parent_daughter_dictionary = {}  # parent_id : list of daughter ids
    id_time_dictionary = {}  # id_: time it shows up
    for row in gt_data:
        id_, t, parent_id = int(row[0]), int(row[1]), int(row[-1])
        if parent_id <= 0:  # new track starts
            pass
        else:
            if parent_id in parent_daughter_dictionary:
                parent_daughter_dictionary[parent_id].append(id_)
            else:
                parent_daughter_dictionary[parent_id] = [id_]
    for row in gt_data:
        id_, t, parent_id = int(row[0]), int(row[1]), int(row[-1])
        id_time_dictionary[id_] = t

    for row in gt_data:
        id_, t, parent_id = int(row[0]), int(row[1]), int(row[-1])
        if parent_id > 0:
            parent_time = id_time_dictionary[parent_id]
            start_node = str(parent_time) + "_" + str(parent_id)
            if len(parent_daughter_dictionary[parent_id]) == 1:
                end_node = str(t) + "_" + str(id_)
                groundtruth_graph.add_edge(start_node, end_node)
            elif len(parent_daughter_dictionary[parent_id]) == 2:
                temporary_node = start_node
                for daughter_node in parent_daughter_dictionary[parent_id]:
                    temporary_node += "_" + str(t) + "_" + str(daughter_node)
                groundtruth_graph.add_node(temporary_node)
                groundtruth_graph.add_edge(start_node, temporary_node)
                for daughter_node in parent_daughter_dictionary[parent_id]:
                    end_node = str(t) + "_" + str(daughter_node)
                    groundtruth_graph.add_edge(temporary_node, end_node)
    return groundtruth_graph


def add_costs(
    solver: motile.Solver,
    use_velocity: bool,
    use_cell_type: bool,
    mean_edge_distance: float = None,
    std_edge_distance: float = None,
    mean_node_embedding_distance: float = None,
    std_node_embedding_distance: float = None,
):
    """add_costs.
    This function adds the various costs, needed prior to solving.

    Parameters
    ----------
    solver : motile.Solver
        solver
    use_velocity : bool
        use_velocity
    use_cell_type : bool
        use_cell_type
    mean_edge_distance : float
        mean_edge_distance
    std_edge_distance : float
        std_edge_distance
    mean_node_embedding_distance : float
        mean_node_embedding_distance
    std_node_embedding_distance : float
        std_node_embedding_distance
    """
    solver.add_costs(
        EdgeDistance(
            weight=1.0,
            constant=0.0,
            position_attribute=NodeAttr.POS.value,
            velocity_attribute=NodeAttr.VELOCITY.value,
            mean_edge_distance=mean_edge_distance,
            std_edge_distance=std_edge_distance,
        ),
        name="Edge Distance",
    )
    if use_cell_type:
        solver.add_costs(
            NodeEmbeddingDistance(
                node_embedding_attribute=NodeAttr.NODE_EMBEDDING.value,
                weight=1.0,
                constant=0.0,
                mean_node_embedding_distance=mean_node_embedding_distance,
                std_node_embedding_distance=std_node_embedding_distance,
            ),
            name="Cell Type Distance",
        )
    solver.add_costs(Appear(constant=1.0, ignore_attribute="ignore_appear_cost"))
    solver.add_costs(Disappear(constant=1.0, ignore_attribute="ignore_disappear_cost"))
    return solver


def add_constraints(solver: motile.Solver, pin_nodes: bool) -> motile.Solver:
    """add_constraints.
    This function adds the various constraints needed prior to solving.

    Parameters
    ----------
    solver : motile.Solver
        solver
    pin_nodes : bool
        pin_nodes

    Returns
    -------
    motile.Solver

    """

    solver.add_constraints(MaxParents(1))
    solver.add_constraints(MaxChildren(1))
    if pin_nodes:
        solver.add_constraints(Pin(attribute=NodeAttr.PINNED.value))
    return solver


def convert_to_one_hot(cell_type: int, number_cell_types: int) -> np.ndarray:
    """convert_to_one_hot.
    This function takes in a discrete, non negative value (`cell_type`)and converts it to a
    one hot representation.

    Parameters
    ----------
    cell_type : int
        cell_type is the actual discrete label.
    number_cell_types : int
        number_cell_types is the number of unique cell types.


    Returns
    -------
    np.ndarray

    """
    data = np.zeros((number_cell_types), dtype=float)
    data[int(cell_type)] = 1.0
    return data


def convert_to_numpy_array(
    pts: List,
    test_image_shape: tuple[int, int],
    use_velocity: bool = False,
    use_cell_type: bool = False,
) -> tuple[np.ndarray, Mapping]:
    """convert_to_numpy_array.
    This function is designed to take in *.pts dataset and convert it to a
    numpy array.
    The numpy array has unique_id, time, [z], y, x as the [five] four columns.

    Parameters
    ----------
    pts : List
        pts
    test_image_shape : tuple[int, int]
        Since the points are presented in a normalized fashion, we get their
        corresponding real world coordinates, by multiplying the point
        coordinates with the image shape on which they lie.
    use_velocity : bool
        use_velocity is True, if velocity should be included in the
        optimization.
    use_cell_type : bool
        use_cell_type is True, if cell type should be included in the
        optimization.

    Returns
    -------
    tuple[np.ndarray, Mapping, Mapping, Mapping]

    """

    old_id_new_id_dictionary = {}
    new_id_velocity_dictionary = {}
    new_id_cell_type_dictionary = {}
    counter = 1
    all_data = []
    H, W = test_image_shape
    for time in range(len(pts)):
        for row in pts[time]:
            old_id_new_id_dictionary[str(time) + "_" + str(int(row[0]))] = counter
            all_data.append([counter, time, H * row[2], W * row[1]])
            if use_velocity:
                new_id_velocity_dictionary[str(time) + "_" + str(counter)] = np.array(
                    [H * row[4], W * row[3]]
                )  # ydot, xdot
            if use_cell_type:
                new_id_cell_type_dictionary[str(time) + "_" + str(counter)] = int(
                    row[5]
                )
            counter += 1

    return (
        np.asarray(all_data),
        old_id_new_id_dictionary,
        new_id_velocity_dictionary,
        new_id_cell_type_dictionary,
    )


def add_parent_id(
    man_track_file_name: str, dictionary: Mapping, array: np.ndarray, id_start: int = 1
) -> np.ndarray:
    """add_parent_id.
    This function takes in an existing dictionary and includes the parent id information.
    This (parent id) is only used for evaluation.

    Parameters
    ----------
    man_track_file_name : str
        man_track_file_name is the path to the file containing four columns.
        id, t_start, t_end, parent_id.
        If the track does not have a parent, then parent_id can be marked as 0.
    dictionary : Mapping
        dictionary created earlier, which has a mapping from old ids to new
        unique ids per time frame.
    array : np.ndarray
        array is the existing array which has unique_id, t, [z], y, x as the
        [five] four columns.
    id_start :
        `id_start` is set to 1 if all ids mentioned in the man_track file start from 1.


    Returns
    -------
    np.ndarray

    """
    man_track_data = np.loadtxt(man_track_file_name, delimiter=" ")
    print("+" * 10)
    print(f"man track data has shape {man_track_data.shape}.")
    updated_dictionary = {}
    for row in man_track_data:
        id_, time_start, time_end, parent_id = row.astype(int)
        if id_start == 1:
            id_ = id_ - 1
            parent_id = np.maximum(0, parent_id - 1)
        for time in range(time_start, time_end + 1):
            t_id = str(time) + "_" + str(id_)
            new_id = dictionary[t_id]
            if time == time_start:
                if parent_id == 0:
                    updated_dictionary[new_id] = 0
                else:
                    parent_key = str(time - 1) + "_" + str(parent_id)
                    updated_dictionary[new_id] = dictionary[parent_key]
            else:
                parent_key = str(time - 1) + "_" + str(id_)
                updated_dictionary[new_id] = dictionary[parent_key]

    array_new = np.zeros((array.shape[0], array.shape[1] + 1))
    for i in range(array.shape[0]):
        array_new[i, :4] = array[i, :4]
        array_new[i, 4] = updated_dictionary[array[i, 0]]
    return array_new


def expand_position(
    data: np.ndarray,
    position: List,
    id_: int,
    nhood: int = 1,
) -> np.ndarray:
    """expand_position.
    This function makes detections from the point data.
    This is needed for traccuracy (the evaluation pipeline) to work since it
    can not compute IoU only from point data.

    Parameters
    ----------
    data : np.ndarray
        data
    position : List
        position
    id_ : int
        id_
    nhood : int
        nhood

    Returns
    -------
    np.ndarray

    """

    outside = True
    if len(position) == 2:
        H, W = data.shape
        y, x = position
        y, x = int(y), int(x)
        while outside:
            data_ = data[
                np.maximum(y - nhood, 0) : np.minimum(y + nhood + 1, H),
                np.maximum(x - nhood, 0) : np.minimum(x + nhood + 1, W),
            ]
            if 0 in data_.shape:
                nhood += 1
            else:
                outside = False
        data[
            np.maximum(y - nhood, 0) : np.minimum(y + nhood + 1, H),
            np.maximum(x - nhood, 0) : np.minimum(x + nhood + 1, W),
        ] = id_
    elif len(position) == 3:
        D, H, W = data.shape
        z, y, x = position
        z, y, x = int(z), int(y), int(x)
        while outside:
            data_ = data[
                np.maximum(z - nhood, 0) : np.minimum(z + nhood + 1, D),
                np.maximum(y - nhood, 0) : np.minimum(y + nhood + 1, H),
                np.maximum(x - nhood, 0) : np.minimum(x + nhood + 1, W),
            ]
            if 0 in data_.shape:
                nhood += 1
            else:
                outside = False
        data[
            np.maximum(z - nhood, 0) : np.minimum(z + nhood + 1, D),
            np.maximum(y - nhood, 0) : np.minimum(y + nhood + 1, H),
            np.maximum(x - nhood, 0) : np.minimum(x + nhood + 1, W),
        ] = id_
    return data


def add_app_disapp_attributes(
    track_graph: motile.TrackGraph, t_min: int, t_max: int
) -> motile.TrackGraph:
    """add_app_disapp_attributes.
    This function adds an ignore appear or ignore disappear attribute to nodes
    if they show up at the start or the end of the acquisition.

    Parameters
    ----------
    track_graph : motile.TrackGraph
        track_graph
    t_min : int
        t_min
    t_max : int
        t_max

    Returns
    -------
    motile.TrackGraph

    """

    num_nodes_previous = {}
    num_nodes_next = {}
    num_nodes_current = {}
    for t in range(t_min, t_max + 1):
        if t == t_min:
            num_nodes_previous[t_min] = 0
        else:
            num_nodes_previous[t] = len(track_graph.nodes_by_frame(t - 1))

        if t == t_max:
            num_nodes_next[t_max] = 0
        else:
            num_nodes_next[t] = len(track_graph.nodes_by_frame(t + 1))
        num_nodes_current[t] = len(track_graph.nodes_by_frame(t))

    for node, attrs in track_graph.nodes.items():
        time = attrs[NodeAttr.TIME.value]
        if num_nodes_previous[time] == 0 and num_nodes_current[time] != 0:
            track_graph.nodes[node][NodeAttr.IGNORE_APPEAR_COST.value] = True
        if num_nodes_next[time] == 0 and num_nodes_current[time] != 0:
            track_graph.nodes[node][NodeAttr.IGNORE_DISAPPEAR_COST.value] = True
    return track_graph
