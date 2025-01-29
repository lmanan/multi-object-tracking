from yaml import load, Loader
import json
import os
import sys
import numpy as np
import networkx as nx
from utils import (
    get_recursion_limit,
    save_ilp_result,
    add_gt_edges_to_graph,
    add_costs,
    add_constraints,
    add_app_disapp_attributes,
    convert_to_one_hot,
    convert_to_numpy_array,
    add_parent_id,
)

from motile_toolbox.candidate_graph import (
    get_candidate_graph_from_points_list,
    graph_to_nx,
    NodeAttr,
)
from motile import TrackGraph, Solver
from saving_utils import save_result
from run_traccuracy import compute_metrics
import pprint
import torch


pp = pprint.PrettyPrinter(indent=4)


def infer(yaml_configs_file_name: str):
    """infer.

    Parameters
    ----------
    yaml_configs_file_name : str
        yaml_configs_file_name is the path to the yaml config file.
    """

    with open(yaml_configs_file_name) as stream:
        args = load(stream, Loader=Loader)

    print("+" * 10)
    pp.pprint(args)

    test_pt_file_name = args["test_pt_file_name"]
    test_man_track_file_name = args["test_man_track_file_name"]
    num_nearest_neighbours = args["num_nearest_neighbours"]
    max_edge_distance = args["max_edge_distance"]
    direction_candidate_graph = args["direction_candidate_graph"]
    test_image_shape = args["test_image_shape"]
    pin_nodes = args["pin_nodes"]
    use_velocity = args["use_velocity"]
    use_cell_type = args["use_cell_type"]
    write_tifs = args["write_tifs"]
    ssvm_weights_array = args["ssvm_weights_array"]
    results_dir_name = args["results_dir_name"]
    whitening = args["whitening"]

    assert direction_candidate_graph in ["forward", "backward"]

    if os.path.exists(results_dir_name + "/jsons/"):
        pass
    else:
        os.makedirs(results_dir_name + "/jsons/")

    with open(results_dir_name + "/jsons/args.json", "w") as f:
        json.dump(args, f)

    # ++++++++
    # Step 1 - build test candidate graph
    # ++++++++

    test_list = torch.load(
        test_pt_file_name, weights_only=True, map_location=torch.device("cpu")
    )

    (
        test_array,
        test_old_id_new_id_dictionary,
        test_new_id_velocity_dictionary,
        test_new_id_cell_type_dictionary,
    ) = convert_to_numpy_array(test_list, test_image_shape, use_velocity, use_cell_type)

    with open(results_dir_name + "/jsons/old_to_new_mapping.json", "w") as f:
        json.dump(test_old_id_new_id_dictionary, f)

    test_array = add_parent_id(
        test_man_track_file_name, test_old_id_new_id_dictionary, test_array
    )

    np.savetxt(
        fname=results_dir_name + "/gt-detections.csv",
        X=test_array,
        fmt=["%i", "%i", "%.3f", "%.3f", "%i"],
    )

    print("+" * 10)
    print(f"Test array has shape {test_array.shape}.")
    test_t_min = int(np.min(test_array[:, 1]))
    test_t_max = int(np.max(test_array[:, 1]))

    test_candidate_graph_initial, mean_edge_distance, std_edge_distance = (
        get_candidate_graph_from_points_list(
            points_list=test_array,
            max_edge_distance=max_edge_distance,
            num_nearest_neighbours=num_nearest_neighbours,
            direction_candidate_graph=direction_candidate_graph,
            whitening=whitening,
        )
    )

    print("+" * 10)
    print(
        f"Mean edge distance is {mean_edge_distance} and std edge distance is {std_edge_distance}."
    )

    print("+" * 10)
    print(
        f"Number of nodes in test graph (before adding hyper edges) is {len(test_candidate_graph_initial.nodes)} and edges is {len(test_candidate_graph_initial.edges)}. "
    )

    mean_node_embedding_distance = std_node_embedding_distance = None

    if use_velocity:
        for node in test_candidate_graph_initial.nodes:
            test_candidate_graph_initial.nodes[node][NodeAttr.VELOCITY.value] = (
                test_new_id_velocity_dictionary[node]
            )

    if use_cell_type:
        number_cell_types = max(list(test_new_id_cell_type_dictionary.values())) + 1
        unique_cell_types = np.unique(list(test_new_id_cell_type_dictionary.values()))
        print("+" * 10)
        print(f"Number of cell types is {number_cell_types}.")
        print(f"Unique cell types are {unique_cell_types}.")
        for node in test_candidate_graph_initial.nodes:
            test_candidate_graph_initial.nodes[node][NodeAttr.NODE_EMBEDDING.value] = (
                convert_to_one_hot(
                    test_new_id_cell_type_dictionary[node], number_cell_types
                )
            )
        if whitening:
            node_embedding_distance_list = []
            for edge_id in test_candidate_graph_initial.edges:
                u, v = edge_id
                node_embedding_u = test_candidate_graph_initial.nodes[u][
                    NodeAttr.NODE_EMBEDDING.value
                ]
                node_embedding_v = test_candidate_graph_initial.nodes[v][
                    NodeAttr.NODE_EMBEDDING.value
                ]
                d = np.linalg.norm(node_embedding_u - node_embedding_v)
                node_embedding_distance_list.append(d)
            mean_node_embedding_distance = np.mean(node_embedding_distance_list)
            std_node_embedding_distance = np.std(node_embedding_distance_list)

            print("+" * 10)
            print(
                f"Mean node embedding distance is {mean_node_embedding_distance} and std node embedding distance is {std_node_embedding_distance}."
            )

    # test_candidate_graph = add_hyper_edges(candidate_graph=test_candidate_graph_initial)
    test_candidate_graph = test_candidate_graph_initial

    test_track_graph = TrackGraph(nx_graph=test_candidate_graph, frame_attribute="time")
    test_track_graph = add_app_disapp_attributes(
        test_track_graph, test_t_min, test_t_max
    )

    print("+" * 10)
    print(
        f"Number of nodes in test track graph (inc. hyper edges) is {len(test_track_graph.nodes)} and edges is {len(test_track_graph.edges)}."
    )

    recursion_limit = get_recursion_limit(candidate_graph=test_candidate_graph)
    if recursion_limit > 1000:
        sys.setrecursionlimit(recursion_limit)

    # ++++++++
    # Step 2 - apply weights on the test candidate graph
    # ++++++++

    solver = Solver(track_graph=test_track_graph)
    solver = add_costs(
        solver=solver,
        use_velocity=use_velocity,
        use_cell_type=use_cell_type,
        mean_edge_distance=mean_edge_distance,
        std_edge_distance=std_edge_distance,
        mean_node_embedding_distance=mean_node_embedding_distance,
        std_node_embedding_distance=std_node_embedding_distance,
    )
    solver = add_constraints(solver=solver, pin_nodes=pin_nodes)
    solver.weights.from_ndarray(ssvm_weights_array)
    solution = solver.solve(verbose=True)
    solution_graph = solver.get_selected_subgraph(solution)

    save_ilp_result(solution_graph, results_dir_name)

    print("+" * 10)
    print(
        f"After optimization, we selected {len(solution_graph.nodes)} nodes and {len(solution_graph.edges)} edges."
    )

    # ++++++++
    # Step 3 - traccuracy numbers
    # ++++++++

    new_mapping, res_track, tracked_masks, tracked_graph = save_result(
        solution_nx_graph=graph_to_nx(solution_graph),
        segmentation_shape=None,  # gt_segmentation.shape,
        output_tif_dir_name=results_dir_name,
        write_tifs=write_tifs,
    )

    tracked_array = []
    solution_nx_graph = graph_to_nx(solution_graph)
    for edge in solution_nx_graph.edges:
        u, v = edge
        t_u, id_u = u.split("_")
        t_v, id_v = v.split("_")
        t_u, id_u, t_v, id_v = int(t_u), int(id_u), int(t_v), int(id_v)

        # check if track starts at u
        in_edges = solution_nx_graph.in_edges(u)
        if len(in_edges) == 0:
            tracked_array.append(
                [
                    id_u,
                    t_u,
                    test_candidate_graph.nodes[u]["pos"][0],
                    test_candidate_graph.nodes[u]["pos"][1],
                    0,
                ]
            )

        tracked_array.append(
            [
                id_v,
                t_v,
                test_candidate_graph.nodes[v]["pos"][0],
                test_candidate_graph.nodes[v]["pos"][1],
                id_u,
            ]
        )

    tracked_array = np.asarray(tracked_array)
    np.savetxt(
        fname=results_dir_name + "/track-detections.csv",
        X=tracked_array,
        fmt=["%i", "%i", "%.3f", "%.3f", "%i"],
    )

    test_gt_graph = nx.DiGraph()
    test_gt_graph.add_nodes_from(test_candidate_graph_initial.nodes(data=True))
    test_gt_graph = add_gt_edges_to_graph(
        groundtruth_graph=test_gt_graph, gt_data=test_array
    )

    # convert to track graph
    test_gt_track_graph = TrackGraph(nx_graph=test_gt_graph, frame_attribute="time")
    print(
        f"Number of nodes in the groundtruth test dataset is {len(test_gt_track_graph.nodes)} and edges is {len(test_gt_track_graph.edges)}"
    )

    for node in test_gt_track_graph.nodes:
        pos = test_gt_track_graph.nodes[node]["pos"]
        if len(pos) == 2:
            y, x = pos
            test_gt_track_graph.nodes[node]["y"] = y
            test_gt_track_graph.nodes[node]["x"] = x
        elif len(pos) == 3:
            z, y, x = pos
            test_gt_track_graph.nodes[node]["z"] = int(z)
            test_gt_track_graph.nodes[node]["y"] = int(y)
            test_gt_track_graph.nodes[node]["x"] = int(x)

    compute_metrics(
        gt_segmentation=None,  # gt_segmentation,
        gt_nx_graph=graph_to_nx(test_gt_track_graph),
        predicted_segmentation=None,  # tracked_masks,
        pred_nx_graph=tracked_graph,
        results_dir_name=results_dir_name,
    )


if __name__ == "__main__":

    # parser = jsonargparse.ArgumentParser()
    # parser.add_argument("--yaml_configs_file_name", dest="yaml_configs_file_name")
    # args = parser.parse_args()
    # infer(yaml_configs_file_name=args.yaml_configs_file_name)

    infer(yaml_configs_file_name="../experiments/configs_infer.yaml")
