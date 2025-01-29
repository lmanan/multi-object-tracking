from tqdm import tqdm
from pathlib import Path
import tifffile
import numpy as np
import networkx as nx
import json
from motile_toolbox.candidate_graph import NodeAttr
from utils import expand_position


def save_result(
    solution_nx_graph: nx.DiGraph,
    segmentation_shape,
    output_tif_dir_name: str,
    write_tifs: bool = False,
):
    if segmentation_shape is not None:
        tracked_masks = np.zeros(segmentation_shape, dtype=np.uint16)
    else:
        tracked_masks = None
    new_mapping = {}  # <t_id> in segmentation mask: id in tracking mask
    res_track = (
        {}
    )  # id in tracking mask: ([t0, t1, ..., tN], parent_id) in tracking mask
    id_counter = 1
    for in_node, out_node in tqdm(solution_nx_graph.edges()):
        t_in, id_in = in_node.split("_")
        t_out, id_out = out_node.split("_")
        t_in, id_in = int(t_in), int(id_in)
        t_out, id_out = int(t_out), int(id_out)
        num_out_edges = len(solution_nx_graph.out_edges(in_node))
        if num_out_edges == 1:
            if in_node in new_mapping.keys():
                # i.e. continuation of an existing edge
                if t_out not in res_track[new_mapping[in_node]][0]:
                    res_track[new_mapping[in_node]][0].append(
                        t_out
                    )  # include the end time for this tracklet
                position_in_node = solution_nx_graph.nodes[in_node][NodeAttr.POS.value]
                if tracked_masks is not None:
                    tracked_masks[t_in] = expand_position(
                        data=tracked_masks[t_in],
                        position=position_in_node,
                        id_=new_mapping[in_node],
                    )
                new_mapping[out_node] = new_mapping[in_node]

                position_out_node = solution_nx_graph.nodes[out_node][
                    NodeAttr.POS.value
                ]
                if tracked_masks is not None:
                    tracked_masks[t_out] = expand_position(
                        data=tracked_masks[t_out],
                        position=position_out_node,
                        id_=new_mapping[out_node],
                    )

            else:
                # i.e. start of a new edge
                res_track[id_counter] = ([t_in, t_out], 0)
                new_mapping[in_node] = id_counter
                new_mapping[out_node] = id_counter

                position_in_node = solution_nx_graph.nodes[in_node][NodeAttr.POS.value]
                position_out_node = solution_nx_graph.nodes[out_node][
                    NodeAttr.POS.value
                ]
                if tracked_masks is not None:
                    tracked_masks[t_in] = expand_position(
                        data=tracked_masks[t_in],
                        position=position_in_node,
                        id_=id_counter,
                    )
                    tracked_masks[t_out] = expand_position(
                        data=tracked_masks[t_out],
                        position=position_out_node,
                        id_=id_counter,
                    )
                id_counter += 1
        elif num_out_edges == 2:
            out_edge1, out_edge2 = solution_nx_graph.out_edges(in_node)
            _, out_node1 = out_edge1
            _, out_node2 = out_edge2
            t_out1, id_out1 = out_node1.split("_")
            t_out1, id_out1 = int(t_out1), int(id_out1)
            t_out2, id_out2 = out_node2.split("_")
            t_out2, id_out2 = int(t_out2), int(id_out2)
            if in_node in new_mapping.keys():
                # i.e. in node was connected by one outgoing edge previously
                if t_in not in res_track[new_mapping[in_node]][0]:
                    res_track[new_mapping[in_node]][0].append(t_in)

                position_in_node = solution_nx_graph.nodes[in_node][NodeAttr.POS.value]

                if tracked_masks is not None:
                    tracked_masks[t_in] = expand_position(
                        data=tracked_masks[t_in],
                        position=position_in_node,
                        id_=new_mapping[in_node],
                    )

                if out_node1 not in new_mapping:
                    new_mapping[out_node1] = id_counter
                    position_out_node1 = solution_nx_graph.nodes[out_node1][
                        NodeAttr.POS.value
                    ]

                    if tracked_masks is not None:
                        tracked_masks[t_out1] = expand_position(
                            data=tracked_masks[t_out1],
                            position=position_out_node1,
                            id_=id_counter,
                        )

                    res_track[id_counter] = ([t_out1], new_mapping[in_node])
                    id_counter += 1
                if out_node2 not in new_mapping:
                    new_mapping[out_node2] = id_counter
                    position_out_node2 = solution_nx_graph.nodes[out_node2][
                        NodeAttr.POS.value
                    ]
                    if tracked_masks is not None:
                        tracked_masks[t_out2] = expand_position(
                            data=tracked_masks[t_out2],
                            position=position_out_node2,
                            id_=id_counter,
                        )
                    res_track[id_counter] = ([t_out2], new_mapping[in_node])
                    id_counter += 1
            else:
                res_track[id_counter] = ([t_in], 0)
                new_mapping[in_node] = id_counter

                position_in_node = solution_nx_graph.nodes[in_node][NodeAttr.POS.value]
                if tracked_masks is not None:
                    tracked_masks[t_in] = expand_position(
                        data=tracked_masks[t_in],
                        position=position_in_node,
                        id_=id_counter,
                    )

                id_counter += 1
                if out_node1 not in new_mapping:
                    new_mapping[out_node1] = id_counter
                    position_out_node1 = solution_nx_graph.nodes[out_node1][
                        NodeAttr.POS.value
                    ]
                    if tracked_masks is not None:
                        tracked_masks[t_out1] = expand_position(
                            data=tracked_masks[t_out1],
                            position=position_out_node1,
                            id_=id_counter,
                        )

                    res_track[id_counter] = ([t_out1], new_mapping[in_node])
                    id_counter += 1
                if out_node2 not in new_mapping:
                    new_mapping[out_node2] = id_counter
                    position_out_node2 = solution_nx_graph.nodes[out_node2][
                        NodeAttr.POS.value
                    ]
                    if tracked_masks is not None:
                        tracked_masks[t_out2] = expand_position(
                            data=tracked_masks[t_out2],
                            position=position_out_node2,
                            id_=id_counter,
                        )

                    res_track[id_counter] = ([t_out2], new_mapping[in_node])
                    id_counter += 1

    # in case there are edges that start and end at the same node, we do a
    # little more
    for node in solution_nx_graph.nodes():
        t, id_node = node.split("_")
        t, id_node = int(t), int(id_node)
        if node in new_mapping.keys():
            pass
        else:
            res_track[id_counter] = ([t], 0)
            new_mapping[node] = id_counter
            position_node = solution_nx_graph.nodes[node][NodeAttr.POS.value]
            if tracked_masks is not None:
                tracked_masks[t] = expand_position(
                    data=tracked_masks[t], position=position_node, id_=id_counter
                )

            id_counter += 1

    # ensure that path where tifs will be saved, exists.
    if Path(output_tif_dir_name).exists():
        filenames = list(Path(output_tif_dir_name).glob("*.tif"))
        for filename in filenames:
            Path(filename).unlink()
    else:
        Path(output_tif_dir_name).mkdir()

    # write tifs
    if write_tifs:
        if tracked_masks is not None:
            for i in range(tracked_masks.shape[0]):
                tifffile.imwrite(
                    Path(output_tif_dir_name) / ("mask" + str(i).zfill(3) + ".tif"),
                    tracked_masks[i].astype(np.uint16),
                )
    # write man_track.json
    with open(output_tif_dir_name + "/jsons/res_track.json", "w") as f:
        json.dump(res_track, f)

    print(f"Final id counter is {id_counter}")

    # write the <t_id --> new_tracklet) in a file
    with open(output_tif_dir_name + "/jsons/tracklet.json", "w") as f:
        json.dump(new_mapping, f)

    tracked_graph = nx.DiGraph()
    for k, v in new_mapping.items():
        t, id_ = k.split("_")
        t, id_ = int(t), int(id_)
        pos = solution_nx_graph.nodes[k]["pos"]
        if len(pos) == 2:
            y, x = pos
            tracked_graph.add_node(k, seg_id=v, time=t, y=y, x=x)
        elif len(pos) == 3:
            z, y, x = pos
            tracked_graph.add_node(k, seg_id=v, time=t, z=z, y=y, x=x)
    for edge in solution_nx_graph.edges:
        u, v = edge
        tracked_graph.add_edge(u, v)

    return new_mapping, res_track, tracked_masks, tracked_graph
