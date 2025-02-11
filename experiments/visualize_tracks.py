import jsonargparse
from motile_tracker.data_model.solution_tracks import SolutionTracks
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
from glob import glob
import tifffile
import napari
import networkx as nx
from tqdm import tqdm
import numpy as np
from yaml import load, Loader
from typing import Literal


def visualize_tracks(
    im_dir_name: str | None,
    gt_segmentation_dir_name: str | None,
    gt_detections_csv_file_name: str,
    ilp_csv_file_name: str,
    visualize_which: Literal["gt", "predicted"] = "predicted",
):

    if im_dir_name is not None:
        im_file_names = sorted(glob(im_dir_name + "/*.tif"))
        im = []
        for im_file_name in tqdm(im_file_names):
            im.append(tifffile.imread(im_file_name))
        im = np.asarray(im)

    if gt_segmentation_dir_name is not None:
        gt_segmentation_file_names = sorted(glob(gt_segmentation_dir_name + "/*.tif"))
        gt_segmentation = []
        for gt_segmentation_file_name in tqdm(gt_segmentation_file_names):
            gt_segmentation.append(tifffile.imread(gt_segmentation_file_name))
        gt_segmentation = np.asarray(gt_segmentation)

    gt_track_data = np.loadtxt(gt_detections_csv_file_name, delimiter=" ")
    gt_id_time_dictionary = {}

    gt_graph = nx.DiGraph()
    for row in tqdm(gt_track_data):
        id_, t, y, x, p_id = row
        id_, t, p_id = int(id_), int(t), int(p_id)
        attrs = {"pos": (y, x), "time": t, "seg_id": id_}
        gt_graph.add_node(str(t) + "_" + str(id_), **attrs)
        gt_id_time_dictionary[id_] = t

    for row in tqdm(gt_track_data):
        id_, t, y, x, p_id = row
        id_, t, p_id = int(id_), int(t), int(p_id)
        # add edges
        if p_id != 0:
            p_time = gt_id_time_dictionary[p_id]
            gt_graph.add_edge(str(p_time) + "_" + str(p_id), str(t) + "_" + str(id_))

    viewer = napari.Viewer()
    if im_dir_name is not None:
        viewer.add_image(im, name="image")
    if gt_segmentation_dir_name is not None:
        viewer.add_labels(gt_segmentation, name="labels")

    if visualize_which == "gt":
        if gt_segmentation_dir_name is not None:
            solution_tracks = SolutionTracks(
                graph=gt_graph,
                segmentation=gt_segmentation,
                time_attr="time",
                pos_attr="pos",
            )
        else:
            solution_tracks = SolutionTracks(
                graph=gt_graph,
                segmentation=None,
                time_attr="time",
                pos_attr="pos",
                ndim=2,
            )
        tracks_viewer = TracksViewer(viewer=viewer)
        tracks_viewer.update_tracks(tracks=solution_tracks, name="tracks")
        napari.run()
    elif visualize_which == "predicted":
        ilp_track_data = np.loadtxt(ilp_csv_file_name, delimiter=" ")

        ilp_graph = nx.DiGraph()
        for row in tqdm(ilp_track_data):
            id_t, t, id_tp1, tp1 = row.astype(int)
            pos = gt_graph.nodes[str(t) + "_" + str(id_t)]["pos"]
            attrs = {"pos": pos, "time": t, "seg_id": id_t}
            ilp_graph.add_node(str(t) + "_" + str(id_t), **attrs)

            pos = gt_graph.nodes[str(tp1) + "_" + str(id_tp1)]["pos"]
            attrs = {"pos": pos, "time": tp1, "seg_id": id_tp1}
            ilp_graph.add_node(str(tp1) + "_" + str(id_tp1), **attrs)
            ilp_graph.add_edge(str(t) + "_" + str(id_t), str(tp1) + "_" + str(id_tp1))

        if gt_segmentation_dir_name is not None:
            solution_tracks = SolutionTracks(
                graph=ilp_graph,
                segmentation=gt_segmentation,
                time_attr="time",
                pos_attr="pos",
            )
        else:
            solution_tracks = SolutionTracks(
                graph=ilp_graph,
                segmentation=None,
                time_attr="time",
                pos_attr="pos",
                ndim=2,
            )
        tracks_viewer = TracksViewer(viewer=viewer)
        tracks_viewer.update_tracks(tracks=solution_tracks, name="tracks")
        napari.run()


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "--yaml_configs_file_name",
        dest="yaml_configs_file_name",
        type=str,
        required=True,
    )
    args_ = parser.parse_args()
    with open(args_.yaml_configs_file_name) as stream:
        args = load(stream, Loader=Loader)
    print(args)

    im_dir_name = args["im_dir_name"]
    gt_segmentation_dir_name = args["gt_segmentation_dir_name"]
    gt_detections_csv_file_name = args["gt_detections_csv_file_name"]
    ilp_csv_file_name = args["ilp_csv_file_name"]
    visualize_which = args["visualize_which"]

    visualize_tracks(
        im_dir_name,
        gt_segmentation_dir_name,
        gt_detections_csv_file_name,
        ilp_csv_file_name,
        visualize_which,
    )

    # python visualize_tracks.py - -yaml_configs_file_name visualize_configs.yaml
