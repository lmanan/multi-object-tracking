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
import cv2


def visualize_tracks(
    im_dir_name: str | None,
    gt_segmentation_dir_name: str | None,
    gt_detections_csv_file_name: str,
    ilp_csv_file_name: str,
    visualize_which: Literal["gt", "predicted"] = "predicted",
    scaling_x: float = 0.5,
    scaling_y: float = 1.0,
):
    """visualize_tracks.

    This function enables visualizing tracks generated from running the
    `../src/infer.py` script.
    It can also visualize the ground truth tracks.
    Note, this function must be run with separate 'motile-tracker' environment 
    which contains the 'motile-tracker' plugin.

    Parameters
    ----------
    im_dir_name : str | None
        im_dir_name is the directory containing all the '*.jpg' images.
        This is optional.
        Please make sure that the image names are zero-padded.
    gt_segmentation_dir_name : str | None
        gt_segmentation_dir_name is the directory containing tif instance masks for
        the images above. This is optional.
        The name of the files should be zero-padded.
    gt_detections_csv_file_name : str
        gt_detections_csv_file_name is the csv file name containing the ground
        truth detections.
        The columns should be in the order of id_, t, y, x and p_id.
    ilp_csv_file_name : str
        ilp_csv_file_name is the name of the csv file containing the ILP
        solution.
        This file has four columns. id at time t, t, id at time t+1 and t+1.
    visualize_which : Literal["gt", "predicted"]
        visualize_which is a string that can be either "gt" or "predicted".
        Setting it to "gt" will visualize the ground truth tracks.
        Setting it to "predicted" will visualize the predicted tracks.
    scaling_x : float
        scaling_x is the scaling factor to be applied to the x coordinate of
        the point data. (In case the data is normalized to be between 0 and
        1).
        x_coordinate_correct =  x_coordinate * width of image * scaling_x
    scaling_y : int
        scaling_y is the scaling factor to be applied to the y coordinate of
        the point data. (In case the data is normalized to be between 0 and 1).
        y_coordinate_correct =  y_coordinate * height of image * scaling_y
    """

    if im_dir_name is not None:
        im_file_names = sorted(glob(im_dir_name + "/*.jpg"))

        im = []
        for im_file_name in tqdm(im_file_names):
            im.append(cv2.imread(im_file_name))
        im = np.asarray(im)
        print(f"Images have shape {im.shape}.")
        T, H, W, C = im.shape

    if gt_segmentation_dir_name is not None:
        gt_segmentation_file_names = sorted(glob(gt_segmentation_dir_name + "/*.tif"))
        gt_segmentation = []
        for gt_segmentation_file_name in tqdm(gt_segmentation_file_names):
            gt_segmentation.append(tifffile.imread(gt_segmentation_file_name))
        gt_segmentation = np.asarray(gt_segmentation)

    gt_track_data = np.loadtxt(gt_detections_csv_file_name, delimiter=" ")
    gt_track_data[:, 2] *= H
    gt_track_data[:, 3] *= W
    gt_track_data[:, 2] *= scaling_y
    gt_track_data[:, 3] *= scaling_x

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
                ndim=3,
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
                ndim=3,
            )
        tracks_viewer = TracksViewer(viewer=viewer)
        tracks_viewer.update_tracks(tracks=solution_tracks, name="tracks")
        for layer in tracks_viewer.viewer.layers:
            print(layer.name)

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
    scaling_x = args["scaling_x"]
    scaling_y = args["scaling_y"]

    visualize_tracks(
        im_dir_name=im_dir_name,
        gt_segmentation_dir_name=gt_segmentation_dir_name,
        gt_detections_csv_file_name=gt_detections_csv_file_name,
        ilp_csv_file_name=ilp_csv_file_name,
        visualize_which=visualize_which,
        scaling_x=scaling_x,
        scaling_y=scaling_y,
    )

    # python visualize_tracks.py - -yaml_configs_file_name visualize_configs.yaml
