from traccuracy import run_metrics
from traccuracy.loaders import load_graph_masks, load_deepcell_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics
import pprint
import argparse
import json
import numpy as np
import networkx as nx

pp = pprint.PrettyPrinter(indent=4)


def compute_metrics(
    gt_segmentation: np.ndarray,
    predicted_segmentation: np.ndarray,
    results_dir_name: str,
    gt_nx_graph: nx.DiGraph | None = None,
    pred_nx_graph: nx.DiGraph | None = None,
    gt_json_file_name: str | None = None,
    predicted_json_file_name: str | None = None,
):
    """compute_metrics.
    This function computes metrics once the GT graph and the predicted graphs
    are available.
    
    Parameters
    ----------
    gt_segmentation : np.ndarray
        gt_segmentation
    predicted_segmentation : np.ndarray
        predicted_segmentation
    results_dir_name : str
        results_dir_name
    gt_nx_graph : nx.DiGraph | None
        gt_nx_graph
    pred_nx_graph : nx.DiGraph | None
        pred_nx_graph
    gt_json_file_name : str | None
        gt_json_file_name
    predicted_json_file_name : str | None
        predicted_json_file_name
    """

    if gt_nx_graph is not None:
        print("Using gt nx graphs ...")
        gt_data = load_graph_masks(G=gt_nx_graph, masks=gt_segmentation, name="gt")
    elif gt_json_file_name is not None:
        print("Using gt json files ...")
        gt_data = load_deepcell_data(
            masks=gt_segmentation, json_file_name=gt_json_file_name, name="gt"
        )

    if pred_nx_graph is not None:
        print("Using val nx graphs ...")
        pred_data = load_graph_masks(
            G=pred_nx_graph, masks=predicted_segmentation, name="pred"
        )
    elif predicted_json_file_name is not None:
        print("Using val json files ...")
        pred_data = load_deepcell_data(
            masks=predicted_segmentation,
            json_file_name=predicted_json_file_name,
            name="pred",
        )

    print("Running CTC metrics now ...")
    ctc_results = run_metrics(
        gt_data=gt_data,
        pred_data=pred_data,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics()],       # DivisionMetrics()
    )
    pp.pprint(ctc_results)

    with open(results_dir_name + "/jsons/results.json", "w") as fp:
        json.dump(ctc_results, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_json_file_name", dest="gt_json_file_name")
    parser.add_argument("--gt_segmentation", dest="gt_segmentation")
    parser.add_argument("--pred_json_file_name", dest="pred_json_file_name")
    parser.add_argument("--pred_segmentation", dest="pred_segmentation")
    parser.add_argument("--results_dir", dest="results_dir")
    args = parser.parse_args()
    compute_metrics(
        args.gt_json_file_name,
        args.gt_segmentation,
        args.pred_json_file_name,
        args.pred_segmentation,
        args.results_dir,
    )
