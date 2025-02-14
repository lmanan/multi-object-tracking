import logging
from pathlib import Path

import napari
import pandas as pd
import zarr
from appdirs import AppDirs
from napari.utils.theme import _themes

from motile_tracker.application_menus import MainApp
from motile_tracker.data_views import TreeWidget

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s:%(lineno)d] %(levelname)-8s %(message)s",
)
logging.getLogger("motile_tracker").setLevel(logging.DEBUG)

_themes["dark"].font_size = "18pt"


# Load Zarr datasets
csv_path = "annotation_yolo.txt"  # replace with your points file
dataframe = pd.read_csv(csv_path, sep=" ")

position_columns = ["frame_id", "y", "x"]  # replace with your position columns
positions = dataframe[position_columns].to_numpy()
positions[:,1:2] = positions[:,1:2] * 1000		# y axis
positions[:,2:3] = positions[:,2:3] * 1100		# x axis
positions[:,0] = positions[:,0]-1

# Initialize Napari viewer
viewer = napari.Viewer()

# Add image and label layers to the viewer
viewer.add_points(positions, name="points", size=50)
# Add your custom widget
widget = MainApp(viewer)
viewer.window.add_dock_widget(widget, name="Motile")

# Start the Napari GUI event loop
napari.run()


# sudo mount -o rw,hard,bg,nolock,nfsvers=4.1,sec=krb5 nrs.hhmi.org:/nrs/ /nrs/
# https://funkelab.github.io/motile_tracker/getting_started.html#installation
# conda create -n motile-tracker python=3.10
# conda activate motile-tracker
# conda install -c conda-forge -c funkelab -c gurobi ilpy
# pip install motile-tracker
