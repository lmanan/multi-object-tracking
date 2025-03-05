
Please install the following environment to be able to run the `visualize_tracks.py` script:
These installation lines of code have been taken from the `motile-tracker` repository, which is available [here](https://github.com/funkelab/motile_tracker):

```
conda create -n motile-tracker python=3.10
conda activate motile-tracker
conda install -c conda-forge -c funkelab -c gurobi ilpy
pip install motile-tracker
pip install opencv-python-headless
```

Then run the `visualize_tracks.py` by running:

```
cd visualization
python visualize_tracks.py --yaml_configs_file_name visualize_configs.yaml
```

