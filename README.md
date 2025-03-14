### Installation

```bash
conda create -n motile-env python==3.10
conda activate motile-env
pip install jsonargparse torch natsort
conda install -c conda-forge -c funkelab -c gurobi ilpy
conda install scip
pip install git+https://github.com/lmanan/motile_toolbox.git
pip install git+https://github.com/lmanan/motile.git
pip install git+https://github.com/lmanan/traccuracy.git
```

### Experiments

```bash
conda activate motile-env
cd experiments
python ../src/infer.py --yaml_configs_file_name 'rat_city_cohort2_exp1.yaml'
```
