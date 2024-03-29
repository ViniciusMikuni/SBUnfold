# Official repository for the paper "Improving Generative Model-based Unfolding with Schroedinger Bridges"

See the ```requirements.txt``` file for required libraries to reproduce the results presented. SBUnfold is based on the [I2SB](https://github.com/NVlabs/I2SB) implementation implemented in pytorch, while the OmniFold and cINN implementations are provided in Tensorflow.

# Data

The data used is available at [Zenodo](https://zenodo.org/record/3548091) and also available as part of the EnergyFlow package

# Running the scripts

To run SBUnfold do:
```bash
cd SBUnfold
python train_physics.py --corrupt 'SBUnfold' --n-gpu-per-node 1  --beta-max 0.1 [--ot-ode]
```
with flag ```ot-ode``` used to call the method using the OT-based implementation.

To run the other comparison algorithms, visit the relevant folder (cINN,omnifold), and run:
```bash
python train_physics.py
```

# Plotting

The results presented in the paper can be reproduced after training each individual unfolding algorithm  by calling:
```
python plot.py
```
where both plots and metrics are calculated. 


## Citation

```
@article{Diefenbacher:2023wec,
    author = "Diefenbacher, Sascha and Liu, Guan-Horng and Mikuni, Vinicius and Nachman, Benjamin and Nie, Weili",
    title = {{Improving Generative Model-based Unfolding with Schr\"odinger Bridges}},
    eprint = "2308.12351",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "8",
    year = "2023"
}
```