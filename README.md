# Feature Context Effects

This repository accompanies the paper

> Kiran Tomlinson and Austin R. Benson. Learning Feature Context Effects in Discrete Choice. 2020. (arXiv link soon)

## Libraries
We used:
- Python 3.8.5
  - matplotlib 3.3.0
  - networkx 2.4
  - numpy 1.19.1
  - pandas 1.0.5
  - scipy 1.5.1
  - tqdm 4.48.0
  - torch 1.5.1
  - scikit-learn 0.23.1
  - statsmodels 0.11.1
  
## Files
- `models.py`: implementations of the LCL, DLCL, MNL, and mixed logit models and training procedures
- `datasets.py`: dataset processing and management
- `experiments.py`: parallelized hyperparameter search and model training
- `plot.py`: makes plots and tables in the paper

The `params/` directory contained trained model parameters in PyTorch format, the `results/` directory contains other experiment outputs, 
and the `hyperparams/` directory contains grid search results used for selecting hyperparameters.

Two classes in `models.py` (noted in comments) are from [Arjun Seshadri's CDM code](https://github.com/arjunsesh/cdm-icml), which accompanies the paper
> Arjun Seshadri, Alex Peysakhovich, and Johan Ugander. 2019. Discovering Context Effects from Raw Choice Data. In International Conference on Machine Learning. 5660–5669.

## Data
All of our datasets are available in two formats. First, they are available [here](https://drive.google.com/file/d/1QAr-tCZ4OWRcrsQ0tHYwmTate5ED21PI/view) in a
common text format with documentation about the original sources, features, and preprocessing steps. For running our code, we also provide binary versions of the
datasets [here](https://drive.google.com/file/d/1kzavt-Kr3vSSzpwqpG0XtNHAeJeXTwqF/view). Place the `.pickle` files in a directory named `data/` in 
`feature-context-effects/`.

## Reproducibility
To create the plots and tables in the paper from the data provided in the repository, just run `python3 plot.py`. To run all experiments, 
run `python3 experiments.py` (by default, this uses 30 threads--you may wish to modify this in the code). It takes a while to run everything (several days
on 30 cores), but you can also run individual experiments by commenting out other method calls at the bottom of `experiments.py`.
