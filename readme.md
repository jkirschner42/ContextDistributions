This repository contains the code used in the experiments of the Neurips 2019 Paper
"Stochastic Bandits with Context Distributions"

The paper can be found here: https://arxiv.org/abs/1906.02685

To reproduce the experiments, follow these steps in the main directory of the repository:

1. Install a Python 3.6 environment
2. Install the febo package with "pip install -e febo" . This is our framework for testing algorithms.
3. Install the sbcd package with "pip install -e sbcd" . This is the implementation of the context distribution environment and algorithms.
4. Create a 'runs' directory with `mkdir runs`


Instructions to run experiments and create plots:

1. febo create {experiment_name} --config config/{experiment_name}.yaml
2. febo run {experiment_name}
                (This will take a while, you can set the number of repetitions in the yaml file.)
3. febo plot {experiment_name} --plots febo.plots.Regret


To reproduce the experiments, replace "{experiment_name}" in the steps above by any of:

* synthetic
* movielens
* crops
