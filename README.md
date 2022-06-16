# Mini project - CS-439 Optimization for Machine Learning
# Zero Order Adaptive Momentum Method (ZO-AdaMM)
## Introduction
 This repository contains the code to reproduce the results of the mini-project done in
 the context of the course CS-439 Optimization for Machine Learning at EPFL.
 
We propose to study the behavior of the zero order version of the AdaMM algorithm (aka AMSGrad), called ZO-AdaMM.
This method was proposed in *ZO-AdaMM: Zeroth-Order Adap-
tive Momentum Method for Black-Box Optimization*, Xiangyi Chen et al.

In particular, we empirically studied this optimizer with simple CNN ranging from 1'400 to more
than 2.5 millions parameters on well known classification task of the MNIST dataset.

## Structure of the repository

```
├── models
    ├── scalable_model.py    # Scalable (nb. params) CNN
    ├── small_model.py       # Small CNN used for tests
├── optimizers
    ├── adamm.py             # First order AdaMM optimizer
    ├── zo_adamm.py          # Zeroth order AdaMM optimizer
    ├── zo_sgd.py            # Zeroth order SGD optimizer
    ├── scheduler.py         # Learning rate scheduler
├── plots
├── results                  # Results of the experiment notebook
    ├── weights              # Recorded weights in the experiment notebook
├── main.py                  # Main functions to setup the training of a model, run the experiments
├── utils.py                 # Functions to train a model and some utilitaries functions
├── experiments.ipynb        # Notebook containing the experiments (models training)
├── analysis.ipynb           # Notebook containing the analysis of the experiments, with plots
├── report.pdf               # The report of the project
├── requirements.txt         # List of all the packages needed to run our code
└── README.md                # You are here
```

## Reproducing our results
The libraries required to run our code can be found in ```requirements.txt```.

The results can be reproduced as follows :
- Run the ```experiments.ipynb``` to produce the data
- Run the ```analysis.ipynb``` to produce the plots used in the report

Please note that you need to create the folders ```./results``` and ```./results/weights``` if they
do not exist in your system.

## Authors
- Kieran Vaudaux
- Elia Fantini
- Cyrille Pittet