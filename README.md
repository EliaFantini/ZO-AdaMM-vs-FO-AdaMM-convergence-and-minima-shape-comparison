<p align="center">
  <img alt="0️⃣ZO-AdaMM" src="https://user-images.githubusercontent.com/62103572/183616627-c8fe0034-1edd-4158-805e-0e5f58ced481.png">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/EliaFantini/ZO-AdaMM-vs-FO-AdaMM-convergence-and-minima-shape-comparison">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/EliaFantini/ZO-AdaMM-vs-FO-AdaMM-convergence-and-minima-shape-comparison">
  <img alt="GitHub code size" src="https://img.shields.io/github/languages/code-size/EliaFantini/ZO-AdaMM-vs-FO-AdaMM-convergence-and-minima-shape-comparison">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/EliaFantini/ZO-AdaMM-vs-FO-AdaMM-convergence-and-minima-shape-comparison">
  <img alt="GitHub follow" src="https://img.shields.io/github/followers/EliaFantini?label=Follow">
  <img alt="GitHub fork" src="https://img.shields.io/github/forks/EliaFantini/ZO-AdaMM-vs-FO-AdaMM-convergence-and-minima-shape-comparison?label=Fork">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/EliaFantini/ZO-AdaMM-vs-FO-AdaMM-convergence-and-minima-shape-comparison?label=Watch">
  <img alt="GitHub star" src="https://img.shields.io/github/stars/EliaFantini/ZO-AdaMM-vs-FO-AdaMM-convergence-and-minima-shape-comparison?style=social">
</p>

The adaptive momentum method (AdaMM, aka AMSGrad) is a
first-order optimisation method that is increasingly used to
solve deep learning problems. However, like any first-order (FO)
method, it is only usable when the gradient is computable. When
this is not the case, a zero-order (ZO) version of the method can
be used. This method was proposed in [ZO-AdaMM: Zeroth-Order Adaptive Momentum Method for Black-Box Optimization, Xiangyi Chen et al](https://proceedings.neurips.cc/paper/2019/file/576d026223582a390cd323bef4bad026-Paper.pdf).

This project aims at comparing ZO-AdaMM with the original first order method (FO-AdaMM). 

In our experiments we have observed the theoretical slowdown of the order of **O(√d)** in the
convergence of the ZO method compared to the FO method, where *d* is the number of the network's parameters to be optimized.
Moreover, we have managed to obtain reasonable performances
with our ZO method and have proposed improvement ways
to obtain better performances. Finally, we have highlighted the
convergence to different minima for the ZO and FO methods.

This project was done for the EPFL course [CS-439 Optimization for Machine Learning](https://edu.epfl.ch/coursebook/en/optimization-for-machine-learning-CS-439), taught by  Jaggi Martin and Flammarion Nicolas Henri Bernard.

## Authors
- [Kieran Vaudaux](https://github.com/KieranVaudaux)
- [Elia Fantini](https://github.com/EliaFantini)
- [Cyrille Pittet](https://github.com/cpittet)

## Results

We empirically studied the ZO-AdaMM optimizer comparing it with the FO-AdaMM one, applying it on simple convolutional neural networks (CNN) ranging from 1'400 to more
than 2.5 millions parameters ( represented by the letter *d*) on the well known classification task of the MNIST dataset. 

Although it achieves acceptable accuracies, the ZO
version still suffers from a certain slowdown compared to
the FO version. This is due to the fact that our ZO version
allows the parameters to move in a single direction which is
certainly biased with respect to the exact gradient. However,
the accuracy continued to increase slowly and the train loss
to decrease, so we think we would achieve better results even
if it would take some time. 

Then, we were interested in the
appearance of this theoretical bound of O(√d). According to
our experiments, this theoretical bound does indeed seem to
occur, which would make it difficult to use our method for
larger and more complex models. As shown in the following image, the training stabilizes to an almost constant ratio of FO/ZO performance.

<p align="center">
<img width="500" alt="aa" src="https://user-images.githubusercontent.com/62103572/183623246-a6805e1a-c46a-4c2e-87f5-2897242647f2.png">
</p>

Since this ratio increases with the number of parameters *d*, this seem to indicate that the theoretical bound of O(√d) is appearing. Indeed, in the following image we can see that such bound that we called *k* is almost constant, it varies very little (from 0.2 to 0.6) despite the changing of *d*.

<p align="center">
<img width="600" alt="cc" src="https://user-images.githubusercontent.com/62103572/183624158-b048ba12-a62a-4591-a464-c251e4d2085b.png">
</p>


Finally, we tried to project
the weights of each individual filter of the CNN into a 2-dimensional
space using the t-SNE algorithm, which tries to preserve
in low dimensions the neighborhood of the data points in
high dimensions. As the following image shows, the learned weights to which the models converge are very different.

<p align="center">
<img width="600" alt="bb" src="https://user-images.githubusercontent.com/62103572/183623250-69231021-4e58-412e-92ed-a3d7c92b5365.png">
</p>

The next image shows the actual filters learned, printed as grey-scale images. As we can see, they don't show similar patterns.
<p align="center">
<img width="400" alt="dd" src="https://user-images.githubusercontent.com/62103572/183628281-a4b9a7d0-2648-4ff6-a52b-df76c1cf9969.png">
</p>

Although the fact that the parameters converge to quite distinct
minima, deducing that one of these minima is ”better” than
the other is a complicated subject. Nevertheless, the fact that
for equal training losses, the FO method has a much higher
accuracies than the ZO method, leads us to believe that the
ZO method converges to a local minimum and the FO to a
global minimum or at least a ”better” local minimum.


The major improvement that can be done to our method lies in reducing
the bias of the gradient estimation by using a mini-batch
of random directions as in Eq.2. We believe that this would
allow us to have a better chance of converging this time to the
minima reached by the FO method and also to accelerate the
convergence, which so far remains rather slow as we explore
the parameter space one direction at a time.

For further details, please read the pdf **report.pdf**.

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
## How to install and reproduce results
Download this repository as a zip file and extract it into a folder. The easiest way to run the code is to install Anaconda 3 distribution (available for Windows, macOS and Linux). To do so, follow the guidelines from the official
website (select python of version 3): https://www.anaconda.com/download/. The libraries required to run our code can be found in ```requirements.txt```.

Such additional packages required are: 
- torch
- torchvision
- numpy
- matplotlib
- scipy
- sklearn
- jupyter notebooks

To install them write the following command on Anaconda Prompt (anaconda3):
```shell
cd *THE_FOLDER_PATH_WHERE_YOU_DOWNLOADED_AND_EXTRACTED_THIS_REPOSITORY*
```
Then write for each of the mentioned packages:
```shell
conda install *PACKAGE_NAME*
```
Some packages might require more complex installation procedures (especially [pytorch](https://pytorch.org/)). If the above command doesn't work for a package, just google "How to install *PACKAGE_NAME* on *YOUR_MACHINE'S_OS*" and follow those guides.

The results can be reproduced as follows :
- Run the ```experiments.ipynb``` to produce the data
- Run the ```analysis.ipynb``` to produce the plots used in the report

Remarks : 
- You need to create the folders ```./results``` and ```./results/weights``` if they
do not exist in your system.
- The zero order optimization method can only be used on the CPU as it produced different behaviors on different machines when using the GPU, with some GPU achieving a lower accuracy and higher losses compared to CPU results, even while using same random seeds. If you still want to use the GPU, you can comment line 54 and decomment line 55 in ```main.py```.

## 🛠 Skills

Python, Pytorch. Deep learning knowledge, Machine Learning optimization knowledge, implementation of first order AdaMM (AMSGrad) and zero order AdaMM optimizers, as well as ZO-SGD. Study and analysis of convergence rates, minima shape with t-SNE alogrithm, visual analysis of CNN's learned filters, cosine similarity.

## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://eliafantini.github.io/Portfolio/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/-elia-fantini/)
