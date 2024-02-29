# AQTIVATE Workshop on ML - TU Berlin & BIFOLD

This repository contains the exercises performed during the AQTIVATE Machine Learning workshop at TU Berlin, more specifically BIFOLD.
Each folder contains the code developed, the relevant generated data and the .pdf file of the presentations given at the end of each exercise.

# Exercise 1

Dataset: [Patch camelyon dataset](https://patchcamelyon.grand-challenge.org/). The goal is to (manually) load this dataset, pre-process it and train a simple logistic classifier.

# Exercise 2

In this exercise we consider once again the Patch Camelyon Dataset. The goal is now to construct a deep CNN for this problem.
For this, we considered five different architectures: LeNet5, AlexNet, VGG16, ZFNet and ResNet. Moreover, we also (briefly) explored the hyperparameter space.

# Exercise 3

This exercise involves handling a dataset derived from molecular dynamics trajectories. The challenge here is that the dataset is missing structures for all minima (only two out of the three of ethanol), and the data is stored in XYZ files. 
The primary objective is to become acquainted with the dataset, and preapre the data for further analysis using [SchnetPack](https://github.com/atomistic-machine-learning/schnetpack)
# Exercise 4

In this final exercise, we delve deeper into the practical aspects of data analysis and neural networks by leveraging force fields to investigate chemical space. The goal is to accurately identify local minima within molecular structures,
extending the search beyond those detected through initial ocular inspection, which may not correspond to the molecule's minimum energy configurations.
