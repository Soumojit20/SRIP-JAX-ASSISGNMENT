# SRIP-JAX-ASSISGNMENT

Implement two hidden layers neural network 
classifier from scratch in JAX

## Authors

- SOUMOJIT BASU -https://github.com/Soumojit20

## Acknowledgements

 -jax tutorial playlist link -https://www.youtube.com/playlist?list=PLBoQnSflObckOARbMK9Lt98Id0AKcZurq
 
 -Getting started with JAX (MLPs, CNNs & RNNs)-https://roberttlange.github.io/posts/2020/03/blog-post-10/
 
 -jax mlp - https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
 
 ## AIM OF THE PROJECT:-
The aim of the project is to create a two layer hidden neural network from scratch using jax
in which we are using the MNIST dataset to create the project.We are using the pytree concept in jax to implement the project gracefully.
In which we are splitting the mnist dataset into train test split in the ratio of 80:20.

## WHAT IS JAX
JAX (Just After eXecution) is a recent machine/deep learning library developed by DeepMind.

Features of JAX:-

JAX is basically a Just-In-Time (JIT) compiler focused on harnessing the maximum number of FLOPs to generate optimized code while using the simplicity of pure Python. Some of the salient features of JAX are:

Just-in-Time (JIT) compilation. ##THIS BASICALLY FASTEN UP THE FUNCTION

Enables NumPy code on not only CPU but GPU and TPU as well.
Automatic differentiation of NumPy and native Python code
Automatic vectorization.
Express and compose transformations of numerical programs.
Advanced (pseudo) random number generation.
More options for control flow.

## How to Use the Project
-MNIST Handwritten Digit Classification Dataset
-It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.

It is a widely used and deeply understood dataset and, for the most part, is “solved.” Top-performing models are deep learning convolutional neural networks that achieve a classification accuracy of above 99%, with an error rate between 0.4 %and 0.2% on the hold out test dataset.

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

import jax 

import numpy as np

import jax.numpy as jnp 

from jax.scipy.special import logsumexp

from torchvision.datasets import MNIST

from torch.utils.data import DataLoader

from jax import jit, vmap, pmap, grad, value_and_grad
