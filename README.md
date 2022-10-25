# DAMR：Dynamic Adjacency Matrix Representation Learning for Multivariate Time Series Imputation
This repository contains the code and datasets for the paper "Dynamic Adjacency Matrix Representation Learning for Multivariate Time Series Imputation". In this paper, we propose a new imputation method based on graph neural network architecture.

we design DAMR that extracts various dynamic patterns of spatial correlations and represents them as adjacency matrices. The adjacency matrices are then aggregated and fed into a well-designed graph representation learning layer for predicting the missing values.

![3 Architecture](https://user-images.githubusercontent.com/98369049/197794496-395db772-0912-48f8-b2d8-b5366ca221e0.jpg)

![10 AM layers](https://user-images.githubusercontent.com/98369049/197795715-4d7065e3-fa56-4468-ac9c-4df0757411c2.jpg)

# DATASET:
For IN, SZ, CO and NZ dataset, please refer to the data folder.
For CN dataset, please download the dataset from this paper: "Forecasting fine-grained air quality based on big data".
For KDD-CUP dataset, please download the dataset from this link: https://www.kdd.org/kdd2018/kdd-cup provided by the paper: "Dynamic Gaussian Mixture based Deep Generative Model For Robust Forecasting on Sparse Multivariate Time Series". You can add this dataset into our folder.

#Experiment
To better understand our approach, we use 
