# DAMR：Dynamic Adjacency Matrix Representation Learning for Multivariate Time Series Imputation
This repository contains the code and datasets for the paper "Dynamic Adjacency Matrix Representation Learning for Multivariate Time Series Imputation". In this paper, we propose a new imputation method based on graph neural network architecture.

we design DAMR that extracts various dynamic patterns of spatial correlations and represents them as adjacency matrices. The adjacency matrices are then aggregated and fed into a well-designed graph representation learning layer for predicting the missing values.

![3 Architecture](https://user-images.githubusercontent.com/98369049/197794496-395db772-0912-48f8-b2d8-b5366ca221e0.jpg)

![10 AM layers](https://user-images.githubusercontent.com/98369049/197795715-4d7065e3-fa56-4468-ac9c-4df0757411c2.jpg)


#Dataset:

For IN, SZ, CO and NZ dataset, please refer to the data folder.
For CN dataset, please download the dataset from this paper: "Forecasting fine-grained air quality based on big data".
For KDD-CUP dataset, please download the dataset from this link: https://www.kdd.org/kdd2018/kdd-cup provided by the paper: "Dynamic Gaussian Mixture based Deep Generative Model For Robust Forecasting on Sparse Multivariate Time Series". You can add this dataset into our folder.

#Experiment:

To better understand our approach, we use several timestamps of DAMR_AIRQUALITY.ipynb and DAMR_NETHERLAND.ipynb datasets for convenience.
In  DAMR_AIRQUALITY.ipynb and DAMR_NETHERLAND.ipynb, we provide the DAMR approach and several statistical baseline models: Mean, Sliding Window, MF, KNN and MICE.
For reproducibility on different datasets, please run DAMR_NETHERLAND.ipynb with complete datasets and modify the input path "data=Csv2Tensor('Data/COVID/raw')" as other datasets.

#Baseline models on GRIN:

Run the code on https://github.com/Graph-Machine-Learning-Group/grin/blob/main/requirements.txt, please run "python main.py" on baseline folder.

#Baseline models BRITS:
Follow the steps on BRITS:https://github.com/caow13/BRITS. 

1.Put the dataset into raw folder.
2.Make a empty folder named json, and run input_process.py.

#Baseline models GRAPE:
Follow the steps on GRAPE: https://github.com/maxiaoba/GRAPE and make minor changes.

1.Enter in the path GRAPE/uci/raw_data/concrete/data/, then modify data.txt into our datasets, eg. ND.
2.Run python train_mdi.py uci --data ND. 

#Ablation study:
Run 
DAMR_Ablation-Diffconv.ipynb,
DAMR_Ablation-GAT+Diffconv.ipynb,
DAMR_Ablation-GAT+GCN.ipynb,
DAMR_Ablation-GAT.ipynb,
DAMR_Ablation-GCN.ipynb,
DAMR_Ablation-Diffconv.ipynb
to conduct ablation study.
