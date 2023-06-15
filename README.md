# DAMR：Dynamic Adjacency Matrix Representation Learning for Multivariate Time Series Imputation
This repository contains the code and datasets for the paper "Dynamic Adjacency Matrix Representation Learning for Multivariate Time Series Imputation". In this paper, we propose a new imputation method based on graph neural network architecture.

we design DAMR that extracts various dynamic patterns of spatial correlations and represents them as adjacency matrices. The adjacency matrices are then aggregated and fed into a well-designed graph representation learning layer for predicting the missing values.

![3 Architecture](https://user-images.githubusercontent.com/98369049/197794496-395db772-0912-48f8-b2d8-b5366ca221e0.jpg)

![10 AM layers](https://user-images.githubusercontent.com/98369049/197795715-4d7065e3-fa56-4468-ac9c-4df0757411c2.jpg)


# Dataset:

For IN, SZ, CO and NZ dataset, please refer to the data folder.
For CN dataset, please download the dataset from the paper [here](https://dl.acm.org/doi/10.1145/2783258.2788573).
For KDD-CUP dataset, please download the dataset [here](https://www.kdd.org/kdd2018/kdd-cup)  provided by the paper [here](https://arxiv.org/abs/2103.02164) and add this dataset into our folder.

# Experiment:

To better understand our approach, we use DAMR_AIRQUALITY.ipynb for convenience.
In  DAMR_AIRQUALITY.ipynb, we provide the DAMR approach and several statistical baseline models: Mean, Sliding Window, MF, KNN and MICE.
For reproducibility on different datasets, please run DAMR_NETHERLAND.ipynb with complete datasets and modify the input path "data=Csv2Tensor('Data/COVID/raw')" by changing "COVID" with other datasets.

# Baseline models 
(1) GRIN:

Run the code on [GRIN](https://github.com/Graph-Machine-Learning-Group/grin/blob/main/requirements.txt), please run "python main.py" on baseline folder.

(2) BRITS:
Follow the steps on [BRITS](https://github.com/caow13/BRITS). 

1.Put the dataset into raw folder.
2.Make a empty folder named json, and run input_process.py.

(3) GRAPE:
Follow the steps on [GRAPE](https://github.com/maxiaoba/GRAPE).

1.Enter in the path GRAPE/uci/raw_data/concrete/data/, then modify data.txt into our datasets, eg. ND.
2.Run python train_mdi.py uci --data ND. 

# Ablation study:
Run 
DAMR_Ablation-Diffconv.ipynb,
DAMR_Ablation-GAT+Diffconv.ipynb,
DAMR_Ablation-GAT+GCN.ipynb,
DAMR_Ablation-GAT.ipynb,
DAMR_Ablation-GCN.ipynb,
DAMR_Ablation-Diffconv.ipynb
to conduct ablation study.


# Citation

- Xiaobin Ren, Kaiqi Zhao and Patricia Riddle et al. (2023). DAMR: Dynamic Adjacency Matrix Representation Learning for Multivariate Time Series Imputation. SIGMOD.https://doi.org/10.1145/3589333.


@article{


    xren,
    title = {{Serenade - Low-Latency Session-Based Recommendation in e-Commerce at Scale}},
    year = {2023},
    journal = {SIGMOD},
    author = {Xiaobin Ren and Kaiqi Zhao and Patricia Riddle and Katerina Ta\v{s}kova and Lianyan Li and Qingyi Pan}
    }

