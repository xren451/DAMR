def KNN_impute(Data_missing,Data_true):
    import pandas as pd
    import numpy as np
    from math import sqrt
    from sklearn.preprocessing import MinMaxScaler
    # importing the KNN from fancyimpute library
    from fancyimpute import KNN
    Data_missing=np.array(Data_missing)
    Data_true=np.array(Data_true)
    # calling the KNN class
    knn_imputer = KNN()
    # imputing the missing value with knn imputer
    Data_KNN = knn_imputer.fit_transform(Data_missing)
    from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
    missing_mask=[]
    for i in range(Data_missing.shape[0]):
        missing_mask.append(np.isnan(np.array(Data_missing[i])))
    missing_mask=np.array(missing_mask).reshape(Data_missing.shape[0],Data_missing.shape[1])
    #Normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled1 = scaler.fit_transform(Data_true)
    scaled2 = scaler.fit_transform(Data_KNN)
    Data_true=scaled1
    Data_KNN=scaled2
    #missing_mask=missing_mask.
    KNN_rmse = sqrt(((np.array(Data_true)[missing_mask] - np.array(Data_KNN)[missing_mask]) ** 2).mean())
    KNN_mae= abs((np.array(Data_true)[missing_mask]-np.array(Data_KNN)[missing_mask])).mean()
#     KNN_mape= abs((np.array(Data_true)[missing_mask]-np.array(Data_KNN)[missing_mask])/np.array(Data_true)[missing_mask]).mean()
    KNN_mape=np.mean(np.abs((np.array(Data_true)[missing_mask]-np.array(Data_KNN)[missing_mask]) / np.clip(np.abs(np.array(Data_true)[missing_mask]), 0.1, None)),axis=-1).mean()
    KNN_mre=sum(abs((np.array(Data_true)[missing_mask]-np.array(Data_KNN)[missing_mask])))/sum(abs(np.array(Data_true)[missing_mask]))   
    KNN_middle1=np.array(Data_true)[missing_mask].mean()
    print("KNNImpute MAE: %f" % KNN_mae)
    print("KNNImpute MAPE: %f" % KNN_mape)
    print("KNNImpute MRE: %f" % KNN_mre)
    print("KNNImpute RMSE: %f" % KNN_rmse)
    print("KNNImpute middle1: %f" % KNN_middle1)

def MICE_impute(Data_missing,Data_true):
    import pandas as pd
    import numpy as np
    from math import sqrt
    # importing IterativeImputer from fancyimpute library
    from fancyimpute import IterativeImputer
    from sklearn.preprocessing import MinMaxScaler
    Data_missing=np.array(Data_missing)
    Data_true=np.array(Data_true)
    # calling the IterativeImputer class
    MICE_imputer = IterativeImputer()
    # imputing the missing value with knn imputer
    Data_MICE = MICE_imputer.fit_transform(Data_missing)
    from fancyimpute import NuclearNormMinimization, SoftImpute, BiScaler
    missing_mask=[]
    for i in range(Data_missing.shape[0]):
        missing_mask.append(np.isnan(np.array(Data_missing[i])))
    missing_mask=np.array(missing_mask).reshape(Data_missing.shape[0],Data_missing.shape[1])
    #Normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled1 = scaler.fit_transform(Data_true)
    scaled2 = scaler.fit_transform(Data_MICE)
    Data_true=scaled1
    Data_MICE=scaled2
    #missing_mask=missing_mask.
    MICE_rmse = sqrt(((np.array(Data_true)[missing_mask] - np.array(Data_MICE)[missing_mask]) ** 2).mean())
    MICE_mae= abs((np.array(Data_true)[missing_mask]-np.array(Data_MICE)[missing_mask])).mean()
    MICE_mape= abs((np.array(Data_true)[missing_mask]-np.array(Data_MICE)[missing_mask])/np.array(Data_true)[missing_mask]).mean()
    MICE_mre=sum(abs((np.array(Data_true)[missing_mask]-np.array(Data_MICE)[missing_mask])))/sum(abs(np.array(Data_true)[missing_mask]))
    print("MICEImpute MAE: %f" % MICE_mae)
    print("MICEImpute MAPE: %f" % MICE_mape)
    print("MICEImpute MRE: %f" % MICE_mre)
    print("MICEImpute RMSE: %f" % MICE_rmse)
    
def MF_impute(Data_missing,Data_true):
    import pandas as pd
    import numpy as np
    from math import sqrt
    # importing IterativeImputer from fancyimpute library
    from fancyimpute import IterativeImputer
    from fancyimpute import NuclearNormMinimization, SoftImpute, BiScaler
    from sklearn.preprocessing import MinMaxScaler
    Data_missing=np.array(Data_missing)
    Data_true=np.array(Data_true)
    # calling the IterativeImputer class
    MF_imputer = NuclearNormMinimization()
    # imputing the missing value with knn imputer
    Data_MF = MF_imputer.fit_transform(Data_missing)
    missing_mask=[]
    for i in range(Data_missing.shape[0]):
        missing_mask.append(np.isnan(np.array(Data_missing[i])))
    missing_mask=np.array(missing_mask).reshape(Data_missing.shape[0],Data_missing.shape[1])
    #Normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled1 = scaler.fit_transform(Data_true)
    scaled2 = scaler.fit_transform(Data_MF)
    Data_true=scaled1
    Data_MF=scaled2
    #missing_mask=missing_mask.
    MF_rmse = sqrt(((np.array(Data_true)[missing_mask] - np.array(Data_MF)[missing_mask]) ** 2).mean())
    MF_mae= abs((np.array(Data_true)[missing_mask]-np.array(Data_MF)[missing_mask])).mean()
    MF_mape= abs((np.array(Data_true)[missing_mask]-np.array(Data_MF)[missing_mask])/np.array(Data_true)[missing_mask]).mean()
    MF_mre=sum(abs((np.array(Data_true)[missing_mask]-np.array(Data_MF)[missing_mask])))/sum(abs(np.array(Data_true)[missing_mask]))
    print("MFImpute MAE: %f" % MF_mae)
    print("MFImpute MAPE: %f" % MF_mape)
    print("MFImpute MRE: %f" % MF_mre)
    print("MFImpute RMSE: %f" % MF_rmse)
def MEAN_impute(Data_missing,Data_true):
    import pandas as pd
    import numpy as np
    from math import sqrt
    from fancyimpute import IterativeImputer
    from sklearn.preprocessing import MinMaxScaler
    from fancyimpute import SimpleFill, NuclearNormMinimization, SoftImpute, BiScaler
    Data_missing=np.array(Data_missing)
    Data_true=np.array(Data_true)
    MEAN_imputer = SimpleFill()
    Data_MEAN = MEAN_imputer.fit_transform(Data_missing)
    missing_mask=[]
    for i in range(Data_missing.shape[0]):
        missing_mask.append(np.isnan(np.array(Data_missing[i])))
    missing_mask=np.array(missing_mask).reshape(Data_missing.shape[0],Data_missing.shape[1])
    #Normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled1 = scaler.fit_transform(Data_true)
    scaled2 = scaler.fit_transform(Data_MEAN)
    Data_true=scaled1
    Data_MEAN=scaled2
    #missing_mask=missing_mask.
    MEAN_rmse = sqrt(((np.array(Data_true)[missing_mask] - np.array(Data_MEAN)[missing_mask]) ** 2).mean())
    MEAN_mae= abs((np.array(Data_true)[missing_mask]-np.array(Data_MEAN)[missing_mask])).mean()
    MEAN_mape= abs((np.array(Data_true)[missing_mask]-np.array(Data_MEAN)[missing_mask])/np.array(Data_true)[missing_mask]).mean()
    MEAN_mre=sum(abs((np.array(Data_true)[missing_mask]-np.array(Data_MEAN)[missing_mask])))/sum(abs(np.array(Data_true)[missing_mask]))
    print("MEANImpute MAE: %f" % MEAN_mae)
    print("MEANImpute MAPE: %f" % MEAN_mape)
    print("MEANImpute MRE: %f" % MEAN_mre)
    print("MEANImpute RMSE: %f" % MEAN_rmse)
def Sliding_impute(Data_missing,Data_true,window):
    import pandas as pd
    import numpy as np
    from math import sqrt
    from sklearn.preprocessing import MinMaxScaler
    Data_missing=np.array(Data_missing)
    Data_true=np.array(Data_true)
    missing_mask=[]
    for i in range(Data_missing.shape[0]):
        missing_mask.append(np.isnan(np.array(Data_missing[i])))
    missing_mask=np.array(missing_mask).reshape(Data_missing.shape[0],Data_missing.shape[1])
    #Sliding window
    Data_Sliding=Data_missing
    Data_Sliding=pd.DataFrame(Data_Sliding)
    for i in range(Data_Sliding.shape[0]):
        for j in range(Data_Sliding.shape[1]):
            if np.isnan(Data_Sliding.iloc[i,j]):
                Data_Sliding.iloc[i,j]=np.mean(Data_Sliding.iloc[max(0,i-window):min(Data_Sliding.shape[0],i+window),j])
    Data_Sliding=np.array(Data_Sliding)
    #Normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled1 = scaler.fit_transform(Data_true)
    scaled2 = scaler.fit_transform(Data_Sliding)
    Data_true=scaled1
    Data_Sliding=scaled2
    #missing_mask=missing_mask.
    Sliding_rmse = sqrt(((np.array(Data_true)[missing_mask] - np.array(Data_Sliding)[missing_mask]) ** 2).mean())
    Sliding_mae= abs((np.array(Data_true)[missing_mask]-np.array(Data_Sliding)[missing_mask])).mean()
    Sliding_mape= abs((np.array(Data_true)[missing_mask]-np.array(Data_Sliding)[missing_mask])/np.array(Data_true)[missing_mask]).mean()
    Sliding_mre=sum(abs((np.array(Data_true)[missing_mask]-np.array(Data_Sliding)[missing_mask])))/sum(abs(np.array(Data_true)[missing_mask]))
    return Sliding_mae,Sliding_mape,Sliding_mre,Sliding_rmse,Data_Sliding
def KNN_impute_3D(Data_missing,Data_true,missing_mask):#Return:RMSE,MAE,MAPE,MRE
    from fancyimpute import KNN
    import pandas as pd
    import numpy as np
    from math import sqrt
    # calling the KNN class.
    knn_imputer = KNN()
    KNN_rmse=[]
    KNN_mae=[]
    KNN_mape=[]
    KNN_mre=[]
    Data_KNN=[]
    for i in range(Data_missing.shape[2]):
        Data_KNN.append(knn_imputer.fit_transform(Data_missing[:,:,i]))
    Data_KNN=np.array(Data_KNN).transpose(1,2,0)
    for i in range(Data_KNN.shape[2]):
        KNN_rmse.append(sqrt(((np.array(Data_true)[missing_mask] - np.array(Data_KNN)[missing_mask]) ** 2).mean()))
        KNN_mae.append(abs((np.array(Data_true)[missing_mask]-np.array(Data_KNN)[missing_mask])).mean())
#         KNN_mape.append(abs((np.array(Data_true)[missing_mask]-np.array(Data_KNN)[missing_mask])/np.array(Data_true)[missing_mask]).mean())
        KNN_mre.append(sum(abs((np.array(Data_true)[missing_mask]-np.array(Data_KNN)[missing_mask])))/sum(abs(np.array(Data_true)[missing_mask])))
        KNN_mape.append(np.mean(np.abs((np.array(Data_true)[missing_mask]-np.array(Data_KNN)[missing_mask]) / np.clip(np.abs(np.array(Data_true)[missing_mask]), 0.1, None)),axis=-1))
    KNN_rmse=np.mean(np.array(KNN_rmse))
    KNN_mae=np.mean(np.array(KNN_mae))
    KNN_mape=np.mean(np.array(KNN_mape))
    KNN_mre=np.mean(np.array(KNN_mre))
    return KNN_rmse,KNN_mae,KNN_mape,KNN_mre
def MICE_impute_3D(Data_missing,Data_true,missing_mask):#Return:RMSE,MAE,MAPE,MRE
    from fancyimpute import IterativeImputer
    import pandas as pd
    import numpy as np
    from math import sqrt
    # calling the KNN class.
    MICE_imputer = IterativeImputer()
    MICE_rmse=[]
    MICE_mae=[]
    MICE_mape=[]
    MICE_mre=[]
    Data_MICE=[]
    for i in range(Data_missing.shape[2]):
        Data_MICE.append(MICE_imputer.fit_transform(Data_missing[:,:,i]))
    Data_MICE=np.array(Data_MICE).transpose(1,2,0)
    for i in range(Data_MICE.shape[2]):
        MICE_rmse.append(sqrt(((np.array(Data_true)[missing_mask] - np.array(Data_MICE)[missing_mask]) ** 2).mean()))
        MICE_mae.append(abs((np.array(Data_true)[missing_mask]-np.array(Data_MICE)[missing_mask])).mean())
#         MICE_mape.append(abs((np.array(Data_true)[missing_mask]-np.array(Data_MICE)[missing_mask])/np.array(Data_true)[missing_mask]).mean())
        MICE_mre.append(sum(abs((np.array(Data_true)[missing_mask]-np.array(Data_MICE)[missing_mask])))/sum(abs(np.array(Data_true)[missing_mask])))
        MICE_mape.append(np.mean(np.abs((np.array(Data_true)[missing_mask]-np.array(Data_MICE)[missing_mask]) / np.clip(np.abs(np.array(Data_true)[missing_mask]), 0.1, None)),axis=-1))
    MICE_rmse=np.mean(np.array(MICE_rmse))
    MICE_mae=np.mean(np.array(MICE_mae))
    MICE_mape=np.mean(np.array(MICE_mape))
    MICE_mre=np.mean(np.array(MICE_mre))
    return MICE_rmse,MICE_mae,MICE_mape,MICE_mre
def MF_impute_3D(Data_missing,Data_true,missing_mask):#Return:RMSE,MAE,MAPE,MRE
    from fancyimpute import NuclearNormMinimization
    import numpy as np
    from math import sqrt
    # calling the KNN class.
    MF_imputer = NuclearNormMinimization()
    MF_rmse=[]
    MF_mae=[]
    MF_mape=[]
    MF_mre=[]
    Data_MF=[]
    for i in range(Data_missing.shape[2]):
        Data_MF.append(MF_imputer.fit_transform(Data_missing[:,:,i]))
    Data_MF=np.array(Data_MF).transpose(1,2,0)
    for i in range(Data_MF.shape[2]):
        MF_rmse.append(sqrt(((np.array(Data_true)[missing_mask] - np.array(Data_MF)[missing_mask]) ** 2).mean()))
        MF_mae.append(abs((np.array(Data_true)[missing_mask]-np.array(Data_MF)[missing_mask])).mean())
#         MF_mape.append(abs((np.array(Data_true)[missing_mask]-np.array(Data_MF)[missing_mask])/np.array(Data_true)[missing_mask]).mean())
        MF_mre.append(sum(abs((np.array(Data_true)[missing_mask]-np.array(Data_MF)[missing_mask])))/sum(abs(np.array(Data_true)[missing_mask])))
        MF_mape.append(np.mean(np.abs((np.array(Data_true)[missing_mask]-np.array(Data_MF)[missing_mask]) / np.clip(np.abs(np.array(Data_true)[missing_mask]), 0.1, None)),axis=-1))
    MF_rmse=np.mean(np.array(MF_rmse))
    MF_mae=np.mean(np.array(MF_mae))
    MF_mape=np.mean(np.array(MF_mape))
    MF_mre=np.mean(np.array(MF_mre))
    return MF_rmse,MF_mae,MF_mape,MF_mre
def MEAN_impute_3D(Data_missing,Data_true,missing_mask):#Return:RMSE,MAE,MAPE,MRE
    from fancyimpute import SimpleFill
    import pandas as pd
    import numpy as np
    from math import sqrt
    # calling the KNN class.
    MEAN_imputer = SimpleFill()
    MEAN_rmse=[]
    MEAN_mae=[]
    MEAN_mape=[]
    MEAN_mre=[]
    Data_MEAN=[]
    for i in range(Data_missing.shape[2]):
        Data_MEAN.append(MEAN_imputer.fit_transform(Data_missing[:,:,i]))
    Data_MEAN=np.array(Data_MEAN).transpose(1,2,0)
    for i in range(Data_MEAN.shape[2]):
        MEAN_rmse.append(sqrt(((np.array(Data_true)[missing_mask] - np.array(Data_MEAN)[missing_mask]) ** 2).mean()))
        MEAN_mae.append(abs((np.array(Data_true)[missing_mask]-np.array(Data_MEAN)[missing_mask])).mean())
#         MEAN_mape.append(abs((np.array(Data_true)[missing_mask]-np.array(Data_MEAN)[missing_mask])/np.array(Data_true)[missing_mask]).mean())
        MEAN_mre.append(sum(abs((np.array(Data_true)[missing_mask]-np.array(Data_MEAN)[missing_mask])))/sum(abs(np.array(Data_true)[missing_mask])))
        MEAN_mape.append(np.mean(np.abs((np.array(Data_true)[missing_mask]-np.array(Data_MEAN)[missing_mask]) / np.clip(np.abs(np.array(Data_true)[missing_mask]), 0.1, None)),axis=-1))
    MEAN_rmse=np.mean(np.array(MEAN_rmse))
    MEAN_mae=np.mean(np.array(MEAN_mae))
    MEAN_mape=np.mean(np.array(MEAN_mape))
    MEAN_mre=np.mean(np.array(MEAN_mre))
    return MEAN_rmse,MEAN_mae,MEAN_mape,MEAN_mre
def Sliding_impute_3D(Data_missing,Data_true,missing_mask,window):#Return:RMSE,MAE,MAPE,MRE
    from FancyImpute import Sliding_impute
    import pandas as pd
    import numpy as np
    from math import sqrt
    Sliding_rmse=[]
    Sliding_mae=[]
    Sliding_mape=[]
    Sliding_mre=[]
    Sliding_out=[]
    for i in range(Data_missing.shape[2]):
        a,b,c,d,e=Sliding_impute(Data_missing[:,:,i],Data_true[:,:,i],window)
        Sliding_rmse.append(a)
        Sliding_mae.append(b)
        Sliding_mape.append(c)
        Sliding_mre.append(d)
        Sliding_out.append(e)
    Sliding_rmse=np.mean(np.array(Sliding_rmse))
    Sliding_mae=np.mean(np.array(Sliding_mae))
    Sliding_mape=np.mean(np.array(Sliding_mape))
    Sliding_mre=np.mean(np.array(Sliding_mre))
    Sliding_out=np.array(Sliding_out).transpose(1,2,0)
    return Sliding_rmse,Sliding_mae,Sliding_mape,Sliding_mre,Sliding_out
