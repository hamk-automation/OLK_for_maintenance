import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt  
import datetime as dt  
import pickle

def pca(input_dataframe, n):
    """
    Performes pca decomposition. It is used for dimensionality reduction. 
    The funcion also plots new principal components and saves pca parameters for a future use. 
    Arguments:
    input_dataframe - pandas dataframe with input data
    n - number of components used for PCA, determines the number of dimensions of output data
    Returns:
    pca_dataframe - dataframe containing principal components
    ratio - explained variance ratio
    """
    #extract and transform time of a datapoint into a suitable format for matplotlib
    time = input_dataframe.index.values
    dt_time = [dt.datetime.strptime(t[:18], '%Y-%m-%dT%H:%M:%S') for t in time]

    pca_model = PCA(n_components=n)
    pca_model.fit(input_dataframe.values)
    print("Explained variance ratio: ")
    print(pca_model.explained_variance_ratio_)

    #export pca_model for a future use
    pickle.dump(pca_model, open("pca_model.sav", 'wb'))
    #pca_model.transform returns numpy array
    components = pca_model.transform(input_dataframe.values)
    #transform numpy array to pandas dataframe
    pca_dataframe = pd.DataFrame(data = components, index=time, columns=['Principal Component ' + str(i) for i in range(1, n+1)])

    #plotting pca components
    plt.title('Principal component analysis')
    for plot in range(1, n+1):
        plt.subplot(n, 1, plot)
        plt.plot(dt_time, pca_dataframe['Principal Component ' + str(plot)])
        plt.ylabel('Principal Component ' + str(plot))
    plt.show()

    return pca_dataframe, pca_model.explained_variance_ratio_


def pca_pickle(input_dataframe):
    """
    Performes pca decomposition. It is used for dimensionality reduction. 
    The funcion uses previously saved pca model 
    Arguments:
    input_dataframe - pandas dataframe with input data
    Returns:
    pca_dataframe - dataframe containing principal components
    """
    with open('pca_model.sav', 'rb') as pickle_file_pca_model:
        pca_model = pickle.load(pickle_file_pca_model)

    #extract and transform time of a datapoint into a suitable format for matplotlib
    time = input_dataframe.index.values
    dt_time = [dt.datetime.strptime(t[:18], '%Y-%m-%dT%H:%M:%S') for t in time]
    #pca_model.transform returns numpy array
    components = pca_model.transform(input_dataframe.values)
    #transform numpy array to pandas dataframe
    pca_dataframe = pd.DataFrame(data = components, index=time, columns=['Principal Component ' + str(i) for i in range(1, 5)])

    #plotting pca components
    plt.title('Principal component analysis')
    for plot in range(1, 5):
        plt.subplot(4, 1, plot)
        plt.plot(dt_time, pca_dataframe['Principal Component ' + str(plot)])
        plt.ylabel('Principal Component ' + str(plot))
    plt.show()

    return pca_dataframe

