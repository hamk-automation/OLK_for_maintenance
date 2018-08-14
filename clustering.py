import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import datetime as dt 
from sklearn.cluster import KMeans, FeatureAgglomeration
import preprocessing as pp 
import pickle
from tvtk.api import tvtk
from mayavi import mlab
from mayavi.scripts import mayavi2
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.modules.outline import Outline
from mayavi.modules.surface import Surface


def cluster(inputdf ,dataframe, n_cluster, plot_clustering = False):
    """
    The function performes kmeans clustering on the input dataframe.
    It also plots graphs to show clustering on a 2d figure(only first 2 dimensions are used for displaying, but all for clustering)

    Arguments:
        dataframe - input dataframe to performe clustering on 
        n_cluster - number of clusters to use, for this task I usually use 2: ok - doesn't look like ok
    Returns:
        classes_df - dataframe, containing class label for each datapoint and coresponding datapoint itself, so its length is equal to the number of datapoints in the input dataframe
    """
    # featureAgg = FeatureAgglomeration(n_clusters = n_cluster)
    # c = featureAgg.fit(dataframe)
    # c = featureAgg.fit_transform(dataframe)
    kmeans = KMeans(n_clusters=n_cluster)
    kmeans.fit(dataframe)
    c = kmeans.fit_predict(dataframe)
    classes = pd.DataFrame(data=c, columns=["class"], index=dataframe.index.values)
    classes_df = pd.concat([dataframe, classes], axis=1)
    classes_df = pd.concat([dataframe, classes], axis=1)
    #export kmeans model for a future use 
    # pickle.dump(featureAgg, open("kmeans_model.sav", 'wb'))
    if plot_clustering:
        #extract and transform time of a datapoint into a suitable format for matplotlib
        # time = inputdf.index.values
        # dts = [pd.to_datetime(dt.datetime.strptime(t[:18], '%Y-%m-%dT%H:%M:%S')) for t in time]
        # dt_time = [(d-min(dts)).days for d in dts]
        plt.switch_backend('Qt5Agg')
        fig, ax = plt.subplots(1,1)
        # ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(dataframe[dataframe.columns.values[0]], dataframe[dataframe.columns.values[1]], c=classes_df['class'])
        plt.draw()
        # plt.show()
        # print(classes_df['class'].get_values())
        # time = inputdf.index.values
        # dts = [pd.to_datetime(dt.datetime.strptime(t[:18], '%Y-%m-%dT%H:%M:%S')) for t in time]
        # dt_time = [(d-min(dts)).days for d in dts]
        # s = mlab.points3d(dataframe[dataframe.columns.values[0]], dataframe[dataframe.columns.values[1]], dt_time, 
        # scale_factor = 0.5)
        # mlab.show()
    return classes_df

def cluster_pickle(dataframe, plot_clustering = False):
    """
    Performes kmeans clustering on the input dataframe using previously saved model,
    
    Arguments:
        dataframe - input dataframe to perform clustering on 
    Returns:
        classes_df - dataframe, containing class label for each datapoint and coresponding datapoint itself, so its length is equal to the number of datapoints in the input dataframe
    """

    with open('kmeans_model.sav', 'rb') as pickle_file_kmeans_model:
        kmeans = pickle.load(pickle_file_kmeans_model)
    c = kmeans.predict(dataframe)
    classes = pd.DataFrame(data=c, columns=["class"], index=dataframe.index.values)
    classes_df = pd.concat([dataframe, classes], axis=1)
    #export kmeans model for a future use 
    pickle.dump(kmeans, open("kmeans_model.sav", 'wb'))
    if plot_clustering:
        plt.scatter(dataframe[dataframe.columns.values[0]], dataframe[dataframe.columns.values[1]], c=classes_df['class'])
        plt.draw()
    return classes_df