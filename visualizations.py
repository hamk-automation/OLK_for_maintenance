import preprocessing as pp
import pandas as pd
import numpy as np
import pca
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import metrics
import clustering
import datetime as dt


def clustering_plt(cls_df, cop_df, n_pca):
    """
    plots figure with all principal components, their assigned classes and COP of the heat pump
    Arguments:
        cls_df - pandas dataframe obtained from cluster method
        cop_df - pandas dataframe with COP values
        n_pca - number of principal components used
    Returns:
        None
    """
    # plt.close('all')

    time_cls = cls_df.index.values
    dt_time_cls = [dt.datetime.strptime(
        t[:18], '%Y-%m-%dT%H:%M:%S') for t in time_cls]

    time_cop = cop_df.index.values
    dt_time_cop = [dt.datetime.strptime(
        t[:18], '%Y-%m-%dT%H:%M:%S') for t in time_cop]

    fig, axarr = plt.subplots(n_pca+2,1)
    plt.suptitle('Principal component analysis');
    for plot in range(1, n_pca):
        # plt.subplot(n_pca + 2, 1, plot)
        axarr[plot-1].plot(dt_time_cls, cls_df['Principal Component ' + str(plot)])
        axarr[plot-1].set_ylabel('PC' + str(plot))
        axarr[plot-1].fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M')

    axarr[n_pca-1].plot(dt_time_cop, cop_df.values)
    axarr[n_pca-1].fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M')
    axarr[n_pca-1].set_ylabel('COP')

    axarr[n_pca].plot(dt_time_cls, cls_df['class'])
    axarr[n_pca].fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M')
    axarr[n_pca].set_ylabel('Class')
    # print(dt_time_cls)
    # fig = plt.figure()   
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    plt.show()
