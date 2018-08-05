import preprocessing as pp 
import pandas as pd 
import numpy as np 
import pca 
import matplotlib.pyplot as plt 
import metrics
import clustering 
import visualizations as vs 
import predictions as pr 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from timeseries_to_supervised import series_to_supervised
import pickle 

def main():
    df = pp.load_folder('data')
    df = pp.standardize_dataframe(df)
    df_pca, ratios = pca.pca(df, 3)
    cls_df = clustering.cluster(df_pca, 5, plot_clustering=True)
    df_main = pd.concat([df, df_pca], axis=1)
    df_corr = df_main.corr()
    df_corr.to_csv('corr.csv', sep=";")
    cop = metrics.calc_test_cop()
    vs.clustering_plt(cls_df, cop, 4)
    out = series_to_supervised(df_pca.values, 96, 1, dropnan=True)
    train_y = out[out.columns[-4:]]
    train_x = out[out.columns[:-4]]
    regr = RandomForestRegressor(max_depth=None, n_estimators=25)
    regr_ada = AdaBoostRegressor(regr, n_estimators=20, loss='square')
    regr_mult = MultiOutputRegressor(regr_ada, n_jobs=-1)
    """
    regr_mult.fit(train_x, train_y)
    #regr = RandomForestRegressor(max_depth=None, n_estimators=400, criterion='mse', random_state=0)
    #regr = ExtraTreesRegressor(n_estimators=400)
    #regr.fit(train_x, train_y)
    #regr = pr.open_model('random_forrest.sav')
    #regr_mult = pr.open_model('regr_mult.sav')
    #regr = pr.open_model('extra_trees.sav')
    pickle.dump(regr_mult, open("regr_mult_2.sav", 'wb'))
        


    df_test = pp.load_folder('test_data')
    df_test = pp.standardize_dataframe_pickle(df_test)
    df_test_pca = pca.pca_pickle(df_test)
    out = series_to_supervised(df_test_pca.values, 96, 1, dropnan=True)

    test_x, test_y = pr.get_test_data(0, out, 4, 96)
    pred, test_y = pr.test_frame(test_x, test_y, regr_mult, 96, 96)
    r2list, r2 = pr.calc_r2(pred, test_y, ratios)
    cls_test = clustering.cluster_pickle(pred)
    test_y = clustering.cluster_pickle(test_y)
    cop_test = metrics.calc_test_cop()
    cls_test.to_csv("pred.csv", sep=";")
    print("R2 score: " + str(r2))
    print("R2 score unweighted")
    print(r2list)

    #-----------------#

    pr.plot_predictions(cls_test, test_y, cop_test[4:12])
    """


if __name__ == '__main__':
    main()