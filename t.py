import pandas as pd 
import preprocessing as pp 
import pca 
from timeseries_to_supervised import series_to_supervised
import predictions as pr
import numpy as np 

df = pp.load_folder('data')
df = pp.standardize_dataframe(df)
df_pca, ratios = pca.pca(df, 4)
out = series_to_supervised(df_pca.values, 96, 1, dropnan=True)
train_x, train_y = pr.get_test_data(0, out, 4, 96)

df_test = pp.load_folder('test_data')
df_test = pp.standardize_dataframe_pickle(df_test)
df_test_pca = pca.pca_pickle(df_test)
out = series_to_supervised(df_test_pca.values, 96, 1, dropnan=True)
test_x, test_y = pr.get_test_data(0, out, 4, 96)


train_x, train_y, test_x, test_y = pr.reframe_ann(train_x, train_y, test_x, test_y)

model = pr.define_ann()
pr.train_ann(model, train_x, train_y, test_x, test_y, 10)