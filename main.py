import preprocessing as pp 
import pandas as pd 
import numpy as np 
import pca 
import matplotlib.pyplot as plt 
import metrics
import clustering 

df = pp.load_folder('data')
df = pp.standardize_dataframe(df)
df_pca = pca.pca(df, 4)
print(metrics.calc_cop())
res = clustering.cluster(df_pca, 2)