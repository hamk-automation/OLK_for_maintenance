import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from scipy import stats
from os import listdir
import pickle 

def load_csv_file(file_name, folder_name):
    """
    loads csv file as a dataframe
    Arguments:
        file_name - name of a file to open in a folder_name
        folder_name - name of a folder, where the file is located
    Returns:
        dataframe - pandas dataframe 
    """
    file_path = folder_name + '/' + file_name
    dataframe = pd.read_csv(file_path, index_col=0, sep=';')
    return dataframe

def clean_dataframe(df, std_dev):
    """
    cleans dataframe from null values and outliers
    Arguments:
        df - dataframe to clean
        std_dev - determines the range in standard deviations to remove outliers
    Returns:
        clean_df - cleaned dataframe
    """
    clean_df = df.replace(to_replace='null', value=np.nan, inplace=False)
    clean_df = clean_df.dropna(axis=0, how='any', inplace=False)
    clean_df = clean_df[(np.abs(stats.zscore(clean_df)) < std_dev).all(axis=1)]
    return clean_df

def load_folder(folder_name):
    """
    loads all csv files into one dataframe from the selected folder
    Arguments:
        folder_name - name of a folder to grab csv files from
    Returns:
        dataframe - clean dataframe containing all csv files from the folder_name
    """
    files_list = listdir(folder_name)
    df_list = []

    #load all csv files as dataframes, clean them and append them to a list
    for file in files_list:
        df_list.append(clean_dataframe(load_csv_file(file, folder_name), 8))
    #merge all separate dataframes into a single one
    dataframe = pd.concat(df_list, axis=1, sort=False)
    #remove all rows with only zero values
    dataframe = dataframe.loc[(dataframe != 0).any(axis=1)]

    return dataframe


def standardize_dataframe(df):
    """
    A function to standardize data - zero mean, unit variance
    This function also saves a scaler into a file for a later use
    Arguments: 
        df - dataframe to standardize
    Returns:
        std_df - standardized dataframe
    """
    #get all features
    features = list(df.columns.values[1:])
    #separate out features
    df_sep = df.loc[:, features].values
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    #fit the scaler
    scaler.fit(df_sep)
    #save the scaler into a file <std_scaler.sav>
    pickle.dump(scaler, open("std_scaler.sav", 'wb'))
    #standardizing the features
    std_arr = scaler.transform(df_sep)
    #convert numpy array back into dataframe
    std_df = pd.DataFrame(data=std_arr, index=df.index.values, columns=features)
    return std_df