import pandas as pd 
import numpy as np 
import preprocessing as pp 

def calc_cop():
    """
    This function expects to find file name COP_in.csv in metrics_data folder.
    The function calculates COP of the heat pump.
    Arguments:
        None
    Returns:
        cop_df - dataframe with COP of the heat pump
    """
    df = pp.load_csv_file('COP_in.csv', 'metrics_data') 
    df = pp.clean_dataframe(df, 5)

    df_cop = df['LP01LM01_QQ'] / df['SJ01_SM01']
    df_cop = df_cop.replace(to_replace=np.nan, value = 0, inplace=False)
    
    return df_cop

def calc_test_cop():
    """
    This function expects to find file name COP_test_in.csv in metrics_data folder.
    The function calculates COP of the heat pump.
    Arguments:
        None
    Returns:
        cop_df - dataframe with COP of the heat pump
    """
    df = pp.load_csv_file('COP_test_in.csv', 'metrics_data') 
    df = pp.clean_dataframe(df, 5)

    df_cop = df['LP01LM01_QQ'] / df['SJ01_SM01']
    df_cop = df_cop.replace(to_replace=np.nan, value = 0, inplace=False)
    
    return df_cop