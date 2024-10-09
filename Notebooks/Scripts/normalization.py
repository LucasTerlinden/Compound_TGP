from sklearn.preprocessing import *
import copy
import pandas as pd
import numpy as np

## This script contains functions to normalize and denormalize the stochastic event sets

def scaler(df_toscale, scaler_type = MinMaxScaler()):
    '''
    Scaler function for the train dataset / inputs.

    # Inputs: train_datset = training dataset including inputs and targets
    #         scaler_type = scaler used for normalization, default = 'MinMaxScaler()'
    
    # Outputs: scaler_x, scaler_y = scalers after partial fit with train dataset for inputs and targets, respectively 
    '''
    scaler_x =  copy.deepcopy(scaler_type) 

    # Number of dimensions
    in_fet = df_toscale.shape[1]

    for idx in range(df_toscale.shape[0]):
        scaler_x.partial_fit(df_toscale.iloc[idx].values.reshape(in_fet, -1).T)
    return scaler_x

def normalize_dataset(df_toscale, scaler_x):
    '''
    # Function for normalizing every dataset. 

    # Inputs: df_toscale = dataset to be normalized, contains inputs
    #         scaler_x = scalers for inputs, created with the scaler function

    # Outputs: df_scaled = dataset after normalization 
    '''

    # get length of simulations, pixels and inputs and targets features
    in_fet = df_toscale.shape[1]
    
    scaled_list = []

    # normalize dataset by looping over all elements
    for idx in range(df_toscale.shape[0]):
        norm_x = scaler_x.transform(
            df_toscale.iloc[idx].values.reshape(in_fet, -1).T)
        scaled_list.append(norm_x[0])
    unitless_col = []
    for column in df_toscale.columns:
        unitless_col.append(column[:5]) 
    df_scaled = pd.DataFrame(scaled_list, columns = unitless_col)
    return df_scaled#

def quick_normalizer(np_to_scale):
    min = np_to_scale.min(axis = 0)
    max = np_to_scale.max(axis = 0)
    np_minmax = (np_to_scale - min) / (max - min)
    return np_minmax, min, max

def denormalize_dataset(df_scaled, scaler_x):
    '''
    # Function for normalizing every dataset. 

    # Inputs: df_scaled = dataset to be denormalized, contains inputs
    #         scaler_x = scalers for inputs, created with the scaler function

    # Outputs: df_descaled = dataset after denormalization 
    '''

    # get length of simulations, pixels and inputs and targets features
    in_fet = df_scaled.shape[1]
    
    descaled_list = []

    # normalize dataset by looping over all elements
    for idx in range(df_scaled.shape[0]):
        denorm_x = scaler_x.inverse_transform(
            df_scaled.iloc[idx].values.reshape(in_fet, -1).T)
        descaled_list.append(denorm_x[0])
    col = []
    for column in df_scaled.columns:
        col.append(column) 
    df_descaled = pd.DataFrame(descaled_list, columns = col)
    return df_descaled

def change_norm(np_scaled, min, max):
    total_cons = np_scaled * (max - min) + min
    max = total_cons.max(axis = 0)
    min = total_cons.min(axis = 0)
    np_rescaled = (total_cons - min) / (max - min)
    return np_rescaled, min, max

