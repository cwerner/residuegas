from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
#from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from category_encoders import CountEncoder, OneHotEncoder

from dask_ml.model_selection import train_test_split
import numpy as np
import joblib
import xarray as xr
import hvplot.xarray
from pathlib import Path
import pandas as pd
from sklearn.inspection import permutation_importance

import dask.dataframe as dd

import random

from dask.distributed import Client, LocalCluster

if __name__ == '__main__':
    c = Client()

    X_train = pd.read_csv('x_train.csv')
    X_test = pd.read_csv('x_test.csv')
    y_train = pd.read_csv('y_train.csv')
    y_test = pd.read_csv('y_test.csv')
    y_train_n2o = pd.read_csv('y_train_n2o.csv')
    y_train_dC = pd.read_csv('y_train_dC.csv')
    
    
    # assert that columns are in identical order
    # if index in files
    if (X_train.columns[0] == 'Unnamed: 0') and (y_train.columns[0] == 'Unnamed: 0'):
        assert np.array_equal(X_train.iloc[0:5,0], y_train.iloc[0:5,0]), "Order of train and test files does not match!"
        
    
    def drop_unnamed(df):
        return df.loc[:,~df.columns.str.match("Unnamed")]
    
    X_train = drop_unnamed(X_train)
    X_test = drop_unnamed(X_test) 
    y_train = drop_unnamed(y_train) 
    y_test = drop_unnamed(y_test) 
    y_train_n2o = drop_unnamed(y_train_n2o)
    y_train_dC = drop_unnamed(y_train_dC)
    
    
    frac = 5 # 1/10th
    ss = random.sample(range(len(X_train)), len(X_train)//frac)
    X_train = X_train.iloc[ss]
    y_train = y_train.iloc[ss]
    y_train_n2o = y_train_n2o.iloc[ss]
    y_train_dC = y_train_dC.iloc[ss]
    

    
    # make pipeline from pre-processing steps and final model
    pipeline_n2o = Pipeline([
        ("one_hot", OneHotEncoder(cols=["scen", "species"])),
        #("freq_encode_species", CountEncoder(cols=["species"])),
        ("rf", RandomForestRegressor(
            verbose=True, 
            n_estimators=500,
            max_features=15,
            max_depth=15)),
    ])

    with joblib.parallel_backend("dask"):
        pipeline_n2o.fit(X_train, y_train_n2o)
        score = pipeline_n2o.score(X_test, y_test['aN2O'])
        print(f"SCORE N2O: {score:.2f}")
        
    # save models
    joblib.dump(pipeline_n2o, "model_n2o_v3.pkl.z")

    # make pipeline from pre-processing steps and final model
    pipeline_dC = Pipeline([
        ("one_hot", OneHotEncoder(cols=["scen", "species"])),
        #("freq_encode_species", CountEncoder(cols=["species"])),
        ("rf", RandomForestRegressor(
            verbose=True, 
            n_estimators=500,
            max_features=15,
            max_depth=15)),
    ])

    with joblib.parallel_backend("dask"):
        pipeline_dC.fit(X_train, y_train_dC)
        score = pipeline_dC.score(X_test, y_test['adC'])
        print(f"SCORE DC: {score:.2f}")

    joblib.dump(pipeline_dC, "model_dC_v3.pkl.z")

