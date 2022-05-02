from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
#from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from category_encoders import CountEncoder, OneHotEncoder

from dask_ml.model_selection import train_test_split
import numpy as np
import joblib
import xarray as xr
#import hvplot.xarray
from pathlib import Path
import pandas as pd
from sklearn.inspection import permutation_importance

import dask.dataframe as dd

import random

from dask.distributed import Client, LocalCluster

import time
import json

from typing import Any, Dict

import os
from dotenv import load_dotenv

#import mlflow
from mlem.api import load, save

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/cwerner/residuegas.mlflow")

SOURCE = Path("data") / "processed" 
DEST = Path("models")

def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    drop_vars = ["crop", "rcp"]
    df = df.drop(columns=drop_vars, axis=1, errors='ignore')
    return df.loc[:,~df.columns.str.match("Unnamed")]


if __name__ == '__main__':
    random.seed(42)

    c = Client()
    # mlflow.start_run()
    # mlflow.sklearn.autolog()

    X_train = load(str(SOURCE / "x_train.csv"))
    X_test = load(str(SOURCE / "x_test.csv"))
    y_train = load(str(SOURCE / "y_train.csv"))
    y_test = load(str(SOURCE / "y_test.csv"))
    y_train_n2o = load(str(SOURCE / "y_train_n2o.csv"))
    y_train_gwp = load(str(SOURCE / "y_train_gwp.csv"))
    
    # assert that columns are in identical order
    # if index in files
    if (X_train.columns[0] == 'Unnamed: 0') and (y_train.columns[0] == 'Unnamed: 0'):
        assert np.array_equal(X_train.iloc[0:5,0], y_train.iloc[0:5,0]), "Order of train and test files does not match!"
        
    
    X_train = drop_unnamed(X_train)
    X_test = drop_unnamed(X_test) 
    y_train = drop_unnamed(y_train) 
    y_test = drop_unnamed(y_test) 
    y_train_n2o = drop_unnamed(y_train_n2o)
    y_train_gwp = drop_unnamed(y_train_gwp)
    
    
    frac = 5 # 1/10th
    sample_row_ids = random.sample(range(len(X_train)), len(X_train)//frac)
    X_train = X_train.iloc[sample_row_ids, :]
    y_train = y_train.iloc[sample_row_ids, :]
    y_train_n2o = y_train_n2o.iloc[sample_row_ids, :]
    y_train_gwp = y_train_gwp.iloc[sample_row_ids, :]
    
    print(X_train.head())
    
    # make pipeline from pre-processing steps and final model
    pipeline_n2o = Pipeline([
        ("one_hot", OneHotEncoder(cols=["mana"])),
        #("freq_encode_species", CountEncoder(cols=["species"])),
        ("rf", RandomForestRegressor(
            verbose=True, 
            n_estimators=500,
            max_features=10,
            max_depth=15)
            ),
        ])

    with joblib.parallel_backend("dask"):

        start = time.time()
        pipeline_n2o.fit(X_train, y_train_n2o)
        end = time.time()
        duration = end - start

        score = pipeline_n2o.score(X_test, y_test['n2o'])
        print(f"SCORE N2O: {score:.2f}")
        
    # save models
    ##joblib.dump(pipeline_n2o, (DEST / "model_n2o.pkl.z").resolve())

    # save(
    #     pipeline_n2o.named_steps['one_hot'],
    #     "rf_onehot_n2o",
    #     tmp_sample_data=X_train,
    #     tags=["random-forest", "regressor", "preprocessing"],
    #     description="Random Forest Regressor One-Hot Encoding",
    # )

    save(
        pipeline_n2o.named_steps['rf'],
        str(DEST / "rf_n2o"),
        tmp_sample_data=pipeline_n2o.named_steps['one_hot'].transform(X_train),
        tags=["random-forest", "regressor"],
        description="Random Forest Regressor N2O",
        link=True, 
        external=True
    )


    # save metrics
    with open('metrics.json', 'w') as fp:

        metrics_data: Dict[str, Any] = {
            "train": {"accuracy": score},
            "duration": duration
        }
        json.dump(metrics_data, fp, indent=4)


    exit()

    # make pipeline from pre-processing steps and final model
    pipeline_gwp = Pipeline([
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
        score = pipeline_dC.score(X_test, y_test['gwp'])
        print(f"SCORE DC: {score:.2f}")

    joblib.dump(pipeline_gwp, "model_gwp.pkl.z")

