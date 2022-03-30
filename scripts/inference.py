import argparse

import joblib
from category_encoders import CountEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from dask.distributed import Client

from pathlib import Path

SOURCE = Path("models")

def load_model_n2o():
    model = joblib.load((SOURCE / "model_n2o.pkl.z").resolve())
    return model


def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    drop_vars = ["crop", "rcp"]
    df = df.drop(columns=drop_vars, axis=1, errors='ignore')
    return df.loc[:,~df.columns.str.match("Unnamed")]

def infer(model, df):

    pipeline_n2o = Pipeline([
        ("one_hot", OneHotEncoder(cols=["mana"])),
        #("freq_encode_species", CountEncoder(cols=["species"])),
        ("rf", RandomForestRegressor(
            verbose=True, 
            n_estimators=500,
            max_features=10,
            max_depth=15)),
    ])

    return model.predict(df)

def test(model):

    # load testset
    X_test = pd.read_csv( "data/processed/x_test.csv.gz" )
    X_test = drop_unnamed(X_test)
    y_test = pd.read_csv( "data/processed/y_test.csv.gz" )
    y_test = drop_unnamed(y_test)[['n2o']]

    with joblib.parallel_backend('dask'):
        res = model.score(X_test, y_test)
    return res

def main():
    c = Client()


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data",
        nargs="?",
        help="(optional) csv data file for prediction"
    )

    parser.add_argument(
        "--test",
        dest="test",
        action="store_true",
        default=False,
        help="evaluate testset"
    )

    args = parser.parse_args()

    if args.test:
        model = load_model_n2o()
        result = test(model)
        print(f"Test acc {result}")
    else:
        if args.data:
            df = pd.read_csv(args.data)
        else:
            data = {
                'precip': 1000,
                'temp': 12,
                'soc': 1.2,
                'bd': 1.3,
                'ph': 7.1,
                'sand': 30,
                'silt': 60,
                'clay': 10,
                'co2': 450,
                'mana': 'Buried',
            }
            df = pd.DataFrame(data, index=[0])

        model = load_model_n2o()
        result = infer(model, df)
        if len(result) == 1:
            print(f"Predicted N2O emission: {result[0]} kg N2O-N ha-1 yr-1")
        else:
            print("Predicted N2O emissions\n", result)


if __name__ == "__main__":
    main()