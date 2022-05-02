from pathlib import Path

import pandas as pd
import xarray as xr
from dask_ml.model_selection import train_test_split

from mlem.api import save

from typing import Any, Dict

SOURCE = Path("data") / "raw" 
DEST = Path("data") / "processed" 

def prepare_soil_variables() -> xr.Dataset:
    soil_vars = "soc1,bd1,ph1,sand1,silt1,clay1".split(",")
    soil_vars_clean = [x.replace("1","") for x in soil_vars]

    soil = xr.open_dataset(SOURCE / "nc_rg_soil.nc")
    soil = soil[soil_vars]
    soil = soil.rename({k:v for k,v in zip(soil_vars, soil_vars_clean)})
    return soil

def prepare_mask() -> xr.Dataset:
    return xr.open_dataset(SOURCE / "nc_rg_mask.nc")[["ID"]]


def prepare_climate_variables(*, ref_dataset: xr.Dataset) -> xr.Dataset:
    clim = xr.open_mfdataset([
        SOURCE / "rg_arable_precipitation.nc", 
        SOURCE / "rg_arable_temp-yearly.nc"])[['prec[mm]', 'temp_above_canopy[oC]']]
    
    clim = clim.rename({'prec[mm]': 'precip', 'temp_above_canopy[oC]': 'temp'})
    clim = clim.reindex_like(ref_dataset)
    clim = clim.sel(year=slice(2000,2099))
    return clim

def prepare_co2() -> Dict[str, Dict[Any, float]]:
    source_files = [
        "rg_arable_airchem_rcp45_1950-2100.txt.gz",
        "rg_arable_airchem_rcp85_1950-2100.txt.gz",
        ]
    data = {}
    for fname, rcp in zip(source_files, ["rcp4p5", "rcp8p5"]):
        df = pd.read_csv(SOURCE / fname, skiprows=7, sep='\t')
        df.columns = ['datetime', 'no3', 'nh4', 'co2']
        df = df.astype({'datetime':'datetime64[ns]'})
        
        df_yearly = df.groupby(df.datetime.dt.year).first().drop(['datetime', 'no3', 'nh4'], axis=1).reset_index()
        data[rcp] = pd.Series(df_yearly.co2.values, index=df_yearly.datetime).to_dict()
    return data




def merge_data() -> pd.DataFrame:
    mask = prepare_mask()
    soil = prepare_soil_variables()
    climate  = prepare_climate_variables(ref_dataset=soil)
    co2 = prepare_co2()

    n2o = xr.open_dataset(SOURCE / "ensemble_n2o.nc")
    gwp = xr.open_dataset(SOURCE / "ensemble_gwp.nc")

    ds = xr.merge([mask, soil, climate, n2o, gwp])

    df = ds.to_dataframe().dropna().reset_index()
    df["co2"] = 0
    df.loc[df.rcp == "RCP4.5", "co2"] = df.year.map(co2["rcp4p5"])
    df.loc[df.rcp == "RCP8.5", "co2"] = df.year.map(co2["rcp8p5"])
    return df


def split_and_write_data(df: pd.DataFrame, *, seed:int = 42) -> None:
    df = df.drop(['ID', 'lat', 'lon', 'year'], axis= 1)

    X_train, X_test, y_train, y_test = train_test_split(df.drop(['n2o', 'gwp'], axis=1), df[['n2o', 'gwp']], shuffle=True, train_size=0.8, random_state=seed)
    y_train_n2o, y_train_gwp = y_train[['n2o']], y_train[['gwp']]

    #gzip_args = {'method': 'gzip', 'compresslevel': '1'}
    #params = {'compression': 'gzip'}
    save(X_train, str(DEST / "x_train.csv"), link=True, external=True) #, compression=gzip_args)
    save(X_test, str(DEST / "x_test.csv"), link=True, external=True) #, compression=gzip_args)
    save(y_train, str(DEST / "y_train.csv"), link=True, external=True)
    save(y_test, str(DEST / "y_test.csv"), link=True, external=True)
    save(y_train_n2o, str(DEST / "y_train_n2o.csv"), link=True, external=True)
    save(y_train_gwp, str(DEST / "y_train_gwp.csv"), link=True, external=True)


def main():
    df = merge_data()
    split_and_write_data(df, seed=42)

if __name__ == "__main__":
    main()