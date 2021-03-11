import streamlit as st
import joblib
from category_encoders import CountEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

pipeline_n2o = Pipeline([
    ("one_hot", OneHotEncoder(cols=["scen", "species"])),
    #("freq_encode_species", CountEncoder(cols=["species"])),
    ("rf", RandomForestRegressor(
        verbose=True, 
        n_estimators=500,
        max_features=15,
        max_depth=15)),
])

pipeline_dC = Pipeline([
    ("one_hot", OneHotEncoder(cols=["scen", "species"])),
    #("freq_encode_species", CountEncoder(cols=["species"])),
    ("rf", RandomForestRegressor(
        verbose=True, 
        n_estimators=500,
        max_features=15,
        max_depth=15)),
])


crops = ['SPWH',
         'WIWH',
         'POTA',
         'FOCO',
         'SICO',
         'SUNFLOWER',
         'BEET',
         'RYE',
         'BEAN',
         'RAPE',
         'OATS',
         'SOYB',
         'SBAR']


boundaries = {
    'fert':    (0, 310, 200),
    'manu':    (0, 250, 50),
    'species': (None, None, None),
    'precip':  (100, 4000, 1000),
    'temp':    (5.0, 20.05, 9.0),
    'soc':     (0.5, 1.0, 6.0),
    'bd':      (1.0, 1.6, 1.2),
    'ph':      (4.0, 7.0, 6.0),
    'sand':    (5, 90, 10),
    'silt':    (5, 90, 70),
    'clay':    (5, 90, 20),
    'co2':     (380, 925, 420),
    'scen':    (None, None, None),
}



@st.cache(allow_output_mutation=True)
def load_model_n2o():
    model = joblib.load("model_n2o_v3.pkl.z")
    return model

@st.cache(allow_output_mutation=True)
def load_model_dC():
    model = joblib.load("model_dC_v3.pkl.z")
    return model


def val(v, x=None):
    if x == 'min':
        return boundaries[v][0]
    elif x == 'max':
        return boundaries[v][1]
    elif x == 'start':
        return boundaries[v][2]
    else:
        return boundaries[v]


def main():

    clay = 10

    model_n2o = load_model_n2o()
    model_dC = load_model_dC()

    st.write("# ResidueGas: GHG and Carbon Estimation")

    # info
    info = st.beta_expander("ℹ️ Info")

    with info:
        """
        This demo is using RandomForest models trained on simulation output 
        of the biogeochemical model LandscapeDNDC for arable soils of Europe
        within the [ResidueGas project](https://www.eragas.eu/en/eragas/Research-projects/ResidueGas.htm).
        For LandscapeDNDC details please click [here](https://ldndc.imk-ifu.kit.edu)

        Simulations conducted by [E. Haas](https://www.imk-ifu.kit.edu/Personen_Edwin.Haas.php), 
        app developed by [C. Werner](https://www.imk-ifu.kit.edu/Personen_Christian.Werner.php), 
        last update: 2020-03-04

        """
        # st.image("permutation_importance.png", caption="Fig.1: Permutation importance of features", width=600)

    st.write("## 👩‍💻 Input")


    b1 = st.beta_container()
    bhc = b1.beta_columns((4,4,1,4))
    bhc[0].write("Management")
    bhc[3].write("Weather")

    b1c = b1.beta_columns((4,4,1,4))
    b2c = b1.beta_columns((4,4,1,4))

    map_res_label = {
        'base': 'Baseline', 
        'one': 'Surface appl.',
        'two': 'Below-ground appl.',
        'zero': 'Exported',
        }

    map_crop_label = {
        'SPWH': 'Spring Wheat',
        'WIWH': 'Winter Wheat',
        'POTA': 'Potatoe',
        'FOCO': 'Food Corn',
        'SICO': 'Silage Corn',
        'SUNFLOWER': 'Sunflower',
        'BEET': 'Sugar Beet',
        'RYE':  'Rye',
        'BEAN': 'Legumes',
        'RAPE': 'Rapeseed',
        'OATS': 'Oat',
        'SOYB': 'Soybean',
        'SBAR': 'Summer Barley',
    }


    scen = b1c[0].selectbox('🚜 Residue use', ['base', 'one', 'two', 'zero'], 
        format_func=lambda x: map_res_label[x])
    species = b2c[0].selectbox('🌱 Crop', crops,
        format_func=lambda x: map_crop_label[x])
    
    # precip = b1c[2].slider(
    #     "🌧 Precip [mm]", 
    #     min_value=val('precip', 'min'), 
    #     max_value=val('precip', 'max'),
    #     value=val('precip', 'start'),
    #     )

    temp = b1c[3].slider(
        "🌡 Temp [°C]", 
        min_value=val('temp', 'min'), 
        max_value=val('temp', 'max'),
        value=val('temp', 'start'),
    )

    co2 = b2c[3].slider(
        "🎈 CO₂ [ppm]",
        min_value=val('co2', 'min'), 
        max_value=val('co2', 'max'),
        value=val('co2', 'start'),
    )

   
    fert = b1c[1].slider(
        "👨‍🌾 Fertilizer [kg N]", 
        min_value=val('fert', 'min'), 
        max_value=val('fert', 'max'),
        value=val('fert', 'start'),
        )

    manu = b2c[1].slider(
        "🐮 Manure [kg N]", 
        min_value=val('manu', 'min'), 
        max_value=val('manu', 'max'),
        value=val('manu', 'start'),
        )


    b3 = st.beta_container()
    b3.write("Soil")
    b3c = b3.beta_columns(3)

    soc = b3c[0].slider(
        "🍂 SOC [%]", 
        min_value=val('soc', 'min'), 
        max_value=val('soc', 'max'),
        value=val('soc', 'start'),
    )

    bd = b3c[1].slider(
        "💪 BD [g cm⁻³]", 
        min_value=val('bd', 'min'), 
        max_value=val('bd', 'max'),
        value=val('bd', 'start'),
    )

    ph = b3c[2].slider(
        "🧪 pH", 
        min_value=val('ph', 'min'), 
        max_value=val('ph', 'max'),
        value=val('ph', 'start'),
    )

    sand = b3c[0].slider(
        "🅂🄰 Sand [%]", 
        min_value=val('sand', 'min'), 
        max_value=(100 - clay), #val('sand', 'max'),
        value=val('sand', 'start'),
        )
    
    clay = b3c[1].slider(
        "🄲🄻 Clay [%]", 
        min_value=val('clay', 'min'), 
        max_value=(100 - sand), #( val('clay', 'max'),
        value=val('clay', 'start'),
        )

    silt = 100 - (sand + clay)
    b3c[2].markdown(
        f"<small>🅂🄸 Silt [%]: {silt}</small>", unsafe_allow_html=True
        ) 



    vnames = ['fert', 'manu', 'species', 'temp', 'soc', 'bd', 'ph', 'sand', 'silt', 'clay', 'co2', 'scen']
    vars = [fert, manu, species, temp, soc, bd, ph, sand, silt, clay, co2, scen]

    df = pd.DataFrame(dict(zip(vnames, [[x] for x in vars])))

    y_hat_n2o = model_n2o.predict(df)
    y_hat_dC = model_dC.predict(df)

    # predictions
    p = st.beta_container()
    p.write("## 🔮 Prediction")
    p1, p2 = p.beta_columns(2)

    p1.success(f"### N₂O: **{y_hat_n2o[0]:.1f}** [kg N ha⁻¹ yr⁻¹]\n#### ")

    p2.warning(f"### Δ SOC: **{y_hat_dC[0]:.1f}** [kg C ha⁻¹ yr⁻¹]\n#### ")
    


if __name__ == "__main__":
    main()
