import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

df = pd.read_csv('data/raw/jena_climate_2009_2016.csv')
features = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", 
    "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", 
    "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"
]

# Ensure we have all 14 features exactly as the model expects
df = df[features]

scaler = StandardScaler()
scaler.fit(df)

scaler_dict = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'feature_names': features
}

with open('pwa/scaler.json', 'w') as f:
    json.dump(scaler_dict, f, indent=2)

print("Successfully regenerated scaler.json with 14 features!")
print("Mean array length:", len(scaler.mean_))
