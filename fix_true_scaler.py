import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

data_path = os.path.join("data", "raw", "jena_climate_2009_2016.csv")
if not os.path.exists(data_path):
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    zip_path = os.path.join("data", "raw", "jena_climate.zip")
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(os.path.join("data", "raw"))
    os.remove(zip_path)
    print("Done downloading!")

df = pd.read_csv(data_path)

date_col = None
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        date_col = col
        break
if date_col is None:
    date_col = df.columns[0]

df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
df = df.set_index(date_col).sort_index()
df = df.select_dtypes(include=[np.number])

for col in ['wv (m/s)', 'max. wv (m/s)']:
    if col in df.columns:
        df.loc[df[col] < 0, col] = 0

df = df.resample('1h').mean().interpolate('linear').ffill().bfill()

# Fit exactly on training split
n = len(df)
train_end = int(n * 0.7)
train_df = df.iloc[:train_end]

scaler = StandardScaler()
scaler.fit(train_df)

feature_names = list(df.columns)

scaler_dict = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'feature_names': feature_names
}

with open('pwa/scaler.json', 'w') as f:
    json.dump(scaler_dict, f, indent=2)

print("SUCCESS: Scaler mathematically matched to original training set.")
print("Features:", len(feature_names))
print("Names:", list(feature_names))
