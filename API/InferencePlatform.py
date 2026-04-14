import os
import zipfile
import tempfile
import pickle
import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile
import Preprocessing

app = FastAPI()

LABELS = ["satisfied", "boring", "horrible", "calm", "funny", "valence", "arousal"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

models = {}

for label in LABELS:
    path = os.path.join(BASE_DIR, "Final", f"final_{label}.pkl")
    with open(path, "rb") as f:
        models[label] = pickle.load(f)

def process_file(path):

    df = pd.read_csv(path)
    df = df.dropna(axis=1,how='all')
    _, df_processed = Preprocessing.process_dataframe(df)

    colsX, _ = Preprocessing.getColumnNames(demo=False)

    df_features = pd.DataFrame(df_processed, columns=colsX)

    cols_to_remove = []

    cols_to_remove.extend([c for c in df_features.columns if "_alpha" in c])

    cols_to_remove.extend([c for c in df_features.columns if c.startswith("AF4")])

    cols_to_remove = list(set(cols_to_remove))

    df_filtered = df_features.drop(columns=cols_to_remove)
    
    return df_filtered

def predict_all_models(df):

    results = {}

    for label, model in models.items():
        try:
            check_is_fitted(model)
            print(f"{label} OK")
        except Exception:
            print(f"{label} NÃO está treinado ❌")
        preds = model.predict(df)

        # se precisar arredondar (caso regressão)
        preds = preds.round().astype(int)

        preds = np.clip(preds, 1, 10)
        
        results[label] = preds.tolist()

    return results

@app.post("/predict_subject")
async def predict_subject(file: UploadFile):

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:

        zip_path = os.path.join(tmpdir, file.filename)

        with open(zip_path, "wb") as f:
            f.write(await file.read())
        
        # extrair
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # procurar arquivos SXXG?AllChannels.csv
        for root, _, files in os.walk(tmpdir):
            for file_name in files:

                if "AllChannels" in file_name and file_name.endswith(".csv"):

                    game_id = file_name.split("AllChannels")[0]  # S28G1

                    file_path = os.path.join(root, file_name)

                    # preprocessar
                    df_processed = process_file(file_path)

                    # prever
                    preds = predict_all_models(df_processed)

                    results[game_id] = preds

    return results