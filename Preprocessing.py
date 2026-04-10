#Bibliotecas para arquivos
import re
from pathlib import Path
import json
from tqdm import tqdm

#Para leitura e escrita de dados
from collections import defaultdict
import pandas as pd
import numpy as np

#Normalização
from sklearn.preprocessing import StandardScaler

#Largura de Banda
from scipy.signal import welch

EPOCH_TIME = 2
SUBJECTS = [f"S{i:02d}" for i in range(1, 29)]
GAMES = [f"G{i}" for i in range(1, 5)]

#Leitura Inicial dos Dados
def read_files():
    files_csv = list(Path("GAMEEMO").rglob("*.csv"))
    files_path = [str(p) for p in files_csv]

    #Organizacao dos Dados
    files = defaultdict(lambda: {
        "pre": {},
        "raw": {}
    })

    for f in files_path:
        subject = re.search(r"S\d{2}", f).group()
        game = re.search(r"G\d", f).group()

        if "AllRawChannels" in f:
            files[subject]["raw"][game] = f
        else:
            files[subject]["pre"][game] = f

    files = dict(files)
    return files

#Utilização da Frequência Amostral

def get_fs(files):
    df = pd.read_csv(files['S01']['pre']['G1'])
    fs = df.shape[0]//300 +1
    return fs

#Limpeza

# 1.Re-Referência (CAR)
def CAR(df):
    df_car = df.sub(df.mean(axis=1), axis=0)
    return df_car

# 2. Baseline correction
def base_line(df):
    fs = get_fs()
    baseline_samples = EPOCH_TIME * fs
    baseline = df.iloc[:baseline_samples].mean()
    df_base = df - baseline
    return df_base

# 3. Normalização por canal
def normalize(df):
    df_norm = df.copy()
    for col in df_norm.columns:
        scaler = StandardScaler()
        df_norm[col] = scaler.fit_transform(df_norm[col].values.reshape(-1,1))
    return df_norm

# 4. Epoching
def epoching(df):
    fs = get_fs()
    window = EPOCH_TIME * fs
    epochs = []
    for start in range(0, len(df) - window, window):
        epoch = df.iloc[start:start+window]
        epochs.append(epoch)
    return epochs

# 5. Remoção de Artefatos
def remove_artifact(epochs):
    clean_epochs = []
    threshold = 3  # depois do z-score
    for epoch in epochs:
        if epoch.abs().max().max() < threshold:
            clean_epochs.append(epoch)
    return clean_epochs

# 6. Criação dos Nomes das Colunas
def getColumnNames(data, files):
    df = pd.read_csv(files['S01']['pre']['G1'])
    df.dropna(axis=1, how='all', inplace=True)
    channels = df.columns.tolist()
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    features_names = []
    labels_names = list(data['S01']['G1'].keys())

    for ch in channels:
        for band in bands:
            features_names.append(f"{ch}_{band}")
    
    features_names = features_names + ['male','female','age']

    return features_names,labels_names
#Criação de Features

# 1. Largura de Banda
def bandpower(signal, fmin, fmax):
    fs = get_fs()
    freqs, psd = welch(signal, fs)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    
    return np.trapz(psd[idx], freqs[idx])

# 2. Extração de Largura de Banda
def extract_features(epoch):
    features = []

    for col in epoch.columns:
        signal = epoch[col].values

        delta = bandpower(signal, 0.5, 4)
        theta = bandpower(signal, 4, 8)
        alpha = bandpower(signal, 8, 13)
        beta  = bandpower(signal, 13, 30)
        gamma = bandpower(signal, 30, 45)

        features.extend([delta, theta, alpha, beta, gamma])

    return features

# 3. Criação de Features com as Larguras de Banda
def feature_creation(clean_epochs):
    X = []

    for epoch in clean_epochs:
        feat = extract_features(epoch)
        X.append(feat)

    X = np.array(X)
    return X

# 4. Processa um dataframe com as features
def process_dataframe(df, subject, game):
    
    df = CAR(df)
    df = base_line(df)
    df = normalize(df)
    
    epochs = epoching(df)
    clean_epochs = remove_artifact(epochs)
    
    X = []
    tuples = []

    for i, epoch in enumerate(clean_epochs):
        feat = extract_features(epoch)
        tuples.append((subject, game, i+1))            
        X.append(feat)
    return tuples, X

# 5. Extrai as informações do txt

def readGAMEEMOdata():
    with open("gameemodatatxt.txt", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 6. Extrai as labels de um game de um subject

def extract_labels(data, subject, game):
    g = data[subject][game]
    
    return np.array([
        g["satisfied"],
        g["boring"],
        g["horrible"],
        g["calm"],
        g["funny"],
        g["valence"],
        g["arousal"]
    ])

# 7. Encoding do gênero
def encode_gender(g):
    return [0, 1] if g == 1 else [1, 0]

# 8. Normalização da idade
def normalize_age(age, min_age=20, max_age=27):
    return [(age - min_age) / (max_age - min_age)]


# 9. Processa todo o dataset, criando as features novas 
def build_dataset(files):
    
    data = readGAMEEMOdata()
    X = []
    y = []
    
    tuples = []
    
    colsX, colsy = getColumnNames(data, files)
    
    total_files = sum(
        len(files[s]['pre'])
        for s in files
    )

    with tqdm(total=total_files, desc="Total processing") as pbar:
        for subject in SUBJECTS:

            # 🔹 dados demográficos
            gender = data[subject]["gender"]
            age = data[subject]["Age"]
            
            gender_feat = encode_gender(gender)
            age_feat = normalize_age(age)
            demo_features = np.array(gender_feat + age_feat)

            for game in GAMES:
                
                pbar.set_postfix({
                    "Subject": subject,
                    "Game": game,
                })

                df = pd.read_csv(files[subject]['pre'][game])
                df.dropna(axis=1, how='all', inplace=True)
                
                idx_tuple, features_epochs = process_dataframe(df)
                tuples.append(idx_tuple)
                # 🔹 labels do jogo
                labels = extract_labels(data, subject, game)
                
                for feat in features_epochs:
                    
                    # X
                    feat = np.concatenate([feat, demo_features])
                    X.append(feat)
                    
                    # y (repete para cada epoch)
                    y.append(labels)
                
                pbar.update(1)
    
    index = pd.MultiIndex.from_tuples(
        tuples,
        names=["subject", "game", "epoch"]
    )

    df_features = pd.DataFrame(X, columns=colsX, index=index)
    df_labels = pd.DataFrame(y, columns=colsy, index=index)
    
    return df_features, df_labels

if __name__ == "__main__":
    files = read_files()
    
    df_features, df_labels = build_dataset(files)
    
    df_features.to_pickle('features_SML.pkl')
    df_labels.to_pickle('labels_SML.pkl')