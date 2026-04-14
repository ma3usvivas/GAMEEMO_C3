#Bibliotecas para arquivos
from pathlib import Path

#Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#Normalização
from sklearn.preprocessing import StandardScaler

#Modelos
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn

#Métricas de precisão
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score
from scipy.stats import spearmanr

#Outros
from tqdm.auto import tqdm
import ast
from itertools import product
from math import prod

df_features = pd.read_pickle('features_SML.pkl')
df_labels = pd.read_pickle('labels_SML.pkl')

'''
data = np.load("dataset_cnn/dataset_eeg.npz")

X_signal = data["X_signal"]
X_demo   = data["X_demo"]
y        = data["y"]

class EEGDataset(Dataset):
    def __init__(self, X_signal, X_demo, y):
        self.X_signal = torch.tensor(X_signal, dtype=torch.float32)
        self.X_demo = torch.tensor(X_demo, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_signal[idx], self.X_demo[idx], self.y[idx]

class CNN1D(nn.Module):
    def __init__(self, in_channels, demo_dim, n_outputs, n_filters=32, kernel_size=3, hidden_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, n_filters, kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(n_filters, n_filters*2, kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1)
        )
        
        self.flatten = nn.Flatten()

        self.fc_signal = nn.Linear(n_filters * 2, hidden_dim)
        self.fc_demo = nn.Linear(demo_dim, 16)

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim + 16, n_outputs)
        )

    def forward(self, x_signal, x_demo):

        x = x_signal.permute(0, 2, 1)  # (B, C, T)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc_signal(x)

        d = self.fc_demo(x_demo)

        x = torch.cat([x, d], dim=1)
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, demo_dim, hidden_dim, num_layers, n_outputs):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc_demo = nn.Linear(demo_dim, 16)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs)
        )

    def forward(self, x_signal, x_demo):
        out, _ = self.lstm(x_signal)
        x = out[:, -1, :]  # último timestep

        d = self.fc_demo(x_demo)

        x = torch.cat([x, d], dim=1)
        return self.fc(x)

class GRUModel(nn.Module):
    def __init__(self, input_dim, demo_dim, hidden_dim, num_layers, n_outputs):
        super().__init__()

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc_demo = nn.Linear(demo_dim, 16)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs)
        )

    def forward(self, x_signal, x_demo):
        out, _ = self.gru(x_signal)
        x = out[:, -1, :]

        d = self.fc_demo(x_demo)

        x = torch.cat([x, d], dim=1)
        return self.fc(x)'''

SUBJECTS = [f"S{i:02d}" for i in range(1, 29)]

SEEDS = [13,44,71]

XGB_GRID = {
    "n_estimators": [100, 200, 300],      # 3
    "learning_rate": [0.01, 0.05, 0.1],   # 3
    "max_depth": [3, 6]                   # 2
}

RF_GRID = {
    "n_estimators": [100, 200],   # 2
    "max_depth": [None, 10]       # 2
}

KNN_GRID = {
    "n_neighbors": [3, 5, 7, 9]   # 4
}

'''CNN_GRID = {
    "n_filters": [16, 32],
    "kernel_size": [3, 5],
    "lr": [1e-3, 1e-4]
}

NN_MODELS = {
    "CNN": (CNN1D, CNN_GRID)
    #"LSTM": (LSTMModel, rnn_grid),
    #"GRU": (GRUModel, rnn_grid)
}'''

SML_MODELS = {
    "XGB": XGB_GRID,
    "RF": RF_GRID,
    "KNN": KNN_GRID
}

BASE_MODELS = {
    "XGB": XGBRegressor(objective='reg:squarederror', eval_metric = 'mae'),
    "RF": RandomForestRegressor(),
    "KNN": KNeighborsRegressor()
}

results_all = {}
metrics_all = {}

METRIC_LIST = ["mae", "accuracy", "qwk", "spearmanr"]

MODEL_TYPE = ['SML','NN']

def createLOSOSplit(features, labels, test_subject, target, seed, n_epochs, remove_demo = False):
    if remove_demo:
        features = features.drop(['male','female','age'], axis=1)

    X_test = features.xs(test_subject, level=0)
    y_test = labels.xs(test_subject, level=0)[target]

    X_train = features.drop(test_subject, level=0)
    y_train = labels.drop(test_subject, level=0)[target]

    def sample_df(X, y):
        df = X.copy()
        df[target] = y.values
        df = (
            df
            .groupby(level=[0,1], group_keys=False)  # Game
            .apply(lambda x: x.sample(n=min(len(x), n_epochs),random_state=seed))
        )
        
        y_sampled = df[target].values
        X_sampled = df.drop(columns=[target]).values
        
        return X_sampled, y_sampled

    X_train, y_train = sample_df(X_train, y_train)
    X_test, y_test   = sample_df(X_test, y_test)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def LOSOCV(model, features, labels, metric_list, seed=13, n_epochs=40, remove_demo=False, test_subjects = subjects):
    
    dfs = {}

    metrics = {}
    for target in tqdm(labels.columns, desc="Targets", leave=False):

        results_subject = {}
        metrics_target = {}
        
        subjects_bar = tqdm(np.unique(test_subjects), desc="Subjects", leave=False)

        for test_subject in subjects_bar:

            subjects_bar.set_postfix({
            "target": target,
            "subject": test_subject
            })
            
            results_subject[test_subject] = {}

            #Com ajuste para caso std não zere
            features = features.groupby(level=0).apply(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )

            X_train, y_train, X_test, y_test = createLOSOSplit(features, labels, test_subject, target, seed=seed, n_epochs=n_epochs, remove_demo=remove_demo)
            
            model_fold = clone(model)

            model_fold.fit(X_train, y_train)
            y_pred = np.clip(np.round(model_fold.predict(X_test)), 1,10)
            
            for metric in metric_list:
                if metric == "accuracy":
                    results_subject[test_subject]["accuracy"] = accuracy_score(y_test, y_pred)
                if metric == "mae":
                    results_subject[test_subject]["mae"] = mean_absolute_error(y_test, y_pred)
                if metric == "rmse":
                    results_subject[test_subject]["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
                if metric == "r2":
                    results_subject[test_subject]["r2"] = r2_score(y_test, y_pred)
                if metric == "qwk":
                    results_subject[test_subject]["qwk"] = cohen_kappa_score(y_test, np.round(y_pred), weights='quadratic')
                if metric == "spearmanr":
                    if np.std(y_pred) == 0:
                        results_subject[test_subject]["spearmanr"] = 0
                    elif np.std(y_test) == 0:
                        results_subject[test_subject]["spearmanr"] = 0
                    else:
                        results_subject[test_subject]["spearmanr"] = spearmanr(y_test, y_pred)[0]

        results_df = pd.DataFrame.from_dict(results_subject, orient='index')
        dfs[target] = results_df
        
        for metric in metric_list:
            metrics_target[metric] = {}
            values = [results_subject[s][metric] for s in results_subject]
            metrics_target[metric]['mean'] = np.mean(values)
            metrics_target[metric]['std'] = np.std(values)
        metrics[target] = metrics_target
    return dfs, metrics


# ---------------------- NN -------------------------
'''
def fitpredict(model, train_loader, test_loader, epochs=10, lr=1e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # 🔹 treino
    model.train()
    for _ in range(epochs):
        for xs, xd, y in train_loader:
            optimizer.zero_grad()
            pred = model(xs, xd).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    # 🔹 avaliação
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for xs, xd, y in test_loader:
            pred = np.round(model(xs, xd).squeeze(-1))
            pred = pred.view(-1)
            y = y.view(-1)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    return np.array(targets), np.array(preds)

def computeMetrics(y_true, y_pred, metric_list):
    
    results = {}

    for metric in metric_list:
        if metric == "mae":
            results["mae"] = mean_absolute_error(y_true, y_pred)

        elif metric == "rmse":
            results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))

        elif metric == "r2":
            results["r2"] = r2_score(y_true, y_pred)

        elif metric == "qwk":
            results["qwk"] = cohen_kappa_score(
                y_true, y_pred, weights="quadratic"
            )

        elif metric == "spearmanr":
            results["spearmanr"] = spearmanr(y_true, y_pred)[0]

    return results

def createLOSOSplitNN(X_signal, X_demo, y, test_subject, target_idx):
    
    meta_df = pd.read_csv('dataset_cnn/metadata.csv')
    subjects = meta_df["subject"].values
    
    train_idx = subjects != test_subject
    test_idx  = subjects == test_subject

    return ( X_signal[train_idx], X_demo[train_idx], y[train_idx][:, target_idx],
        X_signal[test_idx], X_demo[test_idx], y[test_idx][:, target_idx],
    )

def LOSOCV_NN(
    model_class,
    model_params,
    X_signal,
    X_demo,
    y,
    metric_list,
    epochs_num=10,
    batch_size=32
):
    subjects = [f"S{i:02d}" for i in range(1, 29)]

    dfs = {}
    metrics = {}
    target_names = pd.read_csv("dataset_cnn/metadata.csv").columns.tolist()[3:]

    for target_idx, target in enumerate(tqdm(target_names, desc="Targets", leave=False)):

        results_subject = {}
        metrics_target = {}
        
        subjects_bar = tqdm(np.unique(subjects), desc="Subjects", leave=False)

        for test_subject in subjects_bar:
            
            subjects_bar.set_postfix({
            "target": target,
            "subject": test_subject
            })

            X_tr_s, X_tr_d, y_tr, X_te_s, X_te_d, y_te = createLOSOSplitNN(
                X_signal, X_demo, y, test_subject, target_idx
            )

            train_ds = EEGDataset(X_tr_s, X_tr_d, y_tr)
            test_ds  = EEGDataset(X_te_s, X_te_d, y_te)
            
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader  = DataLoader(test_ds, batch_size=batch_size)

            model = model_class(**model_params)
            
            y_true, y_pred = fitpredict(
                model, train_loader, test_loader, epochs=epochs_num
            )

            results_subject[test_subject] = computeMetrics(
                y_true, y_pred, metric_list
            )

        results_df = pd.DataFrame.from_dict(results_subject, orient="index")
        dfs[target] = results_df

        for metric in metric_list:
            values = results_df[metric].values
            metrics_target[metric] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }

        metrics[target] = metrics_target

    return dfs, metrics
'''
# --------------------- Save ------------------------
def format_model_name(model_str):
    # Separar nome do modelo e parâmetros
    model, params_str = model_str.split("_", 1)
    
    # Converter string de dict para dict real
    params = ast.literal_eval(params_str)
    
    # Mapeamento dos nomes
    rename_map = {
        "n_filters":"filters",
        "kernel_size":"krnlsize",
        "hidden_dim":"hidden",
        "num_layers":"layers",
        "lr":"lr",
        "learning_rate": "lr",
        "n_neighbors": "neigh",
        "max_depth": "depth",
        "n_estimators": "ests"
    }
    
    parts = [model]
    
    for key, value in params.items():
        new_key = rename_map.get(key, key)
        
        # substituir ponto por underscore (ex: 0.01 -> 0_01)
        if isinstance(value, float):
            value = str(value).replace(".", "_")
        
        parts.append(f"{new_key}_{value}")
    
    return "_".join(parts)

if __name__ == "__main__":
    
        
    total_grid = sum(len(list(ParameterGrid(grid))) for grid in SML_MODELS.values())

    #Simple ML
    with tqdm(total=total_grid, desc="Total processing") as pbar:
        for model_name in BASE_MODELS:
        
        base_model = BASE_MODELS[model_name]

        pbar.set_description(f"modelo atual {model_name}") 

        for params in ParameterGrid(grids[model_name]):

            for seed in seeds:

                pbar.set_postfix({
                    "model": model_name,
                    "seed": seed
                })

                set_seed(seed)

                model = clone(base_model)
                model.set_params(**params)

                dfs, metrics = LOSOCV(
                    model,
                    df_features,
                    df_labels,
                    metric_list,
                    seed=seed
                )

                key = f"{model_name}_{params}_seed{seed}"

                results_all[key] = dfs
                metrics_all[key] = metrics

            pbar.update(1)
    output_dir = Path("Models")
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in results_all.keys():
        model_name = format_model_name(model)
        file_path = output_dir / f"{file_name}.xlsx"
        with pd.ExcelWriter(file_path) as writer:
            for target in results_all[model].keys():
                results_all[model][target].to_excel(writer, sheet_name=str(target))
        '''elif model_type == 'NN':
            def expand_grid(grid):
                keys = list(grid.keys())
                values = list(grid.values())
                
                for combo in product(*values):
                    yield dict(zip(keys, combo))
            total_grid = sum(
                prod(len(v) for v in grid.values())
                for _, (_, grid) in NN_MODELS.items()
            )
            with tqdm(total=total_grid, desc="Total processing") as pbar:
            
                for model_name, (model_class, grid) in NN_MODELS.items():

                    for params in expand_grid(grid):

                        pbar.set_postfix({
                        "model": model_name,
                        "params": str(params)
                        })
                        
                        model_params = params.copy()
                        lr = model_params.pop("lr")

                        if model_name == "CNN":
                            model_params.update({
                                "in_channels": X_signal.shape[2],
                                "demo_dim": X_demo.shape[1],
                                "n_outputs": 1
                            })
                        else:
                            model_params.update({
                                "input_dim": X_signal.shape[2],
                                "demo_dim": X_demo.shape[1],
                                "n_outputs": 1
                            })

                        dfs, metrics = LOSOCV_NN(
                            model_class=model_class,
                            model_params=model_params,
                            X_signal=X_signal,
                            X_demo=X_demo,
                            y=y, 
                            metric_list=METRIC_LIST,
                            epochs_num=10,
                            batch_size=32
                        )

                        key = f"{model_name}_{params}"
                        results_all[key] = dfs
                        pbar.update(1)
            output_dir = Path("Models")
            output_dir.mkdir(parents=True, exist_ok=True)

            for model_name, (model_class, grid) in NN_MODELS.items():
                for params in expand_grid(grid):
                    key = f"{model_name}_{params}"
                    file_name = format_model_name(key)
                    file_path = output_dir / f"{file_name}.xlsx"

                    with pd.ExcelWriter(file_path) as writer:
                        for target in results_all[key].keys():
                            results_all[key][target].to_excel(
                                writer, 
                                sheet_name=str(target)
                            )'''