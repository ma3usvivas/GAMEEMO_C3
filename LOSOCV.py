import pandas as pd
import numpy as np

#Normalização
from sklearn.preprocessing import StandardScaler

#Modelos
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

#Métricas de precisão
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score
from scipy.stats import spearmanr

#Outros
from tqdm.auto import tqdm
import ast

df_features = pd.read_pickle('features_SML.pkl')
df_labels = pd.read_pickle('labels_SML.pkl')

SUBJECTS = [f"S{i:02d}" for i in range(1, 29)]

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

GRIDS = {
    "XGB": XGB_GRID,
    "RF": RF_GRID,
    "KNN": KNN_GRID
}

BASE_MODELS = {
    "XGB": XGBRegressor(objective='reg:squarederror', eval_metric = 'mae'),
    "RF": RandomForestRegressor(),
    "KNN": KNeighborsRegressor()
}

PARAM_MAP = {
    "learning_rate": "lr",
    "n_neighbors": "neigh",   # corrigi o typo aqui
    "max_depth": "depth",
    "n_estimators": "est"
}

results_all = {}
metrics_all = {}

METRIC_LIST = ["mae", "rmse", "r2", "qwk", "spearmanr"]

def createLOSOSplit(features, labels, test_subject, target, remove_demo = False):
    if remove_demo:
        features.drop(['male','female','age'], axis=1, inplace=True)

    X_test = features.xs(test_subject, level=0)
    y_test = labels.xs(test_subject, level=0)[target]

    X_train = features.drop(test_subject, level=0)
    y_train = labels.drop(test_subject, level=0)[target]

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def LOSOCV(model, features, labels, metric_list):
    
    dfs = {}

    metrics = {}
    
    for target in tqdm(labels.columns, desc="Targets", leave=False):

        results_subject = {}
        metrics_target = {}

        for test_subject in tqdm(SUBJECTS, desc=f"Subject {test_subject} - {target}", leave=False):
            results_subject[test_subject] = {}

            X_train, y_train, X_test, y_test = createLOSOSplit(features, labels, test_subject, target)

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            model_fold = clone(model)

            model_fold.fit(X_train, y_train)
            y_pred = np.round(model_fold.predict(X_test))
            
            for metric in metric_list:
                if metric == "mae":
                    results_subject[test_subject]["mae"] = mean_absolute_error(y_test, y_pred)
                if metric == "rmse":
                    results_subject[test_subject]["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
                if metric == "r2":
                    results_subject[test_subject]["r2"] = r2_score(y_test, y_pred)
                if metric == "qwk":
                    results_subject[test_subject]["qwk"] = cohen_kappa_score(y_test, np.round(y_pred), weights='quadratic')
                if metric == "spearmanr":
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


def format_model_name(model_str):
    # Separar nome do modelo e parâmetros
    model, params_str = model_str.split("_", 1)
    
    # Converter string de dict para dict real
    params = ast.literal_eval(params_str)
    
    # Mapeamento dos nomes
    rename_map = {
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

    total_grid = sum(len(list(ParameterGrid(grid))) for grid in GRIDS.values())

    #Simple ML
    with tqdm(total=total_grid, desc="Total processing") as pbar:
        for model_name in BASE_MODELS:
            
            base_model = BASE_MODELS[model_name]

            pbar.set_description(f"modelo atual {model_name}") 

            for params in ParameterGrid(GRIDS[model_name]):

                model = clone(base_model)  # ✅ SEMPRE aqui
                model.set_params(**params)
                
                dfs, metrics = LOSOCV(model, df_features, df_labels, METRIC_LIST, SUBJECTS)
                
                key = f"{model_name}_{params}"
                results_all[key] = dfs
                metrics_all[key] = metrics

                pbar.update(1)

    for model in results_all.keys():
        model_name = format_model_name(model)
        with pd.ExcelWriter(f"{model_name}.xlsx") as writer:
            for target in results_all[model].keys():
                results_all[model][target].to_excel(writer, sheet_name=str(target))