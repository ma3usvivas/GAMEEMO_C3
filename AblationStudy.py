#Bibliotecas para arquivos
from pathlib import Path

#Dataset
import pandas as pd
import numpy as np

#Gráficos
import matplotlib.pyplot as plt

#Models
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn

#LOSOCV
from LOSOCV import LOSOCV, LOSOCV_NN
#Outros
from tqdm.auto import tqdm


def loadResults(folder_path="."):
    results_all = {}

    for file in Path(folder_path).glob("*.xlsx"):
        print(file)
        model_name = file.stem  # nome do arquivo sem .xlsx
        if file.name.startswith("~$"):
            continue
        # Ler todas as sheets
        sheets = pd.read_excel(file,index_col=0, sheet_name=None)
        
        # Cada sheet vira um target
        results_all[model_name] = sheets
    return results_all

def getMetrics(results):

    metrics = {}
    
    for model in results.keys():
        metrics[model]={}

        for target in results[model].keys():

            metrics[model][target] = {}

            for metric in results[model][target].columns:
                metrics[model][target][metric] = {}
                values = results[model][target][metric].values
                metrics[model][target][metric]['mean'] = np.mean(values)
                metrics[model][target][metric]['std'] = np.std(values)

    return metrics

def selectBestModel(model,metric):
    best_per_target = {}

    for key, metrics in METRICS_LOADED.items():  # cada modelo + params
    
        for target, metric_dict in metrics.items():
            
            qwk_mean = metric_dict[metric]["mean"]
            
            if target not in best_per_target:
                best_per_target[target] = {
                    "model": key,
                    metric: qwk_mean
                }
            else:
                if qwk_mean > best_per_target[target][metric]:
                    best_per_target[target] = {
                        "model": key,
                        metric: qwk_mean
                    }

    return best_per_target

def get_qwk_per_target(metrics):
    return {target: metrics[target]["qwk"]["mean"] for target in metrics}

def get_global_qwk(metrics):
    qwk_values = []
    
    for target in metrics:
        qwk_values.append(metrics[target]["qwk"]["mean"])
    
    return np.mean(qwk_values)

def computeAblationImportance(ablation_results, seeds, models):
    
    importance_all = {}
    
    for model_name, _ in models:
        
        importance_all[model_name] = {}
        
        # pegar ablations disponíveis
        ablations = ablation_results[seeds[0]][model_name].keys()
        
        for ablation in ablations:
            
            if ablation == "baseline":
                continue
            
            importance_all[model_name][ablation] = {}
            
            # incluir global como target extra
            all_targets = targets_list + ["global"]
            
            for target in all_targets:
                
                values = []
                
                for seed in seeds:
                    
                    metrics_base = ablation_results[seed][model_name]["baseline"]
                    metrics_abla = ablation_results[seed][model_name][ablation]
                    
                    # -------------------------
                    # target específico
                    # -------------------------
                    if target != "global":
                        base_qwk = metrics_base[target]["qwk"]["mean"]
                        abla_qwk = metrics_abla[target]["qwk"]["mean"]
                    
                    # -------------------------
                    # global
                    # -------------------------
                    else:
                        base_qwk = get_global_qwk(metrics_base)
                        abla_qwk = get_global_qwk(metrics_abla)
                    
                    values.append(base_qwk - abla_qwk)
                
                importance_all[model_name][ablation][target] = {
                    "mean": np.mean(values),
                    "std": np.std(values)
                }
    
    return importance_all
    
def plotAblationTarget(mean_df, std_df, model_name, target, save=True):
    
    df_plot = mean_df.sort_values(by=target, ascending=True)
    std_plot = std_df.loc[df_plot.index]

    plt.figure()
    
    plt.barh(
        df_plot.index,
        df_plot[target],
        xerr=std_plot[target],   # 🔥 intervalo de confiança (1 std)
        color=target_colors.get(target, "black"),
        capsize=3               # 🔥 deixa visual melhor
    )
    
    plt.title(f"{model_name}\nFeature Importance (QWK Ablation) - {target.capitalize()}")
    plt.xlabel("Δ QWK (baseline - ablation)")
    plt.ylabel("Features")
    plt.axvline(0)
    plt.tight_layout()
    
    if save:
        plt.savefig(f"Ablation/{model_name}_{target}.png", dpi=300)
    
    plt.show()
    plt.close()

def cleanFeatureName(name):
    if name.startswith("remove_"):
        return name.replace("remove_", "").upper()
    elif name.startswith("no_"):
        return name.replace("no_", "").capitalize()
    else:
        return name.upper()

FOLDER_PATH = 'Models'
RESULTS_LOADED = loadResults('Models')
METRICS_LOADED = getMetrics(RESULTS_LOADED)
METRIC = 'qwk'

best_per_target = selectBestModel(METRIC)

for key in best_per_target:
    print(key,' - ',best_per_target[key]['model'],' - ',best_per_target[key]['qwk'])

#Best Model: KNN, n_neighbors = 3

df_features = pd.read_pickle('features_SML.pkl')
df_labels = pd.read_pickle('labels_SML.pkl')
print(df_features.columns)

ALL_FEATURES = list(df_features.columns)

DEMO_FEATURES = ["male", "female", "age"]
EEG_FEATURES = [f for f in ALL_FEATURES if f not in DEMO_FEATURES]

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
CHANNELS = list(set(f.split("_")[0] for f in EEG_FEATURES))
CHANNELS = sorted(CHANNELS)

ablation_sets = {}

# baseline
ablation_sets["baseline"] = ALL_FEATURES

# demográfico
ablation_sets["no_demographic"] = [f for f in ALL_FEATURES if f not in DEMO_FEATURES]
ablation_sets["only_demographic"] = DEMO_FEATURES

# bandas
for band in BANDS:
    ablation_sets[f"remove_{band}"] = [f for f in ALL_FEATURES if not f.endswith(band)]
    ablation_sets[f"only_{band}"] = [f for f in ALL_FEATURES if f.endswith(band)]

# canais
for ch in CHANNELS:
    ablation_sets[f"remove_{ch}"] = [f for f in ALL_FEATURES if not f.startswith(ch)]

ablation_results = {}
models = [
    ("XGB", XGBRegressor(learning_rate=0.1, max_depth=6, n_estimators=300, n_jobs= None)),
]
seeds = [13, 44, 71]
for seed in seeds:

    ablation_results[seed] = {}

    for model_name, model in models:
        
        ablation_results[seed][model_name] = {}
        set_seed(seed=seed)
        selected_subjects = sorted(np.random.choice(subjects, size=15, replace=False))

        for name, feat_set in tqdm(ablation_sets.items(), desc=f"Ablation - {model_name} - seed {seed}", leave = False):

            metric_list = ["mae", "accuracy", "qwk", "spearmanr"]
        
            ablation_features = df_features[feat_set]
            
            dfs, metrics = LOSOCV(model, ablation_features, df_labels, metric_list, seed=seed, test_subjects=selected_subjects)

            ablation_results[seed][model_name][name] = metrics


imporatance_all = computeAblationImportance(ablation_results, seeds, models)

for model in importance_all.keys():

    data_mean = {}
    data_std = {}

    for ablation in importance_all[model]:
        
        if "only_" in ablation:
            continue
        data_mean[ablation] = {}
        data_std[ablation] = {}
        
        for target in importance_all[model][ablation]:
            
            data_mean[ablation][target] = importance_all[model][ablation][target]["mean"]
            data_std[ablation][target] = importance_all[model][ablation][target]["std"]

    mean_df = pd.DataFrame(data_mean).T
    std_df  = pd.DataFrame(data_std).T

    # limpar nomes
    mean_df.index = [clean_feature_name(i) for i in mean_df.index]
    std_df.index  = mean_df.index

    for target in mean_df.columns:
        plotAblationTarget(mean_df, std_df, model, target)