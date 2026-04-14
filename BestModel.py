from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

def load_results(folder_path="."):
    results_all = {}

    for file in Path(folder_path).glob("*.xlsx"):

        model_name = file.stem  
        if file.name.startswith("~$"):
            continue
        
        sheets = pd.read_excel(file,index_col=0, sheet_name=None)
        
        
        results_all[model_name] = sheets
    return results_all

def get_metrics(results):

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

RESULTS_LOADED = load_results('Models')
METRICS_LOADED = get_metrics(RESULTS_LOADED)

def groupBySeed(metrics_loaded):
    grouped = defaultdict(list)

    for model_name, targets in metrics_loaded.items():
        
        if "_seed_" in model_name:
            base_name = model_name.split("_seed_")[0]
        else:
            base_name = model_name
        
        grouped[base_name].append(targets)

    return grouped

def averageMetrics(grouped_metrics):
    
    aggregated = {}

    for base_model, runs in grouped_metrics.items():
        
        aggregated[base_model] = {}

        for target in runs[0].keys():
            
            aggregated[base_model][target] = {}

            for metric in runs[0][target].keys():
                
                values = [
                    run[target][metric]["mean"]
                    for run in runs
                ]

                aggregated[base_model][target][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values)
                }

    return aggregated

def getTopK(metrics_agg, k=5):
    
    top_k_per_target = {}

    # pegar lista de targets
    targets = list(next(iter(metrics_agg.values())).keys())

    for target in targets:
        
        ranking = []

        for model, target_metrics in metrics_agg.items():
            
            qwk_mean = target_metrics[target]["qwk"]["mean"]
            qwk_std  = target_metrics[target]["qwk"]["std"]

            ranking.append({
                "model": model,
                "mean": qwk_mean,
                "std": qwk_std
            })

        # ordenar
        ranking = sorted(ranking, key=lambda x: x["mean"], reverse=True)

        top_k_per_target[target] = ranking[:k]

    return top_k_per_target

if __name__ == '__main__':
    top_k = 5

    grouped = groupBySeed(METRICS_LOADED)
    metrics_avg = averageMetrics(grouped)

    top5 = getTopK(metrics_avg, k=top_k)
    with open("best_models.txt", "w", encoding="utf-8") as f:
        for target in top5:
            print(f"\n{target}\n\n")
            f.write(f"\n{target}\n\n")
            for i in range(len(top5[target])):
                line = f"{top5[target][i]['model']} → QWK: {top5[target][i]['mean']:.4f} ± {top5[target][i]['std']:.4f}"
                print(line)
                f.write(line)
    f.close()