# Example usage: python 4_report_build_average_results.py  --input_dir results/dalmax1/daninhas_full/results/ --pattern SEED*
# Example usage: python 4_report_build_average_results.py

import os
import json
import csv
import glob
import enum
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsType(enum.Enum):
    ALL_ACC = "all_acc"
    ALL_PRECISION = "all_precision"
    ALL_RECALL = "all_recall"
    ALL_F1_SCORE = "all_f1_score"

    DICT = {'Accuracy': ALL_ACC, 'Precision': ALL_PRECISION, 'Recall': ALL_RECALL, 'F1-score': ALL_F1_SCORE}

def create_csv_tables(dados, dir_results):
    # List of NQ configurations to process
    nq_configs = []
    
    # Get all keys seed_data[nq_config] (methods) from dados
    nq_configs_keys = dados[list(dados.keys())[0]].keys()
    nq_configs = sorted(list(nq_configs_keys))
    
    # Loop through each NQ configuration
    for nq_config in nq_configs:
        # Dictionary to hold lists of metrics for each method
        method_metrics = {}
        
        # Loop through each seed
        for seed, seed_data in dados.items():
            if nq_config in seed_data:
                methods = seed_data[nq_config]
                for method, metrics in methods.items():
                    if method not in method_metrics:
                        method_metrics[method] = {
                            'acc': [],
                            'precision': [],
                            'recall': [],
                            'f1_score': []
                        }
                    # Collect metric values
                    method_metrics[method]['acc'].append(metrics['all_acc'])
                    method_metrics[method]['precision'].append(metrics['all_precision'])
                    method_metrics[method]['recall'].append(metrics['all_recall'])
                    method_metrics[method]['f1_score'].append(metrics['all_f1_score'])
        
        # Prepare data for CSV
        csv_data = []
        # Write header
        header = ['method', 'acc', 'precision', 'recall', 'f1_score']
        csv_data.append(header)
        
        # Write method data
        for method, metrics in method_metrics.items():
            # Calculate mean and std for each metric
            acc_mean = np.mean(metrics['acc'])
            acc_std = np.std(metrics['acc'])
            precision_mean = np.mean(metrics['precision'])
            precision_std = np.std(metrics['precision'])
            recall_mean = np.mean(metrics['recall'])
            recall_std = np.std(metrics['recall'])
            f1_mean = np.mean(metrics['f1_score'])
            f1_std = np.std(metrics['f1_score'])
            
            # Format the values as "mean ± std"
            acc = f"{acc_mean:.4f} (±{acc_std:.4f})"
            precision = f"{precision_mean:.4f} (±{precision_std:.4f})"
            recall = f"{recall_mean:.4f} (±{recall_std:.4f})"
            f1_score = f"{f1_mean:.4f} (±{f1_std:.4f})"
            
            # Append the row
            csv_data.append([method, acc, precision, recall, f1_score])
        
        # Create directory if it doesn't exist
        config_dir = os.path.join(dir_results,"results", "AVERAGES", nq_config)
        os.makedirs(config_dir, exist_ok=True)
        
        # Define the CSV file path
        csv_file = os.path.join(config_dir, "tablea_com_media.csv")
        
        # Write to CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerows(csv_data)

        print(f"FULL CSV with data experiment saved in {csv_file}")

def list_folders_with_pattern(initial_path, pattern="*"):
    """Lists folders within a given path that match a specific pattern.

    Args:
        initial_path: The initial directory path.
        pattern: The pattern to match (default is '*', matching all).

    Returns:
        A list of folder paths that match the pattern.
    """

    matching_files = glob.glob(os.path.join(initial_path, pattern))
    # Filter only folders
    folders = [file for file in matching_files if os.path.isdir(file)]
    folders = sorted(folders)
    return folders

def save_dict_to_json(data, json_path):
        # SAVE JSON
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print(f"JSON with data experiment saved in {json_path}")

def plot_results(input_dir, path_nq_folder, folders, basename_path_seed):
    # CREATE DIR RESULTS
    nq_basename = os.path.basename(path_nq_folder)
    dir_results = f"{input_dir}/results/{basename_path_seed}/{nq_basename}/"
    os.makedirs(dir_results, exist_ok=True)

    data_strategies = {}
    data_settings = {}
    

    for folder in folders:
        json_path = os.path.join(folder, "results.json")
        
        if not os.path.exists(json_path):
            raise ValueError(f"File {json_path} not found.")
        
        local_data_json = {}
        with open(json_path, "r") as json_file:
            local_data_json = json.load(json_file)

        method_local = local_data_json['strategy_name']
        data_strategies[method_local] = {
            MetricsType.ALL_ACC.value: local_data_json[MetricsType.ALL_ACC.value],
            MetricsType.ALL_PRECISION.value: local_data_json[MetricsType.ALL_PRECISION.value],
            MetricsType.ALL_RECALL.value: local_data_json[MetricsType.ALL_RECALL.value],
            MetricsType.ALL_F1_SCORE.value: local_data_json[MetricsType.ALL_F1_SCORE.value]
        }

        # Get the last settings data
        data_settings = local_data_json

    # DICT SETTINGS DATA TO SAVE IN JSON
    new_data_config = {}
    new_data_config['dataset_name'] = data_settings['dataset_name']
    new_data_config['n_init_labeled'] = data_settings['n_init_labeled']
    new_data_config['n_query'] = data_settings['n_query']
    new_data_config['n_round'] = data_settings['n_round']
    new_data_config['rounds'] = data_settings['rounds']
    new_data_config['dir_results'] = dir_results
    new_data_config['data'] = data_strategies
    
    data_result = {}
    for method, value_dict in data_strategies.items():
        last_values = {}
        for metric, value_list in value_dict.items():
            last_values[metric] = value_list[-1]

        data_result[method] = last_values
        
    # SAVE CSV
    df = pd.DataFrame(data_result).T
    df.to_csv(os.path.join(dir_results, "data_experiment.csv"), sep=';')
    print(f"CSV with data local experiment saved in {os.path.join(dir_results, 'data_experiment.csv')}")

    # SAVE JSON
    save_dict_to_json(new_data_config, os.path.join(dir_results, "data_experiment.json"))
    
    # SAVE PLOT
    save_plot(new_data_config, is_show=False)

    return data_result

def save_plot(new_data_config, is_show=False):
    data = new_data_config['data']
    local_rounds = new_data_config['rounds']
    dir_results = new_data_config['dir_results']
    
    metrics = MetricsType.DICT.value
    for key, value in metrics.items():
    
        # SETTINGS PLOT
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 6))

        markers = ['o', '*', 's', 'D', '^', 'P', 'X']
        colors = sns.color_palette("husl", len(data))

        # VAR SETTINGS
        value = value
        ylabel = key

        for i, (method, values) in enumerate(data.items()):
            marker = markers[i % len(markers)]
            plt.plot(local_rounds, values[value], label=method, color=colors[i], marker=marker, markersize=8, linestyle='-')
            '''
            if method == 'RandomSampling':
                plt.plot(local_rounds, values[value], label=method, color=colors[i], marker=marker, markersize=8, linestyle='-')
            else:
                plt.plot(local_rounds, values[value], label=method, color=colors[i])
            '''
        
        plt.title("Model comparison", fontsize=14)
        plt.xlabel("Rounds", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(title="Models")
        plt.tight_layout()

        path_plot = os.path.join(dir_results, f"metric_{ylabel.lower()}_methods_comparison.pdf")
        plt.savefig(path_plot)

        print(f"Plot to metric {key} saved in {path_plot}")
        if is_show:
            plt.show()

def main(args):
    input_dir = args.input_dir
    pattern = args.pattern

    # Listar todas as pastas de input_dir
    folders_seeds = list_folders_with_pattern(input_dir, pattern)

    print(f"Pasta de entrada: {input_dir}")
    print(f"Padrão: {pattern}")
    print(f"Diretórios com o padrão {pattern}: {folders_seeds}")
    print(f"Total de diretórios: {len(folders_seeds)}")
    print("\n")

    dict_data_seeds = {}
    for path_folder in folders_seeds:
        print(f"\n==>Path: {path_folder}")
        basename_path_seed = os.path.basename(path_folder)
        
        nq_folders = list_folders_with_pattern(path_folder, "NQ_*")

        dict_data_nq = {}
        for path_nq_folder in nq_folders:
            print(f"\n==>Path NQ: {path_nq_folder}\n")
            basename_path_nq = os.path.basename(path_nq_folder)
            
            json_path = os.path.join(path_nq_folder, "data_experiment.json")
        
            if not os.path.exists(json_path):
                raise ValueError(f"File {json_path} not found.")
            
            local_data_json = {}
            with open(json_path, "r") as json_file:
                local_data_json = json.load(json_file)

            dict_data_nq[basename_path_nq] = local_data_json['data']

        dict_data_seeds[basename_path_seed] = dict_data_nq

    
    # print("\ndict_data_seeds: ", dict_data_seeds)
    
    # NEW: Extract SEEDS, NQS e METHODS from dict_data_seeds
    SEEDS = sorted(list(dict_data_seeds.keys()))
    NQS = sorted(list(dict_data_seeds[SEEDS[0]].keys()))
    METHODS = sorted(list(dict_data_seeds[SEEDS[0]][NQS[0]].keys()))
    
    print("\nSEEDS: ", SEEDS)
    print("NQS: ", NQS)
    print("METHODS: ", METHODS)
    print("\n")
    
    def calculate_mean(list1, list2, list3):
                    return [sum(x) / 3 for x in zip(list1, list2, list3)]
    metrics = MetricsType.DICT.value
    for metric_value, metric_key in metrics.items():
        METRIC = metric_key
        for nq in NQS:
            dict_data_final = {}
            for method in METHODS:
                # print(f'\nMETHOD: {method}')

                acc_1 = dict_data_seeds[SEEDS[0]][nq][method][METRIC]
                acc_2 = dict_data_seeds[SEEDS[1]][nq][method][METRIC]
                acc_3 = dict_data_seeds[SEEDS[2]][nq][method][METRIC]
                
                sum_all_acc_method = calculate_mean(acc_1, acc_2, acc_3)
                # print(f'Average seed for metric {METRIC}, nq ({nq}) and method {method}: {sum_all_acc_method}')
                dict_data_final[method] = sum_all_acc_method

            
            # Plot results
            markers = ['o', '*', 's', 'D', '^', 'P', 'X']
            colors = sns.color_palette("husl", len(dict_data_final))

            plt.figure(figsize=(12, 7))
            for i, (method, values) in enumerate(dict_data_final.items()):
                marker = markers[i % len(markers)]
                
                if method == 'RandomSampling':
                    # COnfigure a fonte da legenda
                    plt.plot(values, label=f'{method}', marker='o', markersize=8, linestyle='-', color='black')
                else:
                    plt.plot(values, label=method, color=colors[i], marker=marker, markersize=8, linestyle='-')

            # Deixar plot com grid 
            plt.grid()
            nq_split = nq.split('_')[1]
            plt.title(f"Model comparison with n_query {nq_split}", fontsize=14, color='black')
            plt.xlabel("Rounds", fontsize=12)
            plt.ylabel(f"{metric_value}", fontsize=12)
            plt.legend()
            path_save = f"results/dalmax1/daninhas_full/results/AVERAGES/{nq}/metric_{metric_key}_methods_comparison.pdf"
            plt.savefig(path_save)
            is_show = False
            print(f"Save plot in {path_save}")
            if is_show:
                plt.show()
            plt.close()

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="results/dalmax1/daninhas_full/results/", help="Input directory path.")
    parser.add_argument("--pattern", type=str, default="SEED*", help="Pattern to match (default is '*', matching all).")

    args = parser.parse_args()

    main(args)