import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Plot Confusion matrix: Gera a matriz de confusão em .pdf para o modelo
import seaborn as sns
from sklearn.metrics import confusion_matrix


import torch
from utils.orchestrator import get_dataset, get_network_deep_learning, get_strategy

# My looger
from utils.LOGGER import get_logger, get_path_logger

# Global logger
logger = get_logger()
path_logger = get_path_logger()

def main(args):
    # Train model with PyTorch
    logger.warning("==========================================================================>")
    logger.warning(f"DalMax - Training the model with PyTorch...")
    logger.warning("==========================================================================>")

    logger.warning("ARGUMENTS (cli) AND PARAMETERS (json)")
    logger.warning("--------------------------------------------------------------------------")

    logger.warning("ARGS: " + json.dumps(vars(args), indent=4))
    logger.warning("--------------------------------------------------------------------------")

    # SETUP HYPERPARAMETERS
    params = None
    with open(args.params_json, "r") as f:
        params = json.load(f)
    logger.warning("PARAMS: " + json.dumps(params[args.dataset_name], indent=4))
    logger.warning("--------------------------------------------------------------------------")

    n_epoch = params[args.dataset_name]['n_epoch']
    data_dir = params[args.dataset_name]['data_dir']
    data_dir_dataset = os.path.basename(data_dir.rstrip("/"))

    # Create results directory
    if args.dir_results[-1] != '/':
        args.dir_results += '/'
    args.dir_results += f'{data_dir_dataset}/SEED_{args.seed}/NQ_{args.n_query}_NIL_{args.n_init_labeled}_NR_{args.n_round}_NE_{n_epoch}/'
    dir_results = args.dir_results + f"{args.strategy_name}/"    
    logger.warning(f"Results directory: {dir_results}")
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False

    # device
    use_cuda = torch.cuda.is_available()
    logger.warning(f"use_cuda: {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.warning(f"device: {device}")

    ## List to store accuracies and rounds
    all_acc = []
    all_precision = []
    all_recall = []
    all_f1_score = []
    all_rounds = []

    dataset = get_dataset(args.dataset_name, params)
    net = get_network_deep_learning(args.dataset_name, device, params)
    strategy = get_strategy(args.strategy_name)(dataset, net, logger)
    

    # start experiment
    start_time = time.time()
    dataset.initialize_labels(args.n_init_labeled, args.strategy_name)

    # round 0 accuracy
    logger.warning("Round 0")
    strategy.info()
    # strategy.train_full()
    strategy.train()
    
    preds = strategy.predict(dataset.get_test_data())

    acc = dataset.cal_test_acc(preds)
    acc_skl, precision, recall, f1_score = dataset.calc_metrics_sklearn(preds)
    
    all_acc.append(acc)
    all_precision.append(precision)
    all_recall.append(recall)
    all_f1_score.append(f1_score)

    logger.warning("==========================================================================>\n")
    print("INITIAL METRICS: ")
    logger.warning(f"Round 0 testing accuracy: {acc}")
    logger.warning(f"Round 0 acc_skl: {acc_skl}")
    logger.warning(f"Round 0 precision: {precision}")
    logger.warning(f"Round 0 recall: {recall}")
    logger.warning(f"Round 0 f1_score: {f1_score}")
    logger.warning("==========================================================================>\n")

    # Get class names from the dataset
    class_names = dataset.get_classes_names()
    # ADD ROUNDS
    all_rounds.append(0)
    if args.n_round > 0:
        preds = None 
    
    for rd in range(1, args.n_round+1):
        print("FOR LOOP")
        logger.warning("==========================================================================>")
        logger.warning(f"Round {rd}")
        # query
        query_idxs = strategy.query(args.n_query)

        # update labels
        strategy.update(query_idxs)
        
        # info after query
        strategy.info()

        # train
        strategy.train()

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        acc = dataset.cal_test_acc(preds)
        acc_skl, precision, recall, f1_score = dataset.calc_metrics_sklearn(preds)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1_score.append(f1_score)
        logger.warning(f"Round {rd} testing accuracy: {acc}")
        logger.warning(f"Round {rd} precision: {precision}")
        logger.warning(f"Round {rd} recall: {recall}")
        logger.warning(f"Round {rd} f1_score: {f1_score}")

        all_acc.append(acc)
        all_rounds.append(rd)

        # Print acc, precision, recall and f1-score
        logger.warning(f'Local Accuracies: {all_acc}')
        logger.warning(f'Local Precision: {all_precision}')
        logger.warning(f'Local Recall: {all_recall}')
        logger.warning(f'Local F1-Score: {all_f1_score}')
        logger.warning(f'Local Rounds: {all_rounds}')
        logger.warning("==========================================================================>")

    cm = confusion_matrix(dataset.Y_test, preds)

    def plot_confusion_matrix(cm, class_names):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names, rotation=45)
        plt.tight_layout()
        plt.savefig(f"{dir_results}/confusion_matrix.pdf")

    plot_confusion_matrix(cm, class_names)

    # Convert all_acc, all_precision, all_recall and all_f1_score to float list
    all_acc = [float(x) for x in all_acc]
    all_precision = [float(x) for x in all_precision]
    all_recall = [float(x) for x in all_recall]
    all_f1_score = [float(x) for x in all_f1_score]
    
    
    logger.warning("==========================================================================>\n")
    logger.warning(f'FINAL METRICS: ')
    logger.warning(f'Final Accuracies: {all_acc}')
    logger.warning(f'Final Accuracies (sklearn): {acc_skl}')
    logger.warning(f'Final Precision: {all_precision}')
    logger.warning(f'Final Recall: {all_recall}')
    logger.warning(f'Final F1-Score: {all_f1_score}')
    logger.warning("==========================================================================>\n")
    logger.warning(f'Final Rounds: {all_rounds}')

    end_time = time.time()
    final_time_in_seconds = end_time - start_time
    logger.warning(f"Total time: {final_time_in_seconds} seconds")
    logger.warning("==========================================================================>")
    
    def plot_metrics(data, title, ylabel, filename):
        plt.figure()
        plt.plot(all_rounds, data, marker='o')
        plt.title(title)
        plt.xlabel('Rounds')
        plt.ylabel(ylabel)
        plt.grid()
        plt.savefig(filename)
        plt.close()
    
    # Plot accuracies: acc, precision, recall and f1-score
    plot_metrics(all_acc, 'Accuracy', 'Accuracy', f"{dir_results}/accuracy.pdf")
    plot_metrics(all_precision, 'Precision', 'Precision', f"{dir_results}/precision.pdf")
    plot_metrics(all_recall, 'Recall', 'Recall', f"{dir_results}/recall.pdf")
    plot_metrics(all_f1_score, 'F1-Score', 'F1-Score', f"{dir_results}/f1_score.pdf")

    # Move generated files to results directory
    os.rename(path_logger, dir_results + "/log-dalmax.log")

    # Save the model
    strategy.save_model(dir_results)

    ## SAVE DATA in JSON
    # Dados de configuração
    dados_config_results = {
        'dataset_name': args.dataset_name,
        'strategy_name': args.strategy_name,

        'n_init_labeled': args.n_init_labeled,
        'n_query': args.n_query,
        'n_round': args.n_round,
        'seed': args.seed,
        
        'all_acc': all_acc,
        'all_precision': all_precision,
        'all_recall': all_recall,
        'all_f1_score': all_f1_score,
        'rounds': all_rounds,
    }

    # Salvar dados em um arquivo JSON
    json_path = os.path.join(dir_results, "results.json")
    with open(json_path, "w") as json_file:
        json.dump(dados_config_results, json_file, indent=4)

    predictions = []
    for i in range(len(dataset.Y_test)):
        real_class = dataset.Y_test[i].item()
        predicted_class = preds[i].item()
        real_class_name = dataset.get_class_name(real_class)
        predicted_class_name = dataset.get_class_name(predicted_class)
        correct = 1 if real_class == predicted_class else 0
        path = dataset.Z_test_paths[i]
        predictions.append([i, real_class_name, predicted_class_name, correct, path])

    predictions_df = pd.DataFrame(predictions, columns=["Image Index", "Real Class", "Predicted Class", "Correct", "Path"])
    predictions_csv_path = os.path.join(dir_results, "predictions.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)
    logger.warning(f"Predictions saved to {predictions_csv_path}")

    print(f"Dados salvos em {json_path}")
    print(f"Predictions saved in {predictions_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_results', type=str, default='results/dalmax', help='Results directory')
    parser.add_argument('--params_json', type=str, default='params.json', help='Params JSON file. See example in README.md')

    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--n_init_labeled', type=int, default=100, help="number of init labeled samples")
    parser.add_argument('--n_query', type=int, default=10, help="number of queries per round")
    parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
    parser.add_argument('--dataset_name', type=str, default="CIFAR10", choices=["CIFAR10", "DANINHAS"], help="dataset")
    parser.add_argument('--strategy_name', type=str, default="RandomSampling", 
                        choices=["RandomSampling", 
                                "LeastConfidence", 
                                "MarginSampling", 
                                "EntropySampling", 
                                "LeastConfidenceDropout", 
                                "MarginSamplingDropout", 
                                "EntropySamplingDropout", 
                                "KMeansSampling",
                                "KCenterGreedy", 
                                "BALDDropout", 
                                "AdversarialBIM", 
                                "AdversarialDeepFool",
                                "SSRAEKmeansHCSampling",
                                "SSRAEKmeansSampling",
                                "VCTexKmeansSampling"], help="query strategy")
    args = parser.parse_args()

    main(args)
