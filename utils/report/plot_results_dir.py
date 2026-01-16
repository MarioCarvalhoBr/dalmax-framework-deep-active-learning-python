# Example usage: python utils/plot_results_dir.py --dir_input results/new_dalmax_balanceado_train_10_epochs_10_n_query

import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def main(args):
    dir_input = args.dir_input

    # Listar todas as pastas de dir_input
    folders = os.listdir(dir_input)

    print(f"Pasta de entrada: {dir_input}")
    print(f"Diretórios: {folders}")

    dados = {}
    dados_config = {}
    local_rounds = []

    for folder in folders:
        # Ler o arquivo JSON
        json_path = os.path.join(dir_input, folder, "results.json")
        
        # Verifica se o arquivo existe: Se não, imprime um aviso e pula para a próxima iteração
        if not os.path.exists(json_path):
            print(f"Arquivo {json_path} não encontrado.")
            continue
        
        with open(json_path, "r") as json_file:
            dados_config = json.load(json_file)

        # Adicionar ao dicionário de dados
        dados[dados_config['strategy_name']] = {'all_acc': dados_config['all_acc']}

        # Adicionar os rounds
        local_rounds = dados_config['rounds']


    dados_config['data'] = dados

    print(f"\nDados: {dados}")
    print(f"\nRounds: {local_rounds}")
    print(f"\nConfiguração: {dados_config}")

    dir_results = f"results/my_reports/{dados_config['dataset_name']}_seed_{dados_config['seed']}_n_init_{dados_config['n_init_labeled']}_n_query_{dados_config['n_query']}_n_round_{dados_config['n_round']}/"
    os.makedirs(dir_results, exist_ok=True)


    # Salvar dados em um arquivo JSON
    json_path = os.path.join(dir_results, "experimento_1_dados.json")
    with open(json_path, "w") as json_file:
        json.dump(dados_config, json_file, indent=4)

    print(f"Dados salvos em {json_path}")

    # Estilo do gráfico
    sns.set_theme(style="whitegrid")

    # Configurar o tamanho do gráfico
    plt.figure(figsize=(8, 6))
    # Lista de marcadores para os modelos
    marcadores = ['o', '*', 's', 'D', '^', 'P', 'X']  # Marcadores possíveis
    # Paleta de cores
    cores = sns.color_palette("husl", len(dados))
    # Plotar os dados
    for i, (modelo, valores) in enumerate(dados.items()):
        # Excluir os 3 ultimos valores

        marcador = marcadores[i % len(marcadores)]  # Reutilizar marcadores se necessário
        if modelo == 'RandomSampling':
            plt.plot(local_rounds, valores['all_acc'], label=modelo, color=cores[i], marker=marcador, markersize=8, linestyle='-')
        else:
            plt.plot(local_rounds, valores['all_acc'], label=modelo, color=cores[i])
    # Configurações do gráfico
    plt.title("Model comparison", fontsize=14)
    plt.xlabel("Rounds", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(title="Models")
    plt.tight_layout()

    plt.savefig(os.path.join(dir_results, "accuracy_model_comparison_epochs.pdf"))
    plt.show()

if __name__ == "__main__":

    # Argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_input", type=str, default="results/")

    args = parser.parse_args()

    main(args)


