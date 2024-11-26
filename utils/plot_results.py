import os
import json
import matplotlib.pyplot as plt
import seaborn as sns


# Dados
dados = {
    'RandomSampling': {'data': [0.5223171889838556, 0.4358974358974359, 0.5356125356125356, 0.5004748338081672, 0.5161443494776828, 0.4919278252611586, 0.47910731244064575, 0.5208926875593543, 0.5246913580246914, 0.4843304843304843, 0.5479582146248813]},
    'MarginSampling': {'data': [0.5223171889838556, 0.6324786324786325, 0.5018993352326686, 0.5593542260208927, 0.5944919278252612, 0.7288698955365622, 0.6438746438746439, 0.5698005698005698, 0.4453941120607787, 0.5246913580246914, 0.5370370370370371]},
    'EntropySampling': {'data': [0.5223171889838556, 0.5090218423551757, 0.47768281101614435, 0.6562203228869895, 0.6452991452991453, 0.6419753086419753, 0.6163342830009497, 0.6220322886989553, 0.5546058879392213, 0.6286799620132953, 0.7022792022792023]},
    'LeastConfidence': {'data': [0.5223171889838556, 0.6533713200379867, 0.4990503323836657, 0.6215574548907882, 0.5716999050332384, 0.6723646723646723, 0.5821462488129154, 0.6752136752136753, 0.5142450142450142, 0.6020892687559354, 0.6481481481481481]},
    'LeastConfidenceDropout': {'data': [0.5223171889838556, 0.5631528964862298, 0.4097815764482431, 0.6001899335232669, 0.5327635327635327, 0.47673314339981004, 0.4876543209876543, 0.5370370370370371, 0.5954415954415955, 0.51994301994302, 0.5660018993352327]},
    'EntropySamplingDropout': {'data': [0.5223171889838556, 0.47863247863247865, 0.49240265906932573, 0.6771130104463438, 0.563627730294397, 0.48005698005698005, 0.5275403608736942, 0.5375118708452041, 0.5546058879392213, 0.49952516619183285, 0.5479582146248813]},
    'MarginSamplingDropout': {'data': [0.5223171889838556, 0.49145299145299143, 0.5147198480531814, 0.5935422602089269, 0.6144349477682811, 0.5626780626780626, 0.5484330484330484, 0.5522317188983855, 0.5474833808167141, 0.5807217473884141, 0.5341880341880342]},
    'KMeansSampling': {'data': [0.5223171889838556, 0.46438746438746437, 0.5123456790123457, 0.5954415954415955, 0.6177587844254511, 0.5446343779677113, 0.5218423551756886, 0.570275403608737, 0.5151946818613485, 0.5465337132003799, 0.5194681861348528]},
    'KCenterGreedy': {'data': [0.5223171889838556, 0.449667616334283, 0.49002849002849, 0.37274453941120606, 0.6096866096866097, 0.5802469135802469, 0.5341880341880342, 0.49382716049382713, 0.5422602089268755, 0.4710351377018044, 0.4862298195631529]},
    #'MODEL': {'data': [0.4, 0.54, 0.85]},
    #'MODEL': {'data': [0.4, 0.54, 0.85]},
}
local_rounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
local_rounds = local_rounds[:-3]

# Dados de configuração
dados_config = {
    'n_init_labeled': 100,
    'n_query': 10,
    'n_round': 10,
    'seed': 1,
    'dataset_name': 'DANINHAS',
    'data': dados,
    'rounds': local_rounds,
}

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
    valores['data'] = valores['data'][:-3]

    marcador = marcadores[i % len(marcadores)]  # Reutilizar marcadores se necessário
    plt.plot(local_rounds, valores['data'], label=modelo, color=cores[i], marker=marcador, markersize=8, linestyle='-')
# Configurações do gráfico
plt.title("Model comparison", fontsize=14)
plt.xlabel("Rounds", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(title="Models")
plt.tight_layout()

plt.savefig(os.path.join(dir_results, "model_comparison_epochs.pdf"))
plt.show()
