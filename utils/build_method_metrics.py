"""
Script para calcular médias de métricas de Active Learning através de múltiplas seeds.

Exemplos de uso:
1. Calcular média do MarginSampling no round 8 com NQ=100:
   python3 main.py --method MarginSampling --round 8 --nq 100

2. Calcular média do LeastConfidence no round 5 com NQ=50 em pasta específica:
   python3 main.py --method LeastConfidence --round 5 --nq 50 --input_folder ./resultados

3. Calcular média do BALDDropout no round 3 com NQ=10 usando padrão customizado de seed:
   python3 main.py --method BALDDropout --round 3 --nq 10 --seed "EXPERIMENTO_*"
"""

import json
import argparse
import os
from glob import glob


def calculate_average(method_name, round_num, nq, input_folder, seed_pattern):
    """
    Calcula a média das métricas para um método específico em um round específico
    através de múltiplas seeds.
    
    Args:
        method_name: Nome do método (e.g., 'MarginSampling')
        round_num: Número do round (1-10, desconsiderando o round 0)
        nq: Número de queries (e.g., 10, 50, 100)
        input_folder: Pasta de entrada onde estão os dados
        seed_pattern: Padrão glob para buscar as pastas de seed (e.g., 'SEED_*')
    """
    # Mudar para a pasta de entrada
    original_dir = os.getcwd()
    os.chdir(input_folder)
    
    # Construir o padrão do diretório NQ
    nq_pattern = f"NQ_{nq}_NIL_100_NR_10_NE_10"
    
    # Encontrar todas as pastas SEED_*
    seed_dirs = sorted(glob(seed_pattern))
    
    if not seed_dirs:
        print(f"Erro: Nenhuma pasta encontrada com o padrão '{seed_pattern}' em '{input_folder}'!")
        os.chdir(original_dir)
        return
    
    print(f"Pasta de entrada: {input_folder}")
    print(f"Processando método: {method_name}")
    print(f"Round: {round_num}")
    print(f"NQ: {nq}")
    print(f"Padrão de seed: {seed_pattern}")
    print(f"Seeds encontradas: {seed_dirs}\n")
    
    # Métricas a processar
    metrics = ['all_acc', 'all_precision', 'all_recall', 'all_f1_score']
    
    # Armazenar valores de cada seed
    seed_values = {metric: [] for metric in metrics}
    
    # Processar cada SEED
    for seed_dir in seed_dirs:
        results_path = os.path.join(seed_dir, nq_pattern, method_name, "results.json")
        
        if not os.path.exists(results_path):
            print(f"Aviso: Arquivo não encontrado: {results_path}")
            continue
        
        # Ler o arquivo JSON
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Verificar se o round existe (lembrando que round_num é 1-indexed, mas no array é 0-indexed após remover o primeiro)
        # Os dados têm 11 valores (rounds 0-10), queremos ignorar o índice 0
        for metric in metrics:
            if metric in data:
                values = data[metric]
                # Verificar se o round_num é válido (1-10)
                if round_num < 1 or round_num > len(values) - 1:
                    print(f"Erro: Round {round_num} inválido! Deve estar entre 1 e {len(values) - 1}")
                    os.chdir(original_dir)
                    return
                
                # Pegar o valor do round (round_num já considera que ignoramos o índice 0)
                value = values[round_num]
                seed_values[metric].append(value)
                print(f"{seed_dir} - {metric}: {value}")
        print("-"*50)
    # Calcular e exibir as médias
    print("\n" + "="*50)
    print(f"MÉTRICAS DE {method_name}")
    print(f"MÉDIAS DE {len(seed_dirs)} SEEDS")
    print("="*50)
    
    for metric in metrics:
        if seed_values[metric]:
            avg = sum(seed_values[metric]) / len(seed_values[metric])
            print(f"{metric}: {avg:.6f}")
        else:
            print(f"{metric}: Sem dados disponíveis")
    
    print("="*50)
    
    # Voltar para o diretório original
    os.chdir(original_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calcula médias de métricas de Active Learning através de múltiplas seeds.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python3 build_method_metrics.py --method MarginSampling --round 8 --nq 100
  python3 build_method_metrics.py --method LeastConfidence --round 5 --nq 50 --input_folder ./resultados
  python3 utils/build_method_metrics.py --method SSRAEKmeansHCSampling --round 10 --nq 100 --seed "SEED_*" --input_folder results/exp_3_dalmax/daninhas_full/
  python3 utils/build_method_metrics.py --method VCTexKmeansHCSampling --round 10 --nq 100 --seed "SEED_*" --input_folder results/exp_3_dalmax/daninhas_full/
        """
    )
    
    parser.add_argument('--method', type=str, required=True,
                        help='Nome do método (e.g., MarginSampling, LeastConfidence, BALDDropout)')
    parser.add_argument('--round', type=int, required=True,
                        help='Número do round (1-10, ignorando o round 0)')
    parser.add_argument('--nq', type=int, required=True,
                        help='Número de queries (e.g., 10, 50, 100)')
    parser.add_argument('--input_folder', type=str, default='.',
                        help='Pasta de entrada onde estão os dados (padrão: diretório atual)')
    parser.add_argument('--seed', type=str, default='SEED_*',
                        help='Padrão glob para buscar as pastas de seed (padrão: "SEED_*")')
    
    args = parser.parse_args()
    
    calculate_average(
        method_name=args.method,
        round_num=args.round,
        nq=args.nq,
        input_folder=args.input_folder,
        seed_pattern=args.seed
    )
