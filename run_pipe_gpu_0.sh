#!/bin/bash

# --- Configurações ---
QUERIES=(10 50 100)
SEEDS=(1 2 3)

GPU_NUMBER=0

PARAMS_FILE="params_df_gpu_${GPU_NUMBER}.json"
DATASET_NAME="DANINHAS"
STRATEGY_1="SSRAEKmeansHCSampling"
STRATEGY_2="VCTexKmeansHCSampling"

# ---------------------

echo "Iniciando bateria de testes..."

# Itera sobre cada valor de n_query
for n_query in "${QUERIES[@]}"; do
  
  # Itera sobre cada valor de seed
  for seed in "${SEEDS[@]}"; do
    
    echo ""
    echo "------------------------------------------------------------"
    echo "EXECUTANDO: n_query=$n_query, seed=$seed"
    echo "------------------------------------------------------------"

    echo "Iniciando $STRATEGY_1 em GPU $GPU_NUMBER..."
    CUDA_VISIBLE_DEVICES=$GPU_NUMBER python demo.py \
      --params_json $PARAMS_FILE \
      --dataset_name=$DATASET_NAME \
      --strategy_name $STRATEGY_1 \
      --n_query $n_query \
      --seed $seed \
      --dir_results=results/dalmax0/

    CUDA_VISIBLE_DEVICES=$GPU_NUMBER python demo.py \
      --params_json $PARAMS_FILE \
      --dataset_name=$DATASET_NAME \
      --strategy_name $STRATEGY_2 \
      --n_query $n_query \
      --seed $seed \
      --dir_results=results/dalmax0/


    echo "Comando executado: CUDA_VISIBLE_DEVICES=$GPU_NUMBER python demo.py --params_json $PARAMS_FILE --dataset_name=$DATASET_NAME --strategy_name $STRATEGY_1 --n_query $n_query --seed $seed --dir_results=results/dalmax0/"
    echo "Comando executado: CUDA_VISIBLE_DEVICES=$GPU_NUMBER python demo.py --params_json $PARAMS_FILE --dataset_name=$DATASET_NAME --strategy_name $STRATEGY_2 --n_query $n_query --seed $seed --dir_results=results/dalmax0/"

    echo "Par (n_query=$n_query, seed=$seed) finalizado."

  done
done

echo ""
echo "------------------------------------------------------------"
echo "Todos os testes foram concluídos."
echo "------------------------------------------------------------"

# Run ExperimentNotifier to send email notification
python3 ExperimentNotifier/main.py --dir_results=results/dalmax0/