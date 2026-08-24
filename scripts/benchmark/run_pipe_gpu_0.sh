#!/bin/bash

# Must be run from the repo root (paths below are relative to it), but this
# guard lets it also be invoked from anywhere via `bash scripts/benchmark/run_pipe_gpu_0.sh`.
cd "$(dirname "$0")/../.." || exit 1

# --- Configurações ---
QUERIES=(10 50 100)
SEEDS=(1 2 3)

GPU_NUMBER=0

PARAMS_FILE="files_config/benchmark/params_df_gpu_${GPU_NUMBER}.json"
DATASET_NAME="DANINHAS"
STRATEGY_1="SSRAEKmeansHCSampling"

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
    CUDA_VISIBLE_DEVICES=$GPU_NUMBER poetry run python trainer.py \
      --params_json $PARAMS_FILE \
      --dataset_name=$DATASET_NAME \
      --strategy_name $STRATEGY_1 \
      --n_query $n_query \
      --seed $seed \
      --n_round 8 \
      --dir_results=results/dalmax1/


    echo "Comando executado: CUDA_VISIBLE_DEVICES=$GPU_NUMBER poetry run python trainer.py --n_round 8 --params_json $PARAMS_FILE --dataset_name=$DATASET_NAME --strategy_name $STRATEGY_1 --n_query $n_query --seed $seed --dir_results=results/dalmax1/"

    echo "Par (n_query=$n_query, seed=$seed) finalizado."

  done
done

echo ""
echo "------------------------------------------------------------"
echo "Todos os testes foram concluídos."
echo "------------------------------------------------------------"

# Run ExperimentNotifier to send email notification
poetry run python ExperimentNotifier/main.py --dir_results=results/dalmax1/ --args "GPU_NUMBER=$GPU_NUMBER, STRATEGY_1=$STRATEGY_1"