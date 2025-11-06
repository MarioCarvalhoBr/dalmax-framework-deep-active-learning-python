#!/bin/bash

# --- Configurações ---
QUERIES=(10 50 100)
SEEDS=(1 2 3)

PARAMS_FILE="params_df.json"
DATASET_NAME="DANINHAS"
STRATEGY="SSRAEKmeansHCSampling"
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

    echo "Iniciando $STRATEGY em GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python demo.py \
      --params_json $PARAMS_FILE \
      --dataset_name=$DATASET_NAME \
      --strategy_name $STRATEGY \
      --n_query $n_query \
      --seed $seed
    
    
    echo "Comando executado: CUDA_VISIBLE_DEVICES=0 python demo.py --params_json $PARAMS_FILE --dataset_name=$DATASET_NAME --strategy_name $STRATEGY --n_query $n_query --seed $seed"


    echo "Par (n_query=$n_query, seed=$seed) finalizado."

  done
done

echo ""
echo "------------------------------------------------------------"
echo "Todos os testes foram concluídos."
echo "------------------------------------------------------------"