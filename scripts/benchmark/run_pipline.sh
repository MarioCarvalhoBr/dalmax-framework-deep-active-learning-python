#!/bin/bash
# Example: bash scripts/run_pipline.sh 0 1
# Lista de estratégias
strategies=(
    RandomSampling
    LeastConfidence
    MarginSampling
    EntropySampling
    LeastConfidenceDropout
    MarginSamplingDropout
    EntropySamplingDropout
    KMeansSampling
    KCenterGreedy
    BALDDropout
    # AdversarialBIM
    # AdversarialDeepFool
)

N_ROUND=10

echo ""
# Loop pelas estratégias
for strategy in "${strategies[@]}"; do
    
    # echo "Running $strategy"
    n_queries=(10)
    for n_query in "${n_queries[@]}"; do
        # echo "Running $strategy with $n_query queries"
        echo CUDA_VISIBLE_DEVICES=$1 python demo.py --params_json params_dnf.json --dataset_name=DANINHAS --strategy_name $strategy --n_query $n_query --seed $2 --n_round $N_ROUND
    done
done

echo ""
echo ""

# Loop pelas estratégias
for strategy in "${strategies[@]}"; do
    # echo "Running $strategy"
    n_queries=(50)
    for n_query in "${n_queries[@]}"; do
        # echo "Running $strategy with $n_query queries"
        echo CUDA_VISIBLE_DEVICES=$1 python demo.py --params_json params_dnf.json --dataset_name=DANINHAS --strategy_name $strategy --n_query $n_query --seed $2 --n_round $N_ROUND
    done
done

echo ""
echo ""

# Loop pelas estratégias
for strategy in "${strategies[@]}"; do
    # echo "Running $strategy"
    n_queries=(100)
    for n_query in "${n_queries[@]}"; do
        # echo "Running $strategy with $n_query queries"
        echo CUDA_VISIBLE_DEVICES=$1 python demo.py --params_json params_dnf.json --dataset_name=DANINHAS --strategy_name $strategy --n_query $n_query --seed $2 --n_round $N_ROUND
    done
done

# Example: CUDA_VISIBLE_DEVICES=1 python demo.py --params_json params_dnf.json --dataset_name=DANINHAS --strategy_name LeastConfidenceDropout --n_query 100 --seed 2 --n_round 10
echo ""
echo ""