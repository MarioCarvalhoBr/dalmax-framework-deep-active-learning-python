#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name RandomSampling --n_query 100 --seed 2 && notify-send "Process finished" "RandomSampling"
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name LeastConfidence --n_query 100 --seed 2 && notify-send "Process finished" "LeastConfidence"
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name MarginSampling --n_query 100 --seed 2 && notify-send "Process finished" "MarginSampling"
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name EntropySampling --n_query 100 --seed 2 && notify-send "Process finished" "EntropySampling"
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name LeastConfidenceDropout --n_query 100 --seed 2 && notify-send "Process finished" "LeastConfidenceDropout"
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name MarginSamplingDropout --n_query 100 --seed 2 && notify-send "Process finished" "MarginSamplingDropout"
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name EntropySamplingDropout --n_query 100 --seed 2 && notify-send "Process finished" "EntropySamplingDropout"
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name KMeansSampling --n_query 100 --seed 2 && notify-send "Process finished" "KMeansSampling"
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name KCenterGreedy --n_query 100 --seed 2 && notify-send "Process finished" "KCenterGreedy"
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name BALDDropout --n_query 100 --seed 2 && notify-send "Process finished" "BALDDropout"

# CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name AdversarialBIM
# CUDA_VISIBLE_DEVICES=0 python demo.py --dataset_name=DANINHAS --strategy_name AdversarialDeepFool