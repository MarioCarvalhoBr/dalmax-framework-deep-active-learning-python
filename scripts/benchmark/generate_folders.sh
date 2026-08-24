#!/bin/bash
# NOTE: uses the CLI default --params_json (params.json, not tracked in this repo) unless overridden
# SEED 1
CUDA_VISIBLE_DEVICES=0 poetry run python demo.py --dataset_name=DANINHAS --strategy_name RandomSampling --n_query 10 --seed 1
CUDA_VISIBLE_DEVICES=0 poetry run python demo.py --dataset_name=DANINHAS --strategy_name RandomSampling --n_query 50 --seed 1
CUDA_VISIBLE_DEVICES=0 poetry run python demo.py --dataset_name=DANINHAS --strategy_name RandomSampling --n_query 100 --seed 1

# SEED 2
CUDA_VISIBLE_DEVICES=0 poetry run python demo.py --dataset_name=DANINHAS --strategy_name RandomSampling --n_query 10 --seed 2
CUDA_VISIBLE_DEVICES=0 poetry run python demo.py --dataset_name=DANINHAS --strategy_name RandomSampling --n_query 50 --seed 2
CUDA_VISIBLE_DEVICES=0 poetry run python demo.py --dataset_name=DANINHAS --strategy_name RandomSampling --n_query 100 --seed 2

# SEED 3
CUDA_VISIBLE_DEVICES=0 poetry run python demo.py --dataset_name=DANINHAS --strategy_name RandomSampling --n_query 10 --seed 3
CUDA_VISIBLE_DEVICES=0 poetry run python demo.py --dataset_name=DANINHAS --strategy_name RandomSampling --n_query 50 --seed 3
CUDA_VISIBLE_DEVICES=0 poetry run python demo.py --dataset_name=DANINHAS --strategy_name RandomSampling --n_query 100 --seed 3