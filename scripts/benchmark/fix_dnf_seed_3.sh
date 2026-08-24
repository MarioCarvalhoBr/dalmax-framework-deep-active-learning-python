# NOTE: requires params_df.json (not tracked in this repo); also references a
# nonexistent strategy name (SSRAESampling) — one-off scratch script, kept for history.
# GPU 0
# MarginSampling
#CUDA_VISIBLE_DEVICES=0  poetry run python demo.py --params_json params_df.json --dataset_name=DANINHAS --strategy_name MarginSampling --n_query 100 --seed 1 --n_round 10
CUDA_VISIBLE_DEVICES=1  poetry run python demo.py --params_json params_df.json --dataset_name=DANINHAS --strategy_name MarginSampling --n_query 100 --seed 2 --n_round 10
CUDA_VISIBLE_DEVICES=0  poetry run python demo.py --params_json params_df.json --dataset_name=DANINHAS --strategy_name MarginSampling --n_query 100 --seed 3 --n_round 10

# MarginSamplingDropout
#CUDA_VISIBLE_DEVICES=1  poetry run python demo.py --params_json params_df.json --dataset_name=DANINHAS --strategy_name MarginSamplingDropout --n_query 100 --seed 1 --n_round 10
CUDA_VISIBLE_DEVICES=0  poetry run python demo.py --params_json params_df.json --dataset_name=DANINHAS --strategy_name MarginSamplingDropout --n_query 100 --seed 2 --n_round 10
CUDA_VISIBLE_DEVICES=0  poetry run python demo.py --params_json params_df.json --dataset_name=DANINHAS --strategy_name MarginSamplingDropout --n_query 100 --seed 3 --n_round 10
# GPU 1


CUDA_VISIBLE_DEVICES=0  poetry run python demo.py --params_json params_df.json --dataset_name=DANINHAS --strategy_name SSRAESampling --n_query 10 --seed 3 --n_round 10
