# Code about paper: Estimating LLM Uncertainty with Evidence.

run_model.py is our main program. After running, you will get a .json file records the result.
plot.py is to plot the result.
metrics.py includes calculations of uncertainty estimation indicators.


Run Example: CUDA_VISIBLE_DEVICES=0 python run_model.py --model_name "./llama_model_cache/LLM-Research/Meta-Llama-3.1-8B-Instruct" --alias "llama3.1_8b_K=4_tem=1"

Good Luck !
