config=config/mistral-inf-llm.yaml

mkdir benchmark/infinite-bench-result-cb

python benchmark/pred.py \
--config_path ${config} \
--output_dir_path benchmark/infinite-bench-result-cb \
--datasets code_debug,kv_retrieval
# kv_retrieval,passkey,number_string,code_debug,math_find,longbook_choice_eng

python benchmark/eval.py --dir_path benchmark/infinite-bench-result-cb