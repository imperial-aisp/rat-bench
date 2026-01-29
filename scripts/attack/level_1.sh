anon_methods="presidio, pre_anon, gemini_basic, tem_eps11.0, azure, scrubadub, uniner, gliner, textwash, gpt_basic, gpt_clio, gpt_rescriber, llama_basic, llama_clio, llama_rescriber"

python3 main.py --data_path="benchmark/" \
                --results_path="benchmark/" \
                --level=1 \
                --attacker="gpt" \
                --model_version="gpt-4.1" \
                --anon_methods="$anon_methods" \
                --scenario="medical" \
                --uniqueness_results_folder="./results" \
                --anonymize="False" \
                --only_correctness="False"