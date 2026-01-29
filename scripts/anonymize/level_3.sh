#!/bin/sh
anon_methods="presidio, azure, scrubadub, uniner, gliner, textwash, gpt_basic, gpt_clio, gpt_rescriber, llama_basic, llama_clio, llama_rescriber, gemini_basic, tem"
timing_flag=1

python3 -m pii_benchmark.anonymize --results_path="benchmark/level_3.jsonl" \
                --data_path="benchmark/level_3.jsonl" \
                --anon_methods="$anon_methods" \
                --timing=$timing_flag \
                --scenario="random" \
                --epsilon=11