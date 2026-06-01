import argparse
import json

from tqdm import tqdm

from pii_benchmark.attackers.get_attacker import get_attacker
from pii_benchmark.evaluation import check_guess_correctness
from pii_benchmark.utility import utility_scores
from synthetic_data_generation.utils import write_output

def compute_utility(anon_methods: list, results_path: str, summary_path: str, profiles: list|None=None, data_path: str|None = None):
    # load profiles

    if profiles is None and data_path is None:
        raise ValueError("Either profiles or data_path must be provided.")
    
    if profiles is None:
        profiles = []
        with open(data_path, "r") as f:
            for line in f:
                profiles.append(json.loads(line))
    
    average_scores = dict()
    for anon_method in anon_methods:
        rouge_scores = 0
        bleu_scores = 0
        print(f"Anon method {anon_method}")
        for profile in profiles:
            result = utility_scores(profile[f"text_anon_{anon_method}"], profile["text"])
            profile[f"text_anon_{anon_method}_rouge_score"] = result[0]
            profile[f"text_anon_{anon_method}_bleu_score"] = result[1]
            rouge_scores += result[0]
            bleu_scores += result[1]
        average_scores[f"text_anon_{anon_method}"] = {"rouge_score": rouge_scores/len(profiles), "bleu_score": bleu_scores/len(profiles)}

    
    write_output(results_path, profiles)

        
    with open(summary_path, "w") as outfile:
        print(json.dumps(average_scores), file=outfile)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--anon_methods", type=str)
    parser.add_argument("--results_path", type=str)
    parser.add_argument("--summary_path", type=str)
    args = parser.parse_args()

    DATA_PATH = args.data_path
    ANON_METHODS = args.anon_methods
    ANON_METHODS = [s.strip() for s in ANON_METHODS.split(",")]
    RESULTS_PATH = args.results_path
    SUMMARY_PATH = args.summary_path

    compute_utility(DATA_PATH, ANON_METHODS, RESULTS_PATH, SUMMARY_PATH)