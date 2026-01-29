import argparse
import json

from tqdm import tqdm

from pii_benchmark.attackers.get_attacker import get_attacker
from pii_benchmark.evaluation import check_guess_correctness
from pii_benchmark.utils import write_output
from synthetic_data_generation.utils import write_output_async
from pii_benchmark.uniqueness import compute_reid_risk

MAX_DELAY = 60
BASE_DELAY = 2
MAX_RETRIES = 10
N_PROCESSORS = 10
RPM_LIMIT = 10
DELAY = 60 / RPM_LIMIT

def only_check_correctness(profiles, anon_methods, attacker_name, scenario,
           results_path, uniqueness_results_path, level):
    # load profiles
    profiles = check_guess_correctness(profiles, anon_methods, attacker_name=attacker_name)
    write_output(f"{results_path}/level_{level}.jsonl", profiles)
    compute_reid_risk(profiles=profiles, methods=anon_methods, attacker=attacker_name,
                      results_path=f"{uniqueness_results_path}/{scenario}/level_{level}_attacker_{attacker_name}.pickle")

def attack(profiles, anon_methods, attacker_name, model_version, scenario,
           results_path, uniqueness_results_path, level):
    
    attacker = get_attacker(attacker_name, model_version)
    
    print("Initialized attacker")

    for anon_method in anon_methods:

        print(f"Anon method {anon_method}")
        results_list = list()

        if anon_method=="pre_anon":
            ff = "text"
        else:
            ff = f"text_anon_{anon_method}"

        inputs = [
            (
                profile["id"],
                profile[ff],
                attacker,
                profile["scenario"],
                profile["features"],
            )
            for profile in profiles
        ]
            
        for input in tqdm(inputs):
            result = attack_one_profile(input)
            results_list.append(result)
        guesses = {profile_id: guess for profile_id, guess, prompt in results_list}
        prompts = {profile_id: prompt for profile_id, guess, prompt in results_list}
        for profile in profiles:
            profile[f"guesses_{anon_method}_{attacker_name}"] = guesses[profile["id"]]
            profile[f"prompts_{anon_method}_{attacker_name}"] = prompts[profile["id"]]
        
        write_output_async(f"{results_path}/level_{level}.jsonl", profiles)

    suff = attacker_name
    profiles = check_guess_correctness(profiles, anon_methods, attacker_name=suff)

    write_output_async(f"{results_path}/level_{level}.jsonl", profiles)
    
    compute_reid_risk(profiles=profiles, methods=anon_methods, attacker=attacker_name,
                      results_path=f"{uniqueness_results_path}/{scenario}/level_{level}_attacker_{attacker_name}.pickle")

def attack_one_profile(args):
    """
    Synchronous wrapper for attack_one_profile so it can run in a separate process.
    args: tuple of (profile_text, attacker_params, scenario, attributes)
    """
    profile_id, text, attacker, scenario, attributes = args
    guess, prompt = attacker.infer(text=text, attributes=attributes, scenario=scenario)
    return profile_id, guess, prompt

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--attacker", type=str)
    parser.add_argument("--model_version", type=str)
    parser.add_argument("--anon_methods", type=str)
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--uniqueness_results_folder", type=str)
    parser.add_argument("--only_correctness", type=str, default="False")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    ATTACKER = args.attacker
    MODEL_VERSION = args.model_version
    ANON_METHODS = args.anon_methods
    ANON_METHODS = [s.strip() for s in ANON_METHODS.split(",")]
    SCENARIO = args.scenario
    RESULTS_PATH = args.results_path
    UNIQUENESS_RESULTS_FOLDER = args.uniqueness_results_folder
    ONLY_CORRECTNESS = args.only_correctness
    LEVEL = args.level

    profiles = []
    with open(f"{DATA_PATH}/level_{LEVEL}.jsonl", "r") as f:
        for line in f:
            profiles.append(json.loads(line))

    print(f"loaded data, {len(profiles)} profiles")

    if ONLY_CORRECTNESS=="True":
        print("Only checking correctness, not re-doing attack")
        only_check_correctness(profiles)
    else:
        print("Running attack from scratch")
        # attack(profiles, ATTACKER, MODEL_VERSION, SCENARIO)
        if RESULTS_PATH is not None:
            PATH_TO_SAVE = RESULTS_PATH
        else:
            PATH_TO_SAVE = DATA_PATH

        attack(profiles=profiles, anon_methods=ANON_METHODS, attacker_name=ATTACKER, model_version=MODEL_VERSION,
               scenario=SCENARIO, results_path=PATH_TO_SAVE, uniqueness_results_path=UNIQUENESS_RESULTS_FOLDER, level=LEVEL)