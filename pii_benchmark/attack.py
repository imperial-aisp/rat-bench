import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from pii_benchmark.attackers.get_attacker import get_attacker
from pii_benchmark.evaluation import check_guess_correctness
from pii_benchmark.utils import write_output
from synthetic_data_generation.utils import write_output_async
from pii_benchmark.uniqueness import compute_reid_risk

def only_check_correctness(profiles, anon_methods, attacker_name, scenario,
           results_path, uniqueness_results_path, level, language, dataset):
    # load profiles
    profiles = check_guess_correctness(profiles, anon_methods, attacker_name=attacker_name)
    write_output(f"{results_path}/level_{level}.jsonl", profiles)
    compute_reid_risk(profiles=profiles, methods=anon_methods, attacker=attacker_name,
                      results_path=f"{uniqueness_results_path}/{scenario}/level_{level}_attacker_{attacker_name}.pickle",
                      dataset=dataset)

def attack(profiles, anon_methods, attacker_name, model_version, scenario,
           results_path, uniqueness_results_path, level, language=None, dataset="PUMS", force_rerun_attack=False):
    
    attacker = get_attacker(attacker_name, model_version)

    # print(f"Starting attack, language = {language}")
    # print(f"language or English = {language or 'English'}")
    
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
                language or "English",
            )
            for profile in profiles
            # profile_id, text, attacker, scenario, attributes, language
        ]

        if f"guesses_{anon_method}_{attacker_name}" not in profiles[0] or force_rerun_attack:

            # print(f"Running attack for {attacker_name} on {anon_method} anonymization method in language {language}")
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(attack_one_profile, inp): inp for inp in inputs}
                for future in tqdm(as_completed(futures), total=len(inputs)):
                    results_list.append(future.result())
            guesses = {profile_id: guess for profile_id, guess, prompt in results_list}
            prompts = {profile_id: prompt for profile_id, guess, prompt in results_list}
            for profile in profiles:
                profile[f"guesses_{anon_method}_{attacker_name}"] = guesses[profile["id"]]
                profile[f"prompts_{anon_method}_{attacker_name}"] = prompts[profile["id"]]
            
            write_output_async(f"{results_path}/level_{level}.jsonl", profiles)

        if scenario in ["Concert ticket purchase", "Tourist information chatbot"]:
            print("Inferring public info for concert ticket purchase scenario")
            inputs = [
                (
                    profile["id"],
                    profile[ff],
                    attacker,
                    profile["public_info"],
                    profile["scenario"],
                    profile["language"] if "language" in profile else "English"
                ) for profile in profiles
            ]
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(infer_public_info_one_profile, inp): inp for inp in inputs}
                results_list = []
                for future in tqdm(as_completed(futures), total=len(inputs)):
                    results_list.append(future.result())
            guesses = {profile_id: guess for profile_id, guess, correctness, prompt in results_list}
            correctness = {profile_id: correctness for profile_id, guess, correctness, prompt in results_list}
            prompts = {profile_id: prompt for profile_id, guess, correctness, prompt in results_list}

            for profile in profiles:
                profile[f"guesses_public_info_{anon_method}_{attacker_name}"] = guesses[profile["id"]]
                profile[f"correctness_public_info_{anon_method}_{attacker_name}"] = correctness[profile["id"]]
                profile[f"prompts_public_info_{anon_method}_{attacker_name}"] = prompts[profile["id"]]
            write_output_async(f"{results_path}/level_{level}.jsonl", profiles)
        else:
            print(f"Not inferring public info since scenario is not concert ticket purchase, it's {scenario}")

    suff = attacker_name
    profiles = check_guess_correctness(profiles, anon_methods, attacker_name=suff)

    write_output_async(f"{results_path}/level_{level}.jsonl", profiles)

    compute_reid_risk(profiles=profiles, methods=anon_methods, attacker=attacker_name,
                      results_path=f"{uniqueness_results_path}/{scenario}/level_{level}_attacker_{attacker_name}.pickle",
                      dataset=dataset)

def attack_one_profile(args):
    profile_id, text, attacker, scenario, attributes, language = args
    guess, prompt = attacker.infer(text=text, attributes=attributes, scenario=scenario, language=language)
    return profile_id, guess, prompt

def infer_public_info_one_profile(args):
    profile_id, text, attacker, public_info, scenario, language = args
    guess, correctness, prompt = attacker.infer_public_info(text=text, public_info=public_info, scenario=scenario, language=language)
    return profile_id, guess, correctness, prompt

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

    PATH_TO_SAVE = RESULTS_PATH if RESULTS_PATH is not None else DATA_PATH

    if ONLY_CORRECTNESS=="True":
        print("Only checking correctness, not re-doing attack")
        only_check_correctness(profiles, ANON_METHODS, ATTACKER, SCENARIO, PATH_TO_SAVE, UNIQUENESS_RESULTS_FOLDER, LEVEL)
    else:
        print("Running attack from scratch")
        attack(profiles=profiles, anon_methods=ANON_METHODS, attacker_name=ATTACKER, model_version=MODEL_VERSION,
               scenario=SCENARIO, results_path=PATH_TO_SAVE, uniqueness_results_path=UNIQUENESS_RESULTS_FOLDER, level=LEVEL)