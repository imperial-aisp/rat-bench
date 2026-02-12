import argparse
import json

from tqdm import tqdm

from pii_benchmark.anonymize import run_anonymization
from pii_benchmark.attack import attack, only_check_correctness
from pii_benchmark.utils import load_data, str2bool

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--results_path", type=str, default=None)
parser.add_argument("--level", type=int, default=1)
parser.add_argument("--attacker", type=str)
parser.add_argument("--model_version", type=str)
parser.add_argument("--anon_methods", type=str)
parser.add_argument("--scenario", type=str)
parser.add_argument("--uniqueness_results_folder", type=str)
parser.add_argument("--anonymize", type=str2bool, default=True)
parser.add_argument("--attack", type=str2bool, default=True)
parser.add_argument("--only_correctness", type=str2bool, default=False)
parser.add_argument("--timing_flag", type=str2bool, default=True)

parser.add_argument("--gemini_version", type=str, default="2.5-flash")
parser.add_argument("--llama_version", type=str, default="3.1-8B-Instruct")
parser.add_argument("--gpt_version", type=str, default="gpt-4o-mini")
parser.add_argument("--epsilon", type=float, default=10.0)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--language", type=str, default="English")
parser.add_argument("--attribute_list_iterative", type=float, default=1.0)

args = parser.parse_args()

DATA_PATH = args.data_path
ATTACKER = args.attacker
MODEL_VERSION = args.model_version
ANON_METHODS = args.anon_methods
ANON_METHODS = [s.strip() for s in ANON_METHODS.split(",")]
SCENARIO = args.scenario
RESULTS_PATH = args.results_path
UNIQUENESS_RESULTS_FOLDER = args.uniqueness_results_folder
LEVEL = args.level
GEMINI_VERSION = args.gemini_version
LLAMA_VERSION = args.llama_version
GPT_VERSION = args.gpt_version
EPSILON = args.epsilon
TEMPERATURE = args.temperature
LANGUAGE = args.language
TIMING_FLAG = args.timing_flag
ATTRIBUTE_LIST_ITERATIVE = args.attribute_list_iterative

ANONYMIZE = args.anonymize
ATTACK = args.attack
ONLY_CORRECTNESS = args.only_correctness

if __name__=="__main__":

    if RESULTS_PATH is not None:
        PATH_TO_SAVE = RESULTS_PATH
    else:
        PATH_TO_SAVE = DATA_PATH

    if DATA_PATH is not None:
        DATA_FILE = f"{DATA_PATH}/level_{LEVEL}.jsonl"
    else:
        DATA_FILE = None

    profiles = load_data(DATA_FILE, SCENARIO, LEVEL)

    ############################## ANONYMIZE ##############################
    if ANONYMIZE:
        run_anonymization(profiles=profiles, anon_methods=ANON_METHODS, results_path=PATH_TO_SAVE,
                          scenario=SCENARIO, level=LEVEL, gemini_version=GEMINI_VERSION, llama_version=LLAMA_VERSION, gpt_version=GPT_VERSION,
                          epsilon=EPSILON, temperature=TEMPERATURE, attribute_list_iterative=ATTRIBUTE_LIST_ITERATIVE,
                          timing_flag=TIMING_FLAG)
            
    ############################## ATTACK ##############################

    if ATTACK:
        if ONLY_CORRECTNESS:
            only_check_correctness(profiles=profiles, anon_methods=ANON_METHODS, attacker_name=ATTACKER, scenario=SCENARIO,
                                   results_path=PATH_TO_SAVE, uniqueness_results_path=UNIQUENESS_RESULTS_FOLDER, level=LEVEL)
        else:
            attack(profiles=profiles, anon_methods=ANON_METHODS, attacker_name=ATTACKER, model_version=MODEL_VERSION,
                scenario=SCENARIO, results_path=PATH_TO_SAVE, uniqueness_results_path=UNIQUENESS_RESULTS_FOLDER, level=LEVEL, language=LANGUAGE)