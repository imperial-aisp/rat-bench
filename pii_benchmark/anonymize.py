import json
from pathlib import Path
from typing import List
from tqdm import tqdm

from pii_benchmark.anonymizers.get_anonymizers import get_anonymizer
import argparse
import time

from synthetic_data_generation.utils import write_output_async

def run_anonymization(profiles: List[dict], anon_methods:List[str], results_path:str, scenario:str, level:int,
                      gemini_version:str|None=None, llama_version:str|None=None, gpt_version:str|None=None,
                      epsilon:int|None=None, temperature:int|None=None, attribute_list_iterative:str|None=None,
                      timing_flag:bool=True):
    print("Anonymizing")
    print(f"Will save results to {results_path}")
    llama_idx = None

    anonymizers = [
        get_anonymizer(
            method=anon_method,
            gemini_version=gemini_version,
            llama_version=llama_version,
            gpt_version=gpt_version,
            scenario=scenario,
            epsilon=epsilon,
            temperature=temperature,
            attribute_list_iterative=attribute_list_iterative
        )
        if anon_method not in ["llama_basic", "llama_full", "llama", "llama_clio", "llama_rescriber"]
        else None
        for anon_method in anon_methods
    ]

    if (
        "llama" in anon_methods
        or "llama_basic" in anon_methods
        or "llama_full" in anon_methods
        or "llama_clio" in anon_methods
        or "llama_rescriber" in anon_methods
    ):
        anonymizers.append(
            get_anonymizer(method="llama", llama_version=LLAMA_VERSION, scenario=SCENARIO)
        )
        llama_idx = len(anonymizers) - 1

    llama_attributes = {
        "llama": [
            "SSN",
            "phone number",
            "credit card number",
            "email",
            "name",
            "address",
        ],
        "llama_full": [
            "SSN",
            "phone number",
            "credit card number",
            "email",
            "name",
            "address",
            "race",
            "citizenship status",
            "educational attainment",
            "state of residence",
            "occupation",
            "marital status",
            "employment status",
            "date of birth",
            "age",
        ],
    }

    output_base = Path(results_path)
    output_dir = output_base.parent if output_base.suffix == ".jsonl" else output_base
    output_file = output_base if output_base.suffix == ".jsonl" else output_base / f"level_{level}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, anonymizer in enumerate(anon_methods):
        print(f"Anonymizing with {anon_methods[i]}")
        if anon_methods[i] == "llama":
            anonymizer = anonymizers[-1]
            for profile in tqdm(profiles):
                if timing_flag:
                    start_time = time.perf_counter()
                anon_text = anonymizer.anonymize(
                    profile["text"],
                    prompt_type="anthropic_attributes",
                    attributes=llama_attributes["llama"],
                )
                if timing_flag:
                    end_time = time.perf_counter()
                    profile[f"runtime_llama"] = end_time - start_time
                profile[f"text_anon_llama"] = anon_text
            write_output_async(output_file, profiles)
        
        elif anon_methods[i] == "llama_full":
            anonymizer = anonymizers[-1]
            for profile in tqdm(profiles):
                if timing_flag:
                    start_time = time.perf_counter()
                anon_text = anonymizers[llama_idx].anonymize(
                    profile["text"],
                    prompt_type="anthropic_attributes",
                    attributes=llama_attributes["llama_full"],
                )
                if timing_flag:
                    end_time = time.perf_counter()
                    profile[f"runtime_llama_full"] = end_time - start_time
                profile[f"text_anon_llama_full"] = anon_text
            write_output_async(output_file, profiles)
        
        elif anon_methods[i] == "llama_basic":
            anonymizer = anonymizers[-1]
            for profile in tqdm(profiles):
                if timing_flag:
                    start_time = time.perf_counter()
                anon_text = anonymizers[llama_idx].anonymize(
                    profile["text"], prompt_type="anthropic"
                )
                if timing_flag:
                    end_time = time.perf_counter()
                    profile[f"runtime_llama_basic"] = end_time - start_time
                profile[f"text_anon_llama_basic"] = anon_text
            write_output_async(output_file, profiles)

        elif anon_methods[i]=="llama_rescriber":
            anonymizer = anonymizers[-1]
            for profile in tqdm(profiles):
                if timing_flag:
                    start_time = time.perf_counter()
                anon_text = anonymizers[llama_idx].anonymize(
                    profile["text"], prompt_type="rescriber"
                )
                if timing_flag:
                    end_time = time.perf_counter()
                    profile[f"runtime_llama_rescriber"] = end_time - start_time
                profile[f"text_anon_llama_rescriber"] = anon_text
            write_output_async(output_file, profiles)

        elif anon_methods[i]=="llama_clio":
            anonymizer = anonymizers[-1]
            for profile in tqdm(profiles):
                if timing_flag:
                    start_time = time.perf_counter()
                anon_text = anonymizers[llama_idx].anonymize(
                    profile["text"], prompt_type="clio"
                )
                if timing_flag:
                    end_time = time.perf_counter()
                    profile[f"runtime_llama_clio"] = end_time - start_time
                profile[f"text_anon_llama_clio"] = anon_text
            write_output_async(output_file, profiles)

        elif anon_methods[i] == "iterative":
            anonymizer = anonymizers[i]
            print(f"Iterative anonymizer, attribute list = {attribute_list_iterative}")
            for profile in tqdm(profiles):
                if timing_flag:
                    start_time = time.perf_counter()
                anon_text = anonymizer.anonymize(profile)
                if timing_flag:
                    end_time = time.perf_counter()
                    profile[f"runtime_{anon_methods[i]}_{attribute_list_iterative}"] = end_time - start_time
                profile[f"text_anon_{anon_methods[i]}_{attribute_list_iterative}"] = anon_text
            write_output_async(output_file, profiles)
        
        elif anon_methods[i]=="llama_clio":
            anonymizer = anonymizers[i]
            for profile in tqdm(profiles):
                if timing_flag:
                    start_time = time.perf_counter()
                anon_text = anonymizer.anonymize(profile)
                if timing_flag:
                    end_time = time.perf_counter()
                    profile[f"runtime_{anon_methods[i]}"] = end_time - start_time
                profile[f"text_anon_{anon_methods[i]}"] = anon_text
            write_output_async(output_file, profiles)

        elif anon_methods[i]=="madlib" or anon_methods[i]=="tem":
            # make sure we write epsilon to the output
            anonymizer = anonymizers[i]
            for profile in tqdm(profiles):
                if timing_flag:
                    start_time = time.perf_counter()
                anon_text = anonymizer.anonymize(profile["text"])
                if timing_flag:
                    end_time = time.perf_counter()
                    profile[f"runtime_{anon_methods[i]}_eps{EPSILON}"] = end_time - start_time
                profile[f"text_anon_{anon_methods[i]}_eps{EPSILON}"] = anon_text
            write_output_async(output_file, profiles)
                
        elif anon_methods[i]=="dp_prompt_gpt":
            # make sure we write temperature to the output
            anonymizer = anonymizers[i]
            for profile in tqdm(profiles):
                if timing_flag:
                    start_time = time.perf_counter()
                anon_text = anonymizer.anonymize(profile["text"])
                if timing_flag:
                    end_time = time.perf_counter()
                    profile[f"runtime_{anon_methods[i]}_temp{TEMPERATURE}"] = end_time - start_time
                profile[f"text_anon_{anon_methods[i]}_temp{TEMPERATURE}"] = anon_text
            write_output_async(output_file, profiles)
        
        else:
            anonymizer = anonymizers[i]
            for profile in tqdm(profiles):
                if timing_flag:
                    start_time = time.perf_counter()
                anon_text = anonymizer.anonymize(profile["text"])
                if timing_flag:
                    end_time = time.perf_counter()
                    profile[f"runtime_{anon_methods[i]}"] = end_time - start_time
                profile[f"text_anon_{anon_methods[i]}"] = anon_text
            write_output_async(output_file, profiles)

    write_output_async(output_file, profiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--gemini_version", type=str, default="2.5-flash")
    parser.add_argument("--llama_version", type=str, default="3.1-8B-Instruct")
    parser.add_argument("--gpt_version", type=str, default="gpt-4o-mini")
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--anon_methods", type=str, default=None)
    parser.add_argument("--results_path", type=str, default="")
    parser.add_argument("--scenario", type=str, default="medical_data")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--timing", type=int, default=0)
    parser.add_argument("--attribute_list", type=str, default="ours")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    ANON_METHODS = args.anon_methods
    ANON_METHODS = [s.strip() for s in ANON_METHODS.split(",")]
    GEMINI_VERSION = args.gemini_version
    LLAMA_VERSION = args.llama_version
    GPT_VERSION = args.gpt_version
    EPSILON = args.epsilon
    TEMPERATURE = args.temperature
    RESULTS_PATH = args.results_path
    SCENARIO = args.scenario
    LEVEL = args.level
    TIMING_FLAG = args.timing

    profiles = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            profiles.append(json.loads(line))

    run_anonymization(
        profiles,
        ANON_METHODS,
        RESULTS_PATH,
        SCENARIO,
        LEVEL,
        gemini_version=GEMINI_VERSION,
        llama_version=LLAMA_VERSION,
        gpt_version=GPT_VERSION,
        epsilon=EPSILON,
        temperature=TEMPERATURE,
        attribute_list_iterative=args.attribute_list,
        timing_flag=TIMING_FLAG
    )
