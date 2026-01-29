import random

import numpy as np
from transformers import pipeline
from synthetic_data_generation.data import (
    get_dataset,
    get_feature_codes,
    read_metadata,
    get_data_entry,
    deserialize_entry,
)
from synthetic_data_generation.prompts import (
    create_generative_prompt,
    create_direct_identifiers_prompt,
)
from synthetic_data_generation.api import get_llm_response
from synthetic_data_generation.utils import (
    write_output,
    write_output_async,
    convert_entry_to_string,
)
from synthetic_data_generation.direct_identifiers import (
    generate_SSN,
    generate_card,
    generate_birthday,
)
from synthetic_data_generation.data import get_target_attributes_from_dataentry
from pii_benchmark.credentials import gemini_api_key, openai_api_key

from multiprocessing import Pool
import argparse
import os
import time

import pandas as pd

# Parse Input Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed_dataset",
    type=str,
    default="PUMS",
)
parser.add_argument(
    "--seed_dataset_link", type=str
)
parser.add_argument(
    "--seed_dataset_metadata",
    type=str,
    default="./data/pums/PUMS2010_metadata.json",
)
parser.add_argument("--no_of_entries", type=int, default=1)
parser.add_argument("--difficulty", type=int, default=1)
parser.add_argument("--scenario", type=str, default="Medical consultation")
parser.add_argument(
    "--n_processors",
    type=int,
    default=1,
    help="the number of available CPU processors",
)
parser.add_argument(
    "--output_file", type=str, default="./results/results.jsonl"
)
parser.add_argument("--requests_per_min", type=int, default=5)

parser.add_argument("--target_features", type=str, default=None)

parser.add_argument(
    "--direct_identifiers",
    type=str,
    default="name,email,phone number,SSN,address,credit card number",
)

parser.add_argument("--generate_identifiers", type=bool, default=True)

parser.add_argument("--llm", type=str, default="gemini")

parser.add_argument("--sample_features", type=int, default=5)

parser.add_argument("--sample_identifiers", type=int, default=1)

parser.add_argument("--target_language", type=str, default="English")

args = parser.parse_args()
SEED_DATASET = args.seed_dataset
SEED_DATASET_LINK = args.seed_dataset_link
SEED_METADATA = args.seed_dataset_metadata
NO_OF_ENTRIES = args.no_of_entries
SCENARIO = args.scenario
DIFFICULTY = args.difficulty
N_PROCESSORS = args.n_processors

OUTPUT_FILE = args.output_file
REQUESTS_PER_MIN = args.requests_per_min

FEATURES = get_feature_codes(SEED_DATASET)

SAMPLE_FEATURES = args.sample_features
SAMPLE_IDENTIFIERS = args.sample_identifiers

TARGET_LANGUAGE = args.target_language

print(f"sample_identifiers = {SAMPLE_IDENTIFIERS}")

ALL_DIRECT_IDENTIFIERS = (
    "name, email, phone number, address, SSN, credit card number"
)
ALL_DIRECT_IDENTIFIERS = [a.strip() for a in ALL_DIRECT_IDENTIFIERS.split(",")]

ALL_INDIRECT_IDENTIFIERS = (
    "RAC2P, CIT, ST, OCCP, MAR, SEX, ESR, SCHL, DOB"
)
ALL_INDIRECT_IDENTIFIERS = [a.strip() for a in ALL_INDIRECT_IDENTIFIERS.split(",")]

if SAMPLE_IDENTIFIERS==0:
    GENERATE_IDENTIFIERS_FLAG = False
else:
    GENERATE_IDENTIFIERS_FLAG = args.generate_identifiers

print(f"Generate Identifiers Flag = {GENERATE_IDENTIFIERS_FLAG}")

LLM = args.llm
if LLM == "gemini":
    API_KEY = gemini_api_key
elif LLM == "chatgpt":
    API_KEY = openai_api_key
else:
    API_KEY = ""

LLM = args.llm
if LLM == "gemini":
    API_KEY = gemini_api_key
elif LLM == "chatgpt":
    API_KEY = openai_api_key
else:
    API_KEY = ""


def generate_email (seed_dataset, dataentry):
    prompt = create_direct_identifiers_prompt(
        seed_dataset,
        ["email"],
        convert_entry_to_string(dataentry),
    )
    response = get_llm_response(prompt, API_KEY, LLM)
    print("Response for email generation:")
    print(response)
    response = response.splitlines()

    outdataentry = dict()
    # Add them to the data entry
    for line in response:
        line = line.split(": ")
        outdataentry[line[0]] = line[1]
    return outdataentry["email"]

# Generate one synthetic test record
def process_record(local_args):
    (
        cols,
        scenario,
        seed_dataset,
        identifiers,
        features,
        difficulty,
        dataentry,
        ground_truth,
        i,
    ) = local_args
    print("Generating Prompt...")

    # if "address" in identifiers:
    #     features.append("zip code")

    # Filter out just the target attributes
    dataentry = get_target_attributes_from_dataentry(
        dataentry, features, seed_dataset
    )

    if "email" in identifiers:
        email = generate_email(seed_dataset, dataentry)
        ground_truth["email"] = email
        dataentry["email"] = email

    # Create LLM Prompt
    prompt, selected_scenario = create_generative_prompt(
        scenario,
        seed_dataset,
        # remove zip code from list of records to leak (passed to the generator)
        [f for f in features if f!="zip code"],
        difficulty,
        convert_entry_to_string(dataentry),
        language=TARGET_LANGUAGE,
    )

    print("Sending API Request...")

    # Get LLM response
    response = get_llm_response(prompt, API_KEY, LLM)
    # print(response)

    # Format output
    output_dict = dict()
    output_dict["id"] = i
    output_dict["profile"] = dataentry
    output_dict["direct_identifiers"] = dict()
    output_dict["indirect_identifiers"] = dict()
    for feature in features:
        if feature in identifiers:
            output_dict["direct_identifiers"][feature] = ground_truth[feature]
            if feature == "address":
                output_dict["indirect_identifiers"]["zip code"] = str(ground_truth["zip code"])
        else:
            output_dict["indirect_identifiers"][feature] = ground_truth[feature]
    output_dict["features"] = features
    output_dict["difficulty"] = difficulty
    output_dict["prompt"] = prompt
    output_dict["text"] = response
    output_dict["scenario"] = selected_scenario
    return output_dict


def main():
    t1 = time.time()
    print("Starting Synthetic Prompt Generation.")
    print(f"Features = {FEATURES}")
    print(f"Generate identifiers = {args.generate_identifiers}")

    # Prepare dataset
    cols = FEATURES
    cols.append("identifiers")
    print(f"Columns used from dataset: {cols}")
    # if "address" in FEATURES:
    df = get_dataset(SEED_DATASET_LINK)
    df["zip code"] = df["zip code"].astype(str)

    # Get data entries for seeding prompts
    data_records = get_data_entry(
        dataset_link=None,
        dataset=df,
        no_of_entries=NO_OF_ENTRIES,
        columns=cols,
    )

    # print(data_records)

    # Generate synthetic test records
    record_args = list()
    for i in range(len(data_records)):
        if SAMPLE_IDENTIFIERS > 0:
            direct_identifiers = random.sample(
                ALL_DIRECT_IDENTIFIERS, SAMPLE_IDENTIFIERS
            )
        elif SAMPLE_IDENTIFIERS==-1:
            direct_identifiers = ALL_DIRECT_IDENTIFIERS
        ## if SAMPLE_IDENTIFIERS==0 do not use any direct identifiers
        else:
            direct_identifiers = []
        
        if SAMPLE_FEATURES == 5:
            # Get indirect identifiers directly from dataset
            indirect_identifiers = data_records[i][1].pop("identifiers")
            indirect_identifiers = indirect_identifiers.split(",")
        else:
            indirect_identifiers = random.sample(
                ALL_INDIRECT_IDENTIFIERS, SAMPLE_FEATURES
            )
        print(f"Generating record {i} with direct identifiers: {direct_identifiers} and indirect identifiers: {indirect_identifiers}")
        
        features = indirect_identifiers + direct_identifiers
        if "address" in direct_identifiers:
            print("address in direct identifiers, appending zip code to features")
            features.append("zip code")
            print(f"{features=}")
        
        record = {k: data_records[i][0][k] for k in data_records[i][0]}
        print(f"record = {record}")
        
        print(f"record with identifiers = {record}")
        record_args.append(
            [
                cols,
                SCENARIO,
                SEED_DATASET,
                direct_identifiers,
                features,
                DIFFICULTY,
                record,
                data_records[i][1],
                i,
            ]
        )
    
    
    print("Starting processors")
    curr_processed = 0
    requests_this_minute = 0
    timestamp = time.time()
    synthetic_entries = list()
    while curr_processed < len(record_args):
        currtime = time.time()
        if currtime - timestamp >= 60:
            requests_this_minute = 0
            timestamp = currtime
        to_process = min(N_PROCESSORS, REQUESTS_PER_MIN - requests_this_minute)
        to_process = max(to_process, 0)
        process_sublist = record_args[
            curr_processed : curr_processed + to_process
        ]
        if len(process_sublist) > 0:
            with Pool(N_PROCESSORS) as p:
                synthetic_entries = synthetic_entries + p.map(
                    process_record, process_sublist
                )
            requests_this_minute = requests_this_minute + to_process
            curr_processed = curr_processed + to_process
        else:
            time.sleep(timestamp + 60 - time.time())

    # Write Output
    print("Writing to File")

    def find_numpy_scalars(obj, path="$"):
        hits = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                hits += find_numpy_scalars(v, f"{path}[{k!r}]")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                hits += find_numpy_scalars(v, f"{path}[{i}]")
        elif isinstance(obj, tuple):
            for i, v in enumerate(obj):
                hits += find_numpy_scalars(v, f"{path}[{i}]")
        else:
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                hits.append((path, type(obj), obj))
        return hits

    # usage:
    for entry in synthetic_entries:
        hits = find_numpy_scalars(entry)
        for p, t, v in hits:
            print(p, t, v)

    write_output(OUTPUT_FILE, synthetic_entries)
    t2 = time.time()
    print(f"Total Time Taken: {t2 - t1} seconds")


if __name__ == "__main__":
    main()
