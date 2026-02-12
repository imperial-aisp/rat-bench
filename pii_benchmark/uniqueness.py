import argparse
from pathlib import Path
import pickle
import json
from typing import List
import correctmatch
import pandas as pd
from tqdm import tqdm
import numpy as np

PUMS_MAPS_PATH = "./data/pums/maps/{col}_map.pickle"
PUMS_DOB_MAP_PATH = "./data/pums/maps/{col}_map.pickle"

### special maps for occupation (type and description) ###
with open("data/pums/maps/OCCP_type_map.pickle", "rb") as f:
    PUMS_OCCUPATION_TYPES = pickle.load(f)

PUMS_OCCUPATION_TYPE_TO_CODE = dict()

for k, v in PUMS_OCCUPATION_TYPES.items():
    PUMS_OCCUPATION_TYPE_TO_CODE[v] = k

############################################################
    
pums_cols = [
    "AGEP",
    "RAC2P",
    "CIT",
    "SCHL",
    "POWSP",
    "JWDP",
    "OCCP",
    "COW",
    "MAR",
    "MIL",
    "SEX",
    "JWTRNS",
    "ESR",
    "ST",
    "DRIVESP",
    "DOB",
    "DOB-Day",
    "DOB-Month",
    "DOB-Year",
    "zip code"
]

PUMS_DF = pd.read_csv("data/pums/pums_pwgtp_sample_.csv")
N = 309349689 
print("data loaded")

def process_col(c, guess_correctness, ground_truth, cols_to_fit):

    if c not in guess_correctness:
        # if there is no field corresponding to the column in the guesses dict, return None
        print("no field for column ", c)
        return None, cols_to_fit
    
    if guess_correctness[c] is None or len(guess_correctness[c]) == 0:
        # if the correctness is None (i.e. the guess was None), return None
        print("none guess for column ", c)
        return None, cols_to_fit
    
    if guess_correctness[c][0] == 1 and c in pums_cols:
        # if the guess is correct, process it

        # load map
        if "DOB" in c:
            with open(PUMS_DOB_MAP_PATH.format(col=c), "rb") as f:
                map = pickle.load(f)
        elif c == "zip code":
            with open("data/pums/maps/zip_to_puma.pickle", "rb") as f:
                zip_to_puma_map = pickle.load(f)
            with open("data/pums/maps/PUMA_FULL_map.pickle", "rb") as f:
                puma_map = pickle.load(f)
            inverse_map = {v:k for k,v in puma_map.items()}
            zip_code = ground_truth["zip code"]
            puma_full = zip_to_puma_map[int(zip_code)]
            puma = inverse_map[puma_full]
            cols_to_fit.append("PUMA_FULL")
            print(f"{cols_to_fit=}")
            return int(puma), cols_to_fit
        else:
            with open(PUMS_MAPS_PATH.format(col=c), "rb") as f:
                map = pickle.load(f)
        inverse_map = dict()
        for k, v in map.items():
            inverse_map[v] = k

        if c == "OCCP":
            cols_to_fit.append(c)
            dloc = ground_truth[c].find("DESCRIPTION")
            occp_type = ground_truth[c][:dloc].split(":")[-1].strip().strip(",")
            occp_desc = ground_truth[c][dloc:].split(":")[-1].strip()

            if occp_type=="N/A":
                val = "1"
            else:

                f = PUMS_OCCUPATION_TYPE_TO_CODE[occp_type] + "-" + occp_desc
                if f == "OFF-FIRST":
                    val = "25"
                elif f == "SAL-FIRST":
                    val = "32"
                elif f == "RPR-FIRST":
                    val = "117"
                elif f == "SAL-DOOR":
                    val = "158"
                elif f == "FLIGHT ATTENDANTS":
                    val = "260"
                elif (
                    f == "CON-MISCELLANEOUS EXTRACTION WORKERS, INCLUDING ROOF BOLTERS AND HELPERS"
                ):
                    val = "346"
                else:
                    val = inverse_map[f]
        elif c == "AGEP":
            cols_to_fit.append(c)
            val = int(ground_truth[c])
        elif (c == "DOB-Day" or c == "DOB-Month" or c == "DOB-Year"):
            if (guess_correctness["DOB"][0]!=1):
                cols_to_fit.append(c)
                date_of_birth = ground_truth["DOB"]
                components = date_of_birth.split(" ")
                if c == "DOB-Day":
                    val = inverse_map[int(components[0])]
                elif c == "DOB-Month":
                    val = inverse_map[components[1]]
                else:
                    val = inverse_map[int(components[2])]
            else:
                return None, cols_to_fit
        elif c == "zip code":
            # handled above
            pass
        else:
            cols_to_fit.append(c)
            val = inverse_map[ground_truth[c]]
        return int(val), cols_to_fit
    else:
        return None, cols_to_fit

def compute_correctness(model, record_to_analyze: List[int], n: int | None):
    if n is None:
        n = N

    correctness_list = []
    uniqueness_list = []

    for i in range(10):
        correctness = correctmatch.individual_correctness(
            model,
            np.array(record_to_analyze),
            n=n
        )
        iu = correctmatch.individual_uniqueness(
            model,
            np.array(record_to_analyze),
            n=n
        )
        correctness_list.append(correctness)
        uniqueness_list.append(iu)
    correctness = np.mean([c for c in correctness_list if not np.isnan(c)])
    iu = np.mean([u for u in uniqueness_list if not np.isnan(u)])
    
    return correctness, iu

def fit_model_and_compute(cols_to_fit: List[str], record_to_analyze: List[int], df: pd.DataFrame | None, n: int | None):
    if df is None:
        dff = PUMS_DF[cols_to_fit]
    else:
        dff = df[cols_to_fit]

    if n is None:
        n = N

    model = correctmatch.fit_model(dff.to_numpy())

    correctness_list = []
    uniqueness_list = []

    for i in range(10):
        
        correctness = correctmatch.individual_correctness(
            model,
            np.array(record_to_analyze),
            n=n
        )
        iu = correctmatch.individual_uniqueness(
            model,
            np.array(record_to_analyze),
            n=n
        )

        correctness_list.append(correctness)
        uniqueness_list.append(iu)

    
    correctness = np.mean([c for c in correctness_list if not np.isnan(c)])
    iu = np.mean([u for u in uniqueness_list if not np.isnan(u)])
    
    return correctness, iu, model


def compute_reid_risk(profiles, methods, attacker, results_path):
    df = pd.read_csv("data/pums/pums_pwgtp_sample_.csv")
    n = 306169200


    n_correct_atts = {k: [] for k in methods}
    correctness_dict = {k: [] for k in methods}
    uniqueness_dict = {k: [] for k in methods}

    correct_direct_ids = {k: [] for k in methods}

    trained_models = {}

    for ii, profile in tqdm(enumerate(profiles)):
        print(f"Analyzing profile {ii}")

        for anon_method in methods:
            print(f"Analyzing anon method {anon_method}")
            cols_to_fit = []
            record_to_analyze = {}

            if attacker != "False":
                crctnss = profile[f"correctness_{anon_method}_{attacker}"]
            else:
                crctnss = profile[f"correctness_{anon_method}"]
            ids_list = []

            for c in crctnss.keys():
                if c in pums_cols:
                    # Indirect identifiers
                    val, cols_to_fit = process_col(
                        c, crctnss, profile["indirect_identifiers"], cols_to_fit
                    )
                    if val is not None:
                        if c=="zip code":
                            record_to_analyze["PUMA_FULL"] = val
                        else:
                            record_to_analyze[c] = int(val)

                else:
                    # Direct identifiers
                    c_guess = crctnss[c]
                    if c_guess is not None and len(c_guess) > 0:
                        if crctnss[c][0] == 1:
                            ids_list.append(c)

            ## creating record
            record = []
            ordered_cols = []

            # sort columns in records
            for c in pums_cols:
                if c=="zip code":
                    if "PUMA_FULL" in record_to_analyze:
                        record.append(record_to_analyze["PUMA_FULL"])
                        ordered_cols.append("PUMA_FULL")
                else:
                    if c in record_to_analyze:
                        record.append(record_to_analyze[c])
                        ordered_cols.append(c)

            if len(cols_to_fit) > 0 and len(record) > 0:

                cols_for_model = []

                for oc in ordered_cols:
                    cols_for_model.append(f"{oc}")

                print(f"record = {record}")
                print(f"cols_for_model = {cols_for_model}")

                if "/".join(cols_for_model) in trained_models:
                    model = trained_models["/".join(cols_for_model)]

                    correctness, iu = compute_correctness(model, record, n)
                else:
                    correctness, iu, fitted_model = fit_model_and_compute(
                        cols_for_model, record, df, n
                    )
                    trained_models["/".join(cols_for_model)] = fitted_model
            else:
                print(f"No cols to fit or empty record: {len(cols_to_fit)=}, {len(record)=}")
                correctness, iu = 0.0, 0.0
            n_correct_atts[anon_method].append(len(ordered_cols))
            correctness_dict[anon_method].append(correctness)
            uniqueness_dict[anon_method].append(iu)

            correct_direct_ids[anon_method].append(ids_list)

    results_dict = {
        "correctness": correctness_dict,
        "uniqueness": uniqueness_dict,
        "n_correct_atts": n_correct_atts,
        "correct_direct_ids": correct_direct_ids,
    }

    n_reided_per_method = dict()

    for m in results_dict["correct_direct_ids"].keys():
        n_reided = 0

        for i in range(len(results_dict["correctness"][m])):
            n_direct_ids = len(results_dict["correct_direct_ids"][m][i])
            correctness = results_dict["correctness"][m][i]
            if n_direct_ids > 0:
                n_reided += 1
            else:
                if correctness == 1:
                    n_reided += 1
        n_reided_per_method[m] = n_reided / len(profiles)

    results_dict["reidentification_rate"] = n_reided_per_method

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "wb") as f:
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--results_path", type=str)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--full_dataset", type=str, default="False")
    parser.add_argument("--attacker", type=str, default="False")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    RESULTS_PATH = args.results_path
    METHODS = args.methods
    FULL_DATASET = args.full_dataset
    ATTACKER = args.attacker

    profiles = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            profiles.append(json.loads(line))

    compute_reid_risk(profiles, METHODS, ATTACKER, RESULTS_PATH)
