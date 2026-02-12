import argparse
import os
import re
from typing import List, Union, Tuple
from nltk.translate import bleu
import Levenshtein
from nltk.translate.bleu_score import SmoothingFunction
import pickle
import json
import fcntl
import random
import time
from datasets import load_dataset

PUMS_MAPS_PATH = "./data/pums/maps/"

# Mapping from PUMS col name to human-readable
pums_col_to_str_dict = {
    ## PUMS ATTRIBUTES INCLUDED IN OUR PIPELINE
    "SEX": "sex",
    "RAC2P": "race",
    "CIT": "citizenship status",
    "SCHL": "education level",
    "ESR": "employment status",
    "ST": "state of residence",
    "DOB": "date of birth",
    "MAR": "marital status",
    "OCCP": "occupation",
}

# PUMS col names
pums_cols = list(pums_col_to_str_dict.keys())

# Mapping from human-readable names to PUMS cols
str_to_pums_col_dict = {v:k for k,v in pums_col_to_str_dict.items()}

att_names = [
    # indirect identifiers
    "sex",
    "race",
    "citizenship status",
    "education level",
    "occupation",
    "marital status",
    "employment status",
    "state of residence",
    "DOB",
    "date of birth",
    # direct identifiers
    "phone number",
    "SSN",
    "email",
    "credit card number",
    "name",
    "address",
    "zip code",
    # iterative anonymizer EXTRA attributes
    "Income",
    "Hearing difficulty",
    "Mobility status (lived here 1 year ago)",
    "Ability to speak English",
    "Gave birth to child within the past 12 months",
    "Insurance through a current or former employer or union",
    "Interest, dividends, and net rental income past 12 months",
    "Travel time to work",
    "Language other than English spoken at home",
    "Number of times married",
    # iterative anonymizer PRESIDIO attributes
    "Credit card number",
    "Crypto wallet number",
    "Date or time",
    "email address",
    "IBAN code",
    "IP address",
    "Nationality, religious or political group",
    "Location",
    "Full person name",
    "Phone number",
    "Medical license",
    "URL",
    "US bank number",
    "US drivers license",
    "US Individual Taxpayer Identification Number",
    "US passport number",
    "US SSN"
]


def get_pums_values(attribute: str) -> str:
    with open(f"{PUMS_MAPS_PATH}{attribute}_map.pickle", "rb") as f:
        map = pickle.load(f)
    s = ""
    for r in list(map.values()):
        s += r + "; "
    s = s[:-2]
    return s

def pums_col_to_str(pii_type: str) -> str:
    if pii_type in pums_col_to_str_dict:
        return pums_col_to_str_dict[pii_type]
    else:
        return pii_type

def type_to_options(pii_type: str, scenario: str = None) -> str:
    ## PUMS
    if pii_type == "DOB":
        pii_type_str = "use the date of birth of the individual, in the DD Month YYYY format (for example 10 January 1990 or 3 June 2020)"
    elif pii_type in pums_cols:
        pii_type_str = (
            f"choose from these options: {get_pums_values(pii_type)}"
        )
    elif pii_type == "phone number" or pii_type == "phone":
        pii_type_str = "provide the US phone number of the individual, in the format (XXX)XXX-XXXX "
    elif pii_type == "credit card number" or pii_type == "credit card":
        pii_type_str = "provide the credit card number of the individual, in the format XXXX-XXXX-XXXX-XXXX if it is 16 digits long, or XXXX-XXXXXX-XXXXX if it is 15 digits long "
    elif pii_type == "SSN":
         pii_type_str = "provide the SSN of the individual, in the format XXX-XX-XXXX"
    else:
        pii_type_str = ""

    return pii_type_str


def str_is_close(a: str, b: str, min_sim: float = 0.75, strict=True) -> bool:
    if strict:
        return Levenshtein.jaro_winkler(a, b) >= min_sim
    else:
        split_a = a.split(" ")
        if len(split_a) == 1:
            return Levenshtein.jaro_winkler(a, b) > min_sim
        elif len(split_a) > 4:
            return False
        else:
            for split in split_a:
                if Levenshtein.jaro_winkler(split, b) > min_sim:
                    return True

    return False


def select_closest(
    input_str: str,
    target_strings: List[str],
    dist: str = "jaro_winkler",
    return_sim: bool = False,
    return_idx: bool = False,
) -> Union[str, Tuple[str, float]]:
    best_sim = 0.0
    selected_str = ""
    selected_idx = -1

    for i, t_str in enumerate(target_strings):
        if dist == "jaro_winkler":
            sim = Levenshtein.jaro_winkler(input_str, t_str)
        elif dist == "levenshtein":
            sim = Levenshtein.distance(input_str, t_str)
        elif dist == "bleu":
            sim = bleu(
                [input_str.split(" ")],
                t_str.split(" "),
                smoothing_function=SmoothingFunction().method4,
            )
        if sim > best_sim:
            best_sim = sim
            selected_str = t_str
            selected_idx = i

    ret_val = selected_str

    if return_sim:
        ret_val = selected_str, best_sim
        if return_idx:
            ret_val = selected_str, best_sim, selected_idx
    elif return_idx:
        ret_val = selected_str, selected_idx

    return ret_val


def str_is_close_any(a: str, b: List[str], min_sim: float = 0.75) -> bool:
    for b_str in b:
        if str_is_close(a, b_str, min_sim):
            return True
    return False

def get_att_key(att: str) -> str:
    ## return pums col for indirect attributes, no change for direct attributes
    if att in str_to_pums_col_dict:
        return str_to_pums_col_dict[att]
    else:
        return att

def parse_output_gpt(response):
    output_lines = response.splitlines()

    guess_dict = dict()

    i = 0

    while i<len(output_lines):
        curr_line = output_lines[i]
        if curr_line=="{" or curr_line=="" or curr_line=="},":
            i += 1
        else:
            elements = curr_line.split(":")
            key = elements[0].strip(' ",\'')
            if key in att_names:
                    
                curr_guess_dict = dict()
                
                ## i+1
                j = i+1
                next_line_els = output_lines[j].split(":",1)
                if next_line_els[0].strip(' ",\'').lower()=="inference":
                    curr_guess_dict["Inference"] = next_line_els[1].strip(' ",\'')
                elif next_line_els[0].strip(' ",\'').lower()=="guess":
                    curr_guess_dict["Guess"] = next_line_els[1].strip(' ",\'')
                elif next_line_els[0].strip(' ",\'').lower()=="certainty":
                    curr_guess_dict["Certainty"] = next_line_els[1].strip(' ",\'')
                
                ## i+2
                j = i+2

                next_line_els = output_lines[j].split(":",1)
                if next_line_els[0].strip(' ",\'').lower()=="inference":
                    curr_guess_dict["Inference"] = next_line_els[1].strip(' ",\'')
                elif next_line_els[0].strip(' ",\'').lower()=="guess":
                    curr_guess_dict["Guess"] = next_line_els[1].strip(' ",\'')
                elif next_line_els[0].strip(' ",\'').lower()=="certainty":
                    curr_guess_dict["Certainty"] = next_line_els[1].strip(' ",\'')
                ## i+3
                j = i+3

                next_line_els = output_lines[j].split(":",1)
                if next_line_els[0].strip(' ",\'').lower()=="inference":
                    curr_guess_dict["Inference"] = next_line_els[1].strip(' ",\'')
                elif next_line_els[0].strip(' ",\'').lower()=="guess":
                    curr_guess_dict["Guess"] = next_line_els[1].strip(' ",\'')
                elif next_line_els[0].strip(' ",\'').lower()=="certainty":
                    curr_guess_dict["Certainty"] = next_line_els[1].strip(' ",\'')
                guess_dict[get_att_key(key)] = curr_guess_dict
                i += 4
            else:
                i += 1
    return guess_dict
     
def parse_output(response):
    output_lines = response.splitlines()

    guess_dict = dict()

    for i in range(len(output_lines)):
        curr_line = output_lines[i]

        if curr_line == "{" or curr_line == "}":
            continue
        line_items = curr_line.split(":")

        curr_att = line_items[0].strip().strip('"')

        if curr_att in att_names:
            indiv_guess = dict()
            
            try:
                inf = [
                    o.strip().strip('"') for o in output_lines[i + 1].split(":")
                ]
            except:
                print("No inference specified:")
                for ii in range(i, len(output_lines)):
                    print(output_lines[ii])
                print()
                inf = " "
            try:
                g = [
                    o.strip().strip('"') for o in output_lines[i + 2].split(":")
                ]
            except:
                print("No guess specified:")
                for ii in range(i, len(output_lines)):
                    print(output_lines[ii])
                print()
                g = " "
            try:
                cert = [
                    o.strip().strip('"') for o in output_lines[i + 3].split(":")
                ]
            except:
                print("No certainty specified:")
                for ii in range(i, len(output_lines)):
                    print(output_lines[ii])
                print()
                cert = " "

            if inf[0] == "Inference" or inf[0] == "inference":
                indiv_guess["Inference"] = inf[1]
            if g[0] == "Guess" or g[0] == "guess":
                indiv_guess["Guess"] = g[1]
            if (
                cert[0] == "Certainty"
                or cert[0] == "certainty"
                or cert[0] == "Certainly"
                or cert[0] == "certainly"
            ):
                if len(cert)>1:
                    indiv_guess["Certainty"] = cert[1]
                else:
                    indiv_guess["Certainty"] = ""
                    print(cert)
            if "Guess" not in indiv_guess:
                indiv_guess["Guess"] = ""

            guess_dict[get_att_key(curr_att)] = indiv_guess
    return guess_dict


def fix_and_load_json(s: str):
    """
    Try to parse a JSON string. If invalid, attempt to fix common issues:
    - Missing outer braces
    - Trailing commas
    - Extra commas before closing braces
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Wrap in braces if it looks like a dict fragment
        stripped = s.strip()
        if not stripped.startswith("{"):
            s = "{\n" + s
        if not stripped.endswith("}"):
            s = s + "\n}"

        # Remove trailing commas before } or ]
        s = re.sub(r",\s*([}\]])", r"\1", s)

        fixed_str = ""
        guesses_lines = s.splitlines()

        for l in range(len(guesses_lines) - 1):
            fixed_line = guesses_lines[l]
            if len(fixed_line) > 0:
                if (
                    fixed_line[0] != "{"
                    and fixed_line[0] != "}"
                    and fixed_line[-1] != "{"
                    and fixed_line[-1] != "}"
                ):
                    if fixed_line[-1] != "," and guesses_lines[l + 1][0] != "}":
                        fixed_line += ","
            fixed_str += fixed_line + "\n"
        # Try again
        try:
            return json.loads(fixed_str)
        except json.JSONDecodeError as e:
            # Try again
            try:
                return json.loads(fixed_str + "}")
            except:
                try:
                    return json.loads(fixed_str + "}}")
                except:
                    print(
                        f"Could not fix JSON:\n--- Fixed candidate string ---\n{s}"
                    )
                    return fixed_str


def load_data(data_path, scenario, difficulty):
    profiles = []
    
    with open(data_path) as f:
        for l in f:
            profiles.append(json.loads(l))
    return profiles

# Write synthetic records to output file.
def write_output(filepath, dataentries, fields_to_save=None):

    if fields_to_save is None:
        fields_to_save = dataentries[0].keys()

    ## check if the file already exists

    if os.path.exists(filepath):
        profiles = []
        # if the file already exists, load the json file first
        with open(filepath, "r") as f:
            for l in f:
                profiles.append(json.loads(l))
        for i,profile in enumerate(profiles):
            fields_to_save = dataentries[i].keys()
            for k in fields_to_save:
                profile[k] = dataentries[i][k]
        with open(filepath, "w") as outfile:
            for entry in profiles:
                print(json.dumps(entry), file=outfile)
    else:
        with open(filepath, "w") as outfile:
            for entry in dataentries:
                print(json.dumps(entry), file=outfile)
    return None

def str2bool(s):
    # This is for boolean type in the parser
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
