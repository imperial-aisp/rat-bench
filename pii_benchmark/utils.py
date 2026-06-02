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


def retry_with_backoff(fn, *args, max_retries=10, base_delay=2, max_delay=60, **kwargs):
    """Call fn(*args, **kwargs), retrying with exponential backoff on any exception."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 1)
            print(f"[Retry {attempt}/{max_retries}] {e}. Waiting {wait:.1f}s...")
            time.sleep(wait)

PUMS_MAPS_PATH = "./data/maps/"
SRB_MAPS_PATH = "./data/srb/maps/"
NL_MAPS_PATH = "./data/nl/maps/"

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

# MEX census attribute codes -> Spanish labels (used in attack prompts)
MEX_COL_TO_STR = {
    "CLASE_VIV":          "tipo de vivienda",
    "SEXO":               "sexo",
    "EDAD":               "edad",
    "ENT_PAIS_NAC":       "entidad o pais de nacimiento",
    "DHSERSAL1":          "servicio de salud",
    "RELIGION":           "religion",
    "HLENGUA":            "habla lengua indigena",
    "HESPANOL":           "habla espanol",
    "ASISTEN":            "asistencia escolar",
    "NIVACAD":            "nivel academico",
    "SITUA_CONYUGAL":     "situacion conyugal",
    "HIJOS_NAC_VIVOS":    "hijos nacidos vivos",
    # direct identifiers
    "name":               "nombre",
    "phone number":       "numero de telefono",
    "address":            "domicilio",
    "email":              "correo electronico",
    "credit card number": "numero de tarjeta",
    "CURP":               "CURP",
}

# Reverse: Spanish label -> original key (maps parsed guesses back to ground-truth keys)
MEX_STR_TO_COL = {v: k for k, v in MEX_COL_TO_STR.items()}

def mex_col_to_str(pii_type: str) -> str:
    return MEX_COL_TO_STR.get(pii_type, pii_type)


# Serbian census column names -> English human-readable labels (used in attack prompts)
# Note: "marital status" and "date of birth" intentionally share labels with PUMS;
# evaluation handles the resulting key aliasing via fallback lookups.
SRB_COL_TO_STR = {
    "urban":          "urban status",
    "age":            "age",
    "marital_status": "marital status",
    "given_birth":    "ever given birth",
    "dob":            "date of birth",
    "dom":            "date of first marriage",
    "age_mar":        "age at first marriage",
    "partner_age":    "partner's age",
    "ethnicity":      "ethnicity",
    "language":       "language spoken",
    # direct identifiers
    "phone number":       "phone number",
    "credit card number": "credit card number",
    "email":              "email",
    "name":               "name",
    "address":            "address",
    "JMBG":               "JMBG",
}

# Reverse map (excludes labels that alias to PUMS keys to avoid get_att_key conflicts)
SRB_STR_TO_COL = {
    "urban status":            "urban",
    "age":                     "age",
    "ever given birth":        "given_birth",
    "date of first marriage":  "dom",
    "age at first marriage":   "age_mar",
    "partner's age":           "partner_age",
    "ethnicity":               "ethnicity",
    "language spoken":         "language",
    "JMBG":                    "JMBG",
    # "marital status" and "date of birth" are intentionally omitted here;
    # get_att_key will return the PUMS key ("MAR"/"DOB") for those labels,
    # and evaluation.py has fallback logic to handle both keys.
}

# Column names that have categorical maps (for building option lists in prompts)
SRB_COL_TO_MAPFILE = {
    "urban":          "urban_status_map",
    "marital_status": "marital_status_map",
    "given_birth":    "ever_given_birth_map",
    "ethnicity":      "ethnicity_map",
    "language":       "language_map",
}

def srb_col_to_str(pii_type: str) -> str:
    return SRB_COL_TO_STR.get(pii_type, pii_type)

def get_srb_values(attribute: str) -> str:
    mapfile = SRB_COL_TO_MAPFILE.get(attribute)
    if mapfile is None:
        return ""
    with open(f"{SRB_MAPS_PATH}{mapfile}.json", "r", encoding="utf-8") as f:
        d = json.load(f)
    return "; ".join(str(k) for k in d.keys())


# NL (Belgium/Flemish) column codes → English labels used in attack prompts
NL_COL_TO_STR = {
    "age":        "age",
    "sex":        "sex",
    "marstd":     "marital status",
    "nativity":   "nativity",
    "bplcountry": "country of birth",
    "nation":     "nationality",
    "educnl":     "education level",
    "empstatd":   "employment status",
    "labforce":   "labor force participation",
    "occisco":    "occupation",
    "indgen":     "industry",
    "dob":        "date of birth",
    # direct identifiers
    "name":               "name",
    "email":              "email",
    "phone number":       "phone number",
    "address":            "address",
    "RRN":                "RRN",
    "credit card number": "credit card number",
}

# Reverse: English label → NL column code
# Only labels not already covered by str_to_pums_col_dict (sex→SEX, marital status→MAR,
# education level→SCHL, employment status→ESR, occupation→OCCP, date of birth→DOB).
# Those aliases are handled via fallback in evaluation.py.
NL_STR_TO_COL = {
    "nativity":                  "nativity",
    "country of birth":          "bplcountry",
    "nationality":               "nation",
    "labor force participation":  "labforce",
    "industry":                  "indgen",
    "RRN":                       "RRN",
}

# NL columns that have categorical map files (stem = file name without _map.json)
NL_COL_TO_MAPFILE = {
    "sex":        "sex",
    "marstd":     "marital_status",
    "nativity":   "nativity",
    "bplcountry": "country",
    "nation":     "nation",
    "educnl":     "education",
    "empstatd":   "employment_status",
    "labforce":   "labor_force",
    "occisco":    "occupation",
    "indgen":     "industry",
}

# _NL_SKIP_VALUES = {"NIU (not in universe)", "Unknown/missing", "Unknown"}

def nl_col_to_str(pii_type: str) -> str:
    return NL_COL_TO_STR.get(pii_type, pii_type)

def get_nl_values(attribute: str) -> str:
    stem = NL_COL_TO_MAPFILE.get(attribute)
    if stem is None:
        return ""
    with open(f"{NL_MAPS_PATH}{stem}_map.json", "r", encoding="utf-8") as f:
        d = json.load(f)
    return "; ".join(str(k) for k in d.keys())
    # return "; ".join(str(k) for k in d.keys() if k not in _NL_SKIP_VALUES)


att_names = [
    # indirect identifiers (PUMS / general English)
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
    # Serbian indirect identifiers
    "age",
    "urban status",
    "ever given birth",
    "date of first marriage",
    "age at first marriage",
    "partner's age",
    "ethnicity",
    "language spoken",
    # direct identifiers
    "phone number",
    "SSN",
    "JMBG",
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
    "US SSN",
    # MEX Spanish labels (indirect identifiers)
    "tipo de vivienda",
    "sexo",
    "edad",
    "entidad o pais de nacimiento",
    "servicio de salud",
    "religion",
    "habla lengua indigena",
    "habla espanol",
    "asistencia escolar",
    "nivel academico",
    "situacion conyugal",
    "hijos nacidos vivos",
    # MEX Spanish labels (direct identifiers)
    "CURP",
    "nombre",
    "numero de telefono",
    "domicilio",
    "numero de tarjeta",
    "correo electronico",
    # NL (Flemish) indirect identifier labels
    "nativity",
    "country of birth",
    "nationality",
    "labor force participation",
    "industry",
    # NL direct identifier labels
    "RRN",
]


def get_pums_values(attribute: str) -> str:
    with open(f"{PUMS_MAPS_PATH}{attribute}_map.pickle", "rb") as f:
        map = pickle.load(f)
    s = ""
    for r in list(map.values()):
        s += r + "; "
    s = s[:-2]
    return s

def get_mex_values(attribute: str) -> str:

    with open(f"data/es/maps/{attribute}_map.json", "r") as f:
        map = json.load(f)
    s = ""
    for r in list(map.keys()):
        s += str(r) + "; "
    s = s[:-2]
    return s


def pums_col_to_str(pii_type: str) -> str:
    if pii_type in pums_col_to_str_dict:
        return pums_col_to_str_dict[pii_type]
    else:
        return pii_type

def type_to_options(pii_type: str, scenario: str = None, language: str = "English") -> str:
    # print(f"PII type is {pii_type}, scenario is {scenario}, language is |{language}")
    if language == "Flemish":
        if pii_type == "RRN":
            return "provide the Belgian rijksregisternummer of the individual in the format YY.MM.DD-NNN.CC"
        elif pii_type == "phone number":
            return "provide the Belgian mobile phone number of the individual in the format 04XX XX XX XX"
        elif pii_type == "credit card number":
            return "provide the credit card number in the format XXXX-XXXX-XXXX-XXXX if it is 16 digits long, or XXXX-XXXXXX-XXXXX if it is 15 digits long"
        elif pii_type in ("dob", "date of birth"):
            return "use the date of birth in the DD Month YYYY format (for example 10 January 1990)"
        elif pii_type == "age":
            return "provide the exact number"
        elif pii_type in NL_COL_TO_MAPFILE:
            # print(f"{pii_type} is in NL_COL_TO_MAPFILE")
            return f"choose from these options: {get_nl_values(pii_type)}"
        else:
            # print(f"{pii_type} is not in NL_COL_TO_MAPFILE")
            return ""
    if language == "Spanish":
        if pii_type == "CURP":
            return "proporciona el CURP del individuo (18 caracteres alfanumericos)"
        elif pii_type in ("phone number", "numero de telefono"):
            return "proporciona el numero de telefono mexicano del individuo (10 digitos)"
        elif pii_type in ("credit card number", "numero de tarjeta"):
            return "proporciona el numero de tarjeta en formato XXXX-XXXX-XXXX-XXXX"
        elif pii_type in ["ASISTEN", "CLASE_VIV", "DHSERSAL1", "ENT_PAID_NAC", "HESPANOL", "NIVACAD", "RELIGION", "SEXO", "SITUA_CONYUGAL", "ZIP_CODE",
                          "tipo de vivienda", "sexo", "edad", "entidad o pais de nacimiento", "servicio de salud", "religion", "habla lengua indigena", 
                          "habla espanol", "asistencia escolar", "nivel academico", "situacion conyugal", "hijos nacidos vivos",
                          ]:
            return f"elige de estas opciones: {get_mex_values(pii_type)}"
        else:
            return ""
    ## Serbian
    if pii_type in SRB_COL_TO_MAPFILE:
        return f"choose from these options: {get_srb_values(pii_type)}"
    elif pii_type in ("age", "age_mar", "partner_age"):
        return "provide the exact number"
    elif pii_type == "dob":
        return "use the date of birth in the DD Month YYYY format (for example 10 January 1990)"
    elif pii_type == "dom":
        return "use the date of first marriage in the DD Month YYYY format (for example 10 January 1990)"
    elif pii_type == "JMBG":
        return "provide the 13-digit JMBG number of the individual"
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
    if att in str_to_pums_col_dict:
        return str_to_pums_col_dict[att]
    elif att in MEX_STR_TO_COL:
        return MEX_STR_TO_COL[att]
    elif att in SRB_STR_TO_COL:
        return SRB_STR_TO_COL[att]
    elif att in NL_STR_TO_COL:
        return NL_STR_TO_COL[att]
    else:
        return att

def parse_output_gpt(response):
    output_lines = response.splitlines()

    guess_dict = dict()

    i = 0

    try:
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
    except Exception as e:
        print(f"Error occurred while parsing output: {e}")
        print("Output was:")
        for line in output_lines:
            print(line)
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
            except IndexError:
                print("No inference specified:")
                for ii in range(i, len(output_lines)):
                    print(output_lines[ii])
                print()
                inf = " "
            try:
                g = [
                    o.strip().strip('"') for o in output_lines[i + 2].split(":")
                ]
            except IndexError:
                print("No guess specified:")
                for ii in range(i, len(output_lines)):
                    print(output_lines[ii])
                print()
                g = " "
            try:
                cert = [
                    o.strip().strip('"') for o in output_lines[i + 3].split(":")
                ]
            except IndexError:
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
            except json.JSONDecodeError:
                try:
                    return json.loads(fixed_str + "}}")
                except json.JSONDecodeError:
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
def write_output(filepath, dataentries):
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
