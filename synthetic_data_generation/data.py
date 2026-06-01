# Select the individuals from the real dataset we will use as seeds to generate our synthetic data.
import pandas as pd
import random
import numpy as np
import json
import pickle

from synthetic_data_generation.direct_identifiers import (
    get_full_name_mex,
    generate_curp,
    generate_card,
    generate_mexican_phone,
    generate_mexican_address,
    get_full_name_srb,
    generate_jmbg,
    generate_serbian_phone,
    generate_serbian_address,
    get_full_name_nl,
    generate_rrn,
    generate_nl_phone,
    generate_nl_address,
)


PUMS_ATT_CODE_TO_NAME = {
    "RAC2P": "race",
    "CIT": "citizenship status",
    "ST": "state of residence",
    "OCCP": "occupation",
    "MAR": "marital status",
    "SEX": "sex",
    "ESR": "employment status",
    "SCHL": "educational attainment",
    "DOB": "date of birth",
    "DOB-Day": "day of birth",
    "DOB-Month": "month of birth",
    "DOB-Year": "year of birth",
    "name": "name",
    "SSN": "SSN",
    "credit card number": "credit card number",
    "phone number": "phone number",
    # "address": "residential address",
    "address": "address",
    "zip code": "zip code",
    "PUMA_FULL": "puma code"
}

PUMS_NAME_TO_CODE = {
    v:k for k,v in PUMS_ATT_CODE_TO_NAME.items()
}

PUMS_OCCUPATION_TYPES = {
    "MGR": "Management",
    "BUS": "Business",
    "FIN": "Finance",
    "CMM": "Computer and mathematical occupations",
    "ENG": "Architecture and engineering",
    "SCI": "Life, physical, and social science",
    "CMS": "Community and social service",
    "LGL": "Legal occupations",
    "EDU": "Educational instruction, and library occupations",
    "ENT": "Arts, design, entertainment, sports, and media",
    "MED": "Healthcare practitioners and technical occupations",
    "HLS": "Healthcare support",
    "PRT": "Protective service",
    "EAT": "Food preparation and serving",
    "CLN": "Building and Grounds Cleaning and Maintenance",
    "PRS": "Personal Care and Service Occupations",
    "SAL": "Sales and Related Occupations",
    "OFF": "Office and Administrative Support",
    "FFF": "Farming, Fishing, and Forestry",
    "CON": "Construction and Extraction",
    "EXT": "Construction and Extraction",
    "RPR": "Installation, Maintenance, and Repair Occupations",
    "PRD": "Production Occupations",
    "TRN": "Transportation and Material Moving Occupations",
    "MIL": "Military Occupations",
}

PARTIAL_CREDIT_LIST = ["DOB-Day", "DOB-Month", "DOB-Year", "zip code", "PUMA_FULL"]

# ── Serbian MICS attribute metadata ───────────────────────────────────────────

SRB_ATT_CODE_TO_NAME = {
    "urban":          "residential area",
    "age":            "age",
    "marital_status": "marital status",
    "given_birth":    "ever given birth",
    "dob":            "date of birth",
    "dom":            "date of marriage",
    "age_mar":        "age at marriage",
    "partner_age":    "partner's age",
    "ethnicity":      "ethnicity",
    "language":       "language",
    # direct identifiers
    "name":               "name",
    "JMBG":               "JMBG",
    "credit card number": "credit card number",
    "phone number":       "phone number",
    "address":            "address",
    "email":              "email",
}

SRB_DIRECT_IDENTIFIER_COLS = frozenset(
    {"name", "JMBG", "credit card number", "phone number", "address", "email"}
)

SRB_SKIP_VALUES = frozenset({"NIU", "Missing", "NaN"})

# Columns decoded via a JSON map in data/srb/maps/
_SRB_MAPPED_COLS = frozenset({
    "urban", "marital_status", "given_birth", "dob", "dom",
    "ethnicity", "language", "birth_year", "birth_month",
    "marriage_year", "marriage_month",
})

# Column name → map filename stem (only needed when they differ)
_SRB_MAP_FILENAME = {
    "urban":       "urban_status",
    "given_birth": "ever_given_birth",
}

# Columns that are raw numerics (no map file)
_SRB_RAW_NUMERIC_COLS = frozenset({
    "age", "age_mar", "partner_age", "birth_day", "marriage_day",
})

# ── Dutch/Flemish (NL) attribute metadata ─────────────────────────────────────

NL_ATT_CODE_TO_NAME = {
    "age":        "leeftijd",
    "sex":        "geslacht",
    "marstd":     "burgerlijke staat",
    "nativity":   "herkomst",
    "bplcountry": "geboorteland",
    "nation":     "nationaliteit",
    "educnl":     "opleidingsniveau",
    "empstatd":   "arbeidsstatus",
    "labforce":   "arbeidsparticipatie",
    "occisco":    "beroep",
    "indgen":     "sector",
    "dob":        "geboortedatum",
    # direct identifiers
    "name":               "naam",
    "RRN":                "rijksregisternummer",
    "credit card number": "kredietkaartnummer",
    "phone number":       "telefoonnummer",
    "address":            "adres",
    "email":              "e-mailadres",
}

NL_DIRECT_IDENTIFIER_COLS = frozenset(
    {"name", "RRN", "credit card number", "phone number", "address", "email"}
)

NL_SKIP_VALUES = frozenset({"NIU (not in universe)", "Unknown/missing", "Unknown", "NaN"})

# column name → map file stem
_NL_MAP_FILENAME = {
    "age":        "age",
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
    "dob":        "dob",
}

# ── Mexican Census (MEX) attribute metadata ────────────────────────────────────

MEX_ATT_CODE_TO_NAME = {
    "CLASE_VIV":        "tipo de vivienda",
    "SEXO":             "sexo",
    "EDAD":             "edad",
    "ENT_PAIS_NAC":     "entidad o país de nacimiento",
    "DHSERSAL1":        "servicio de salud",
    "RELIGION":         "religión",
    "HLENGUA":          "habla lengua indígena",
    "HESPANOL":         "habla español",
    "ASISTEN":          "asistencia escolar",
    "NIVACAD":          "nivel académico",
    "SITUA_CONYUGAL":   "situación conyugal",
    "HIJOS_NAC_VIVOS":  "hijos nacidos vivos",
    # direct identifiers — stored in groundtruth under the same key
    "name":             "nombre",
    "CURP":             "CURP",
    "credit card number": "número de tarjeta",
    "phone number":     "número de teléfono",
    "address":          "domicilio",
    "email":            "correo electrónico",
}

MEX_DIRECT_IDENTIFIER_COLS = frozenset(
    {"name", "CURP", "credit card number", "phone number", "address", "email"}
)

# Values that are "not applicable / unknown" in the Mexican census and should be
# omitted from the generated profile so the LLM is not asked to reveal them.
MEX_SKIP_VALUES = frozenset({
    "Blanco por pase", "No especificado",
    "No especificado de Entidad Federativa", "No especificado de país",
    "NaN",
})

# Get Human (LLM) Readable format of PUMS data
def get_pums_profile(sample, cols):
    s = ""
    dataentry = dict()
    groundtruth = dict()
    print(cols)
    for col in cols:
        print(f"Processing column: {col}")
        
        # Direct identifiers and DOB are unencoded and do not need a mapping to recover the original value
        if col == "identifiers" or col == "name" or col == "SSN" or col == "credit card number" or col == "phone number" or col == "address" or col == "zip code" or col == "PUMA_FULL" or col == "DOB-Day" or col == "DOB-Month" or col == "DOB-Year":
            if type(sample[col]) == np.float64:
                identifier_sample = int(sample[col])
            else:
                identifier_sample = sample[col]
            groundtruth[col] = identifier_sample
            if col == "zip code":
                dataentry[PUMS_ATT_CODE_TO_NAME[col]] = str(identifier_sample)
            elif col not in PARTIAL_CREDIT_LIST and col != "identifiers":
                dataentry[PUMS_ATT_CODE_TO_NAME[col]] = identifier_sample
        
        # All other identifiers need to be recovered using a mapping
        else:
            with open(f"./data/maps/{col}_map.pickle", "rb") as f:
                map = pickle.load(f)

            if col == "OCCP":
                k = int(sample["OCCP"])

                occp = map[k]
                occp = occp.split("-")
                if len(occp) > 1:
                    occp_type, occp_desc = occp[0], "-".join(occp[1:])
                else:
                    occp_type, occp_desc = "", occp[0]

                if len(occp) > 1:
                    s += f"{PUMS_ATT_CODE_TO_NAME[col]}: TYPE: {PUMS_OCCUPATION_TYPES[occp_type]}, DESCRIPTION: {occp_desc}"
                    dataentry[PUMS_ATT_CODE_TO_NAME[col]] = (
                        f"TYPE: {PUMS_OCCUPATION_TYPES[occp_type]}, DESCRIPTION: {occp_desc}"
                    )
                    groundtruth[col] = (
                        f"TYPE: {PUMS_OCCUPATION_TYPES[occp_type]}, DESCRIPTION: {occp_desc}"
                    )
                elif occp[0]=='N/A (less than 16 years old/NILF who last worked more than 5 years ago or never worked)':

                    s += f"{PUMS_ATT_CODE_TO_NAME[col]}: TYPE: N/A, DESCRIPTION: less than 16 years old/NILF who last worked more than 5 years ago or never worked"

                    dataentry[PUMS_ATT_CODE_TO_NAME[col]] = (
                        f"TYPE: N/A, DESCRIPTION: less than 16 years old/NILF who last worked more than 5 years ago or never worked"
                    )

                    groundtruth[col] = (
                        f"TYPE: N/A, DESCRIPTION: less than 16 years old/NILF who last worked more than 5 years ago or never worked"
                    )
                else:
                    print(f"problem with occupation: {occp}, k={k}")
                    dataentry[PUMS_ATT_CODE_TO_NAME[col]] = occp_type
                    groundtruth[col] = occp_type
            else:
                k = int(sample[col])
                s += f"{PUMS_ATT_CODE_TO_NAME[col]}: {map[k]}"
                dataentry[PUMS_ATT_CODE_TO_NAME[col]] = map[k]
                groundtruth[col] = map[k]
    return dataentry, groundtruth


def get_mex_profile(sample, cols):
    """Decode one row of the Mexican census encoded CSV into human-readable dicts.

    Returns (dataentry, groundtruth) where:
      - dataentry  keys are Spanish descriptive labels (used in the prompt)
      - groundtruth keys are the original column codes / identifier names
    Direct identifiers absent from the CSV (name, CURP, etc.) are generated here.
    """
    dataentry = {}
    groundtruth = {}

    for col in cols:
        # "identifiers" column holds the pre-selected attribute codes for this record
        if col == "identifiers":
            if "identifiers" in sample.index:
                groundtruth["identifiers"] = sample["identifiers"]
            continue

        # Direct identifiers are not in the CSV; generated below after decoding SEXO
        if col in MEX_DIRECT_IDENTIFIER_COLS:
            continue

        if col not in sample.index:
            continue

        map_path = f"./data/es/maps/{col}_map.json"
        with open(map_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        # Maps are stored as {human_label: encoded_int}; invert to decode
        inv_map = {v: k for k, v in raw_map.items()}

        k = int(sample[col])
        decoded = inv_map.get(k, str(k))

        groundtruth[col] = decoded

        if decoded in MEX_SKIP_VALUES:
            continue

        human_name = MEX_ATT_CODE_TO_NAME.get(col, col)

        # HIJOS_NAC_VIVOS values are floats ("0.0", "1.0", …); strip to int string.
        # Codes 98 / 99 represent "not specified" — skip them.
        if col == "HIJOS_NAC_VIVOS":
            try:
                count = float(decoded)
                if count >= 97:
                    continue
                decoded = str(int(count))
            except (ValueError, TypeError):
                continue

        dataentry[human_name] = decoded

    # Retrieve decoded sex for name generation (default Hombre if unavailable)
    sexo = groundtruth.get("SEXO", "Hombre")

    # Generate direct identifiers programmatically
    name = get_full_name_mex(sexo)
    curp = generate_curp()
    card = generate_card()
    phone = generate_mexican_phone()
    address = generate_mexican_address()

    groundtruth["name"] = name
    groundtruth["CURP"] = curp
    groundtruth["credit card number"] = card
    groundtruth["phone number"] = phone
    groundtruth["address"] = address

    # Use Spanish labels as keys in dataentry so the prompt reads naturally
    dataentry["nombre"] = name
    dataentry["CURP"] = curp
    dataentry["número de tarjeta"] = card
    dataentry["número de teléfono"] = phone
    dataentry["domicilio"] = address

    # Read pre-computed zip code from the CSV if the column exists
    # (populated by add_zip_codes_mex.py before running the generator)
    if "zip code" in sample.index and pd.notna(sample["zip code"]):
        zc = str(sample["zip code"]).strip()
        groundtruth["zip code"] = zc
        dataentry["zip code"] = zc   # skipped by convert_entry_to_string, kept for partial credit

    return dataentry, groundtruth


def get_srb_profile(sample, cols):
    """Decode one row of the Serbian MICS CSV into human-readable dicts.

    Returns (dataentry, groundtruth) where:
      - dataentry  keys are English descriptive labels (used in the prompt)
      - groundtruth keys are original column codes / identifier names
    Direct identifiers absent from the CSV are generated programmatically.
    """
    dataentry = {}
    groundtruth = {}

    for col in cols:
        if col == "identifiers":
            if "identifiers" in sample.index:
                groundtruth["identifiers"] = sample["identifiers"]
            continue

        if col in SRB_DIRECT_IDENTIFIER_COLS:
            continue

        if col not in sample.index:
            continue

        val = sample[col]
        if pd.isna(val):
            continue

        human_name = SRB_ATT_CODE_TO_NAME.get(col, col)

        if col in _SRB_RAW_NUMERIC_COLS:
            decoded = str(int(val))
            groundtruth[col] = decoded
            dataentry[human_name] = decoded

        elif col in _SRB_MAPPED_COLS:
            stem = _SRB_MAP_FILENAME.get(col, col)
            map_path = f"./data/srb/maps/{stem}_map.json"
            with open(map_path, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
            inv_map = {v: k for k, v in raw_map.items()}
            decoded = inv_map.get(int(val), str(val))
            groundtruth[col] = decoded
            if decoded not in SRB_SKIP_VALUES:
                dataentry[human_name] = decoded

    # Generate direct identifiers programmatically
    name = get_full_name_srb()
    jmbg = generate_jmbg()
    card = generate_card()
    phone = generate_serbian_phone()
    address = generate_serbian_address()

    groundtruth["name"] = name
    groundtruth["JMBG"] = jmbg
    groundtruth["credit card number"] = card
    groundtruth["phone number"] = phone
    groundtruth["address"] = address

    dataentry["name"] = name
    dataentry["JMBG"] = jmbg
    dataentry["credit card number"] = card
    dataentry["phone number"] = phone
    dataentry["address"] = address

    return dataentry, groundtruth


def get_nl_profile(sample, cols):
    """Decode one row of the NL IPUMS CSV into human-readable dicts (Flemish Dutch).

    Returns (dataentry, groundtruth) where:
      - dataentry  keys are Flemish Dutch descriptive labels (used in the prompt)
      - groundtruth keys are original column codes / identifier names
    Direct identifiers absent from the CSV are generated programmatically.
    """
    dataentry = {}
    groundtruth = {}

    for col in cols:
        if col == "identifiers":
            if "identifiers" in sample.index:
                groundtruth["identifiers"] = sample["identifiers"]
            continue

        if col in NL_DIRECT_IDENTIFIER_COLS:
            continue

        # skip dob_year / dob_month / dob_day — dob encodes the full date
        if col in ("dob_year", "dob_month", "dob_day"):
            continue

        if col not in sample.index:
            continue

        val = sample[col]
        if pd.isna(val):
            continue

        stem = _NL_MAP_FILENAME.get(col)
        if stem is None:
            continue

        map_path = f"./data/nl/maps/{stem}_map.json"
        with open(map_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        inv_map = {v: k for k, v in raw_map.items()}

        decoded = inv_map.get(int(val), str(val))

        groundtruth[col] = decoded

        if decoded in NL_SKIP_VALUES:
            continue

        # clean up age midpoints: "47.0" → "47"
        if col == "age":
            try:
                decoded = str(int(float(decoded)))
            except (ValueError, TypeError):
                pass

        human_name = NL_ATT_CODE_TO_NAME.get(col, col)
        dataentry[human_name] = decoded

    # Decode sex for name generation (default Male if unavailable)
    sex = groundtruth.get("sex", "Male")

    name  = get_full_name_nl(sex)
    rrn   = generate_rrn()
    card  = generate_card()
    phone = generate_nl_phone()
    addr  = generate_nl_address()

    groundtruth["name"]               = name
    groundtruth["RRN"]                = rrn
    groundtruth["credit card number"] = card
    groundtruth["phone number"]       = phone
    groundtruth["address"]            = addr

    dataentry["naam"]                 = name
    dataentry["rijksregisternummer"]  = rrn
    dataentry["kredietkaartnummer"]   = card
    dataentry["telefoonnummer"]       = phone
    dataentry["adres"]                = addr

    return dataentry, groundtruth


# Read metadata from a json file (is necessary for the reprosyn generators)
def read_metadata(metadata_path: str) -> tuple:
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    cols = [col["name"] for col in meta_data]
    return meta_data, cols


# Turn series into dict
def deserialize_entry(dataentry, cols) -> dict:
    output = dict()
    for col in cols:
        output[col] = dataentry[col]
    return output

# Read dataset from csv file
def get_dataset(dataset_link: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_link, na_values=" ?")
    return df


# Get data entry from pandas table
def get_data_entry(
    dataset_link: str | None,
    dataset: pd.DataFrame | None,
    no_of_entries: int,
    columns: list,
    data_source: str = "PUMS",
) -> list:
    if dataset is None:
        df = pd.read_csv(dataset_link, na_values=" ?")
    else:
        df = dataset

    selected_people = list()
    print(f"{no_of_entries=}, {len(df)=}")

    if data_source == "MEX":
        # The MEX CSV only contains census columns + "identifiers".
        # Direct identifiers are generated inside get_mex_profile().
        census_cols = [c for c in df.columns if c != "identifiers"]
        df_sel = df[census_cols + ["identifiers"]].dropna(subset=census_cols)
        for i in range(no_of_entries):
            sample = df_sel.iloc[i]
            entry, groundtruth = get_mex_profile(sample, columns)
            selected_people.append((entry, groundtruth))
        return selected_people

    if data_source == "SRB":
        for i in range(no_of_entries):
            sample = df.iloc[i]
            entry, groundtruth = get_srb_profile(sample, columns)
            selected_people.append((entry, groundtruth))
        return selected_people

    if data_source == "NL":
        for i in range(no_of_entries):
            sample = df.iloc[i]
            entry, groundtruth = get_nl_profile(sample, columns)
            selected_people.append((entry, groundtruth))
        return selected_people

    # ── PUMS path (unchanged) ──────────────────────────────────────────────────
    df = df[columns]
    df = df[columns].dropna()
    df = df.convert_dtypes(convert_integer=True)
    df["credit card number"] = df["credit card number"].astype(int).astype(str)

    for i in range(no_of_entries):
        random_entry = df.iloc[i]
        random_entry, groundtruth = get_pums_profile(random_entry, columns)
        selected_people.append((random_entry, groundtruth))

    return selected_people


def get_feature_codes(dataset: str):
    if dataset == "MEX":
        return list(MEX_ATT_CODE_TO_NAME.keys())
    if dataset == "SRB":
        return list(SRB_ATT_CODE_TO_NAME.keys())
    if dataset == "NL":
        return list(NL_ATT_CODE_TO_NAME.keys())
    return list(PUMS_ATT_CODE_TO_NAME.keys())


def get_target_attributes_from_dataentry(dataentry, features, dataset):
    filteredentry = dict()
    if dataset == "MEX":
        ATT_MAP = MEX_ATT_CODE_TO_NAME
    elif dataset == "SRB":
        ATT_MAP = SRB_ATT_CODE_TO_NAME
    elif dataset == "NL":
        ATT_MAP = NL_ATT_CODE_TO_NAME
    else:
        ATT_MAP = PUMS_ATT_CODE_TO_NAME
    for feature in features:
        try:
            if feature == "address":
                if dataset == "MEX":
                    filteredentry["domicilio"] = dataentry["domicilio"]
                    if "zip code" in dataentry:
                        filteredentry["zip code"] = dataentry["zip code"]
                elif dataset == "SRB":
                    filteredentry["address"] = dataentry["address"]
                elif dataset == "NL":
                    filteredentry["adres"] = dataentry["adres"]
                else:
                    print(dataentry.keys())
                    filteredentry["address"] = dataentry["address"]
                    filteredentry["zip code"] = dataentry["zip code"]
            else:
                if feature in dataentry:
                    filteredentry[feature] = dataentry[feature]
                elif feature in ATT_MAP:
                    key = ATT_MAP[feature]
                    if key in dataentry:
                        filteredentry[key] = dataentry[key]
                else:
                    pass
        except (TypeError, KeyError) as e:
            print(f"Error: {e} for feature: {feature}")
    return filteredentry
