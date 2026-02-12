# Select the individuals from the real dataset we will use as seeds to generate our synthetic data.
import pandas as pd
import random
import numpy as np
import json
import pickle


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
) -> list:
    # Load into pandas
    if dataset is None:
        df = pd.read_csv(
            dataset_link,
            na_values=" ?"
        )
    else:
        df = dataset

    df = df[columns]

    # Drop rows with missing values for simplicity
    df = df[columns].dropna()
    df = df.convert_dtypes(convert_integer=True)
    df["credit card number"] = df["credit card number"].astype(int).astype(str)


    # Pick a random row
    selected_people = list()

    print(f"{no_of_entries=}, {len(df)=}")

    for i in range(no_of_entries):
        random_entry = df.iloc[i]
        random_entry, groundtruth = get_pums_profile(random_entry, columns)
        selected_people.append((random_entry, groundtruth))

    return selected_people


def get_feature_codes(dataset: str):
    return list(PUMS_ATT_CODE_TO_NAME.keys())


def get_target_attributes_from_dataentry(dataentry, features, dataset):
    filteredentry = dict()
    for feature in features:
        try:
            if feature == "address":
                print(dataentry.keys())
                key = "address"
                filteredentry[key] = dataentry["address"]
                filteredentry["zip code"] = dataentry["zip code"]
            else:
                if feature in dataentry:
                    key = feature
                    filteredentry[key] = dataentry[key]
                elif feature in PUMS_ATT_CODE_TO_NAME:
                    key = PUMS_ATT_CODE_TO_NAME.get(feature, None)
                    filteredentry[key] = dataentry[key]
                else:
                    pass
        except TypeError as e:
            print(f"Error: {e} for feature: {feature}")
            print(key)
            print()
    return filteredentry
