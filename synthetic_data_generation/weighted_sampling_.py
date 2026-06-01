import json
import pandas as pd
import random
import pickle
from tqdm import tqdm
import numpy as np
import correctmatch

def spanish():
     
    N_POP = 125515554 
    N = 100

    IDENTIFIER_COUNT = 5
    PERSONAS_DF = pd.read_csv("data/es/personas_sample_enc.csv")
    PERSONAS_DF_NO_NANS = PERSONAS_DF.dropna().astype(int)
    INDIRECT_IDENTIFIERS = PERSONAS_DF.columns.tolist()
    OUTPUT_DATASET = "data/es/100_profiles.csv"
    CORRECTNESS_OUTPUT = "data/es/correctness_dict.pkl"

    # Initialize Result Dataset
    df_new = pd.DataFrame(columns=PERSONAS_DF_NO_NANS.columns.tolist())

    model_dict = {}
    correctness_dict = {}

    for i in tqdm(range(N)):
        # Select indirect identifiers of interest
        identifiers = random.sample(INDIRECT_IDENTIFIERS, k=IDENTIFIER_COUNT)
        identifiers = sorted(identifiers)

        print(f"{identifiers=}")

        # Obtain count of each value combination for identifiers of interest
        grouped_df = PERSONAS_DF_NO_NANS.groupby(identifiers).size()
        grouped_df = grouped_df.sort_values()

        selected_df = grouped_df[grouped_df == 1]
        selected_df = pd.DataFrame(selected_df).reset_index().rename({0:"count"}, axis=1).sample(frac=1, replace=False)
        # selected_df = selected_df.drop("count", axis=1)
        ## train the model
        
        identifiers_str = "/".join(identifiers)
        if identifiers_str in model_dict:
            model = model_dict[identifiers_str]
        else:
            model = correctmatch.fit_model(PERSONAS_DF.fillna(0).astype(int)[identifiers].to_numpy())
            model_dict[identifiers_str] = model

        current_max = 0
        current_max_row = None
        found_unique_person = False

        for j, row in selected_df.iterrows():
            correctness_list = []
            for _ in range(10):
                correctness = correctmatch.individual_correctness(model, np.array(list(row[identifiers])), n=N_POP)
                correctness_list.append(correctness)
                if correctness < 0.1:
                    break
            
            mean_c = np.mean([c for c in correctness_list if not np.isnan(c)])
            if mean_c > current_max:
                current_max = mean_c
                current_max_row = row.copy()

            if np.mean([c for c in correctness_list if not np.isnan(c)]) > 0.9:
                selected_row = row
                print(f"Found correctness = {mean_c}")
                found_unique_person = True
                correctness_dict[j] = correctness_list
                break

        if not found_unique_person:
            selected_row = current_max_row

        assert(len(selected_row.shape) == 1)

        # Remove the sampled row from our dataset to avoid resampling it

        selected_row = pd.DataFrame(selected_row).transpose().drop("count", axis=1)
        print(selected_row)

        query = ""
        for _, identifier in enumerate(identifiers):
            if query != "":
                query = query + " and "
            query = query + "(" + identifier + "==" + str(selected_row.iloc[0][identifier]) + ")"
        selected_df = PERSONAS_DF_NO_NANS.query(query)

        PERSONAS_DF_NO_NANS = PERSONAS_DF_NO_NANS.drop(selected_df.index, axis=0)
        print(selected_df)
        print(f"{len(selected_df)=}")

        # Add the sampled row to our output
        identifier_list = ""
        for identifier in identifiers:
            if identifier_list != "":
                identifier_list += ","
            identifier_list += identifier
        selected_df.insert(len(selected_df.columns), "identifiers", identifier_list)
        # selected_row.insert(len(selected_row.columns), "identifiers", identifier_list)
        df_new = pd.concat([df_new, selected_df])
        
    # Write output dataset to file
    df_new.to_csv(OUTPUT_DATASET, index=False)

    with open(CORRECTNESS_OUTPUT, "wb") as f:
        pickle.dump(correctness_dict, f)

def english():
    INPUT_DATASET = "data/pums/pums_pwgtp_sample_.csv" # IPUMS sample you sample from (incl all attributes and direct identifiers)
    # INPUT_DATASET = "data/pums/pums_full.csv"
    OUTPUT_DATASET = "data/pums/pums_pwgtp_sample_100.csv" # Output dataset (from which you will then generate text, one row = one profile)
    CORRECTNESS_OUTPUT = "data/pums/correctness_dict_pwgtp.pkl" # sanity check file
    INDIRECT_IDENTIFIERS = ["RAC2P", "CIT", "ST", "OCCP", "MAR", "SEX", "ESR", "SCHL", "DOB"] # All indirect identifiers in the dataset (column names)
    IDENTIFIER_COUNT = 5 # Number of indirect identifiers to sample at each iteration
    N = 100 # Number of profiles you need (these 100 profiles will be used to generate text later)
    N_model = 309349689 # population size (for US data it was 300M)

    # Read Dataset
    df_old = pd.read_csv(INPUT_DATASET, na_values=" ?")
    df_full = pd.read_csv(INPUT_DATASET, na_values=" ?")
    df_old = df_old.dropna()

    df_full = df_full.reset_index(drop=True)
    df_old = df_old.reset_index(drop=True)

    # Initialize Result Dataset
    df_new = pd.DataFrame(columns=df_old.columns.tolist())

    model_dict = {}
    correctness_dict = {}

    for i in tqdm(range(N)):
        # Select indirect identifiers of interest
        print(f"{identifiers=}")
        identifiers = random.sample(INDIRECT_IDENTIFIERS, k=IDENTIFIER_COUNT)
        identifiers = sorted(identifiers)

        # Obtain count of each value combination for identifiers of interest
        grouped_df = df_old.groupby(identifiers).size()
        grouped_df = grouped_df.sort_values()

        selected_df = grouped_df[grouped_df == 1]
        selected_df = pd.DataFrame(selected_df).reset_index().rename({0:"count"}, axis=1).sample(frac=1, replace=False)
        
        ## train the model
        
        identifiers_str = "/".join(identifiers)
        if identifiers_str in model_dict:
            model = model_dict[identifiers_str]
        else:
            model = correctmatch.fit_model(df_full[identifiers].to_numpy())
            model_dict[identifiers_str] = model

        current_max = 0
        current_max_row = None
        found_unique_person = False

        for j, row in selected_df.iterrows():
            
            correctness_list = []
            for _ in range(10):
                correctness = correctmatch.individual_correctness(model, np.array(list(row[identifiers])), n=N_model)
                correctness_list.append(correctness)
                if correctness < 0.1:
                    break
            
            mean_c = np.mean([c for c in correctness_list if not np.isnan(c)])
            if mean_c > current_max:
                current_max = mean_c
                current_max_row = row.copy()

            if np.mean([c for c in correctness_list if not np.isnan(c)]) > 0.9:
                selected_row = row
                print(f"Found correctness = {mean_c}")
                found_unique_person = True
                correctness_dict[j] = correctness_list
                break

        if not found_unique_person:
            selected_row = current_max_row

        assert(len(selected_row.shape) == 1)

        # Remove the sampled row from our dataset to avoid resampling it

        selected_row = pd.DataFrame(selected_row).transpose().drop("count", axis=1)
        print(selected_row)

        query = ""
        for _, identifier in enumerate(identifiers):
            if query != "":
                query = query + " and "
            query = query + "(" + identifier + "==" + str(selected_row.iloc[0][identifier]) + ")"
        selected_df = df_old.query(query)

        df_old = df_old.drop(selected_df.index, axis=0)
        print(selected_df)
        print(f"{len(selected_df)=}")

        # Add the sampled row to our output
        identifier_list = ""
        for identifier in identifiers:
            if identifier_list != "":
                identifier_list += ","
            identifier_list += identifier
        selected_df.insert(len(selected_df.columns), "identifiers", identifier_list)
        # selected_row.insert(len(selected_row.columns), "identifiers", identifier_list)
        df_new = pd.concat([df_new, selected_df])
        
    # Write output dataset to file
    df_new.to_csv(OUTPUT_DATASET, index=False)

    with open(CORRECTNESS_OUTPUT, "wb") as f:
        pickle.dump(correctness_dict, f)

def serbian():
    N_POP = 1_471_189  # Serbia population
    N = 100
    IDENTIFIER_COUNT = 5

    INPUT_DATASET = "data/srb/data/sample_enc.csv"
    OUTPUT_DATASET = "data/srb/100_profiles.csv"
    CORRECTNESS_OUTPUT = "data/srb/correctness_dict.pkl"

    # Identifier pool. Conditional columns have NaN for non-applicable rows:
    #   dom, age_mar: NaN when never married (marital_status == 2)
    #   partner_age:  NaN when not currently in union (marital_status != 1)
    #   language:     NaN for ~2.6% of rows (missing data)
    # dropna(subset=identifiers) enforces validity automatically — selecting
    # partner_age as an identifier implicitly restricts to partnered people.
    IDENTIFIER_POOL = [
        "urban", "age", "marital_status", "given_birth", "ethnicity",
        "language", "dob", "dom", "age_mar", "partner_age",
    ]

    df = pd.read_csv(INPUT_DATASET)
    df["age_mar"] = pd.to_numeric(df["age_mar"], errors="coerce")  # "NIU" → NaN
    df = df.reset_index(drop=True)

    df_work = df.copy()
    df_new = pd.DataFrame(columns=df.columns.tolist())

    model_dict = {}
    correctness_dict = {}

    for i in tqdm(range(N)):
        # Sample identifiers; retry if the filtered working set is empty
        for _ in range(20):
            identifiers = sorted(random.sample(IDENTIFIER_POOL, k=IDENTIFIER_COUNT))
            df_filtered = df_work.dropna(subset=identifiers)
            if len(df_filtered) > 0:
                break

        print(f"{identifiers=}")

        grouped_df = df_filtered.groupby(identifiers).size()
        grouped_df = grouped_df.sort_values()

        selected_df = grouped_df[grouped_df == 1]
        if selected_df.empty:
            selected_df = grouped_df[grouped_df == grouped_df.min()]
        selected_df = pd.DataFrame(selected_df).reset_index().rename({0: "count"}, axis=1).sample(frac=1, replace=False)

        # CorrectMatch requires each column's values to be in {1,...,K}.
        # Re-encode here since columns like `age` or `age_mar` hold actual
        # values (e.g. 49) that exceed K (e.g. 35 unique values).
        identifiers_str = "/".join(identifiers)
        if identifiers_str in model_dict:
            model, encoders = model_dict[identifiers_str]
        else:
            df_full_filtered = df.dropna(subset=identifiers)
            encoders = {
                col: {v: idx + 1 for idx, v in enumerate(sorted(df_full_filtered[col].unique().tolist()))}
                for col in identifiers
            }
            X = df_full_filtered[identifiers].apply(
                lambda col: col.map(encoders[col.name])
            ).astype(int).to_numpy()
            model = correctmatch.fit_model(X)
            model_dict[identifiers_str] = (model, encoders)

        current_max = 0
        current_max_row = None
        found_unique_person = False

        for j, row in selected_df.iterrows():
            individual = np.array([encoders[id][row[id]] for id in identifiers], dtype=int)
            correctness_list = []
            for _ in range(10):
                correctness = correctmatch.individual_correctness(
                    model, individual, n=N_POP
                )
                correctness_list.append(correctness)
                if correctness < 0.1:
                    break

            valid_c = [c for c in correctness_list if not np.isnan(c)]
            mean_c = np.mean(valid_c) if valid_c else 0.0

            if mean_c > current_max:
                current_max = mean_c
                current_max_row = row.copy()

            if mean_c > 0.9:
                selected_row = row
                print(f"Found correctness = {mean_c}")
                found_unique_person = True
                correctness_dict[j] = correctness_list
                break

        if not found_unique_person:
            selected_row = current_max_row

        assert len(selected_row.shape) == 1

        selected_row = pd.DataFrame(selected_row).transpose().drop("count", axis=1)
        print(selected_row)

        query = " and ".join(
            f"({id}=={selected_row.iloc[0][id]})" for id in identifiers
        )
        matched_df = df_work.query(query)

        df_work = df_work.drop(matched_df.index, axis=0)
        print(matched_df)
        print(f"{len(matched_df)=}")

        matched_df = matched_df.copy()
        identifier_list = ",".join(identifiers)
        matched_df.insert(len(matched_df.columns), "identifiers", identifier_list)
        df_new = pd.concat([df_new, matched_df])

    df_new.to_csv(OUTPUT_DATASET, index=False)

    with open(CORRECTNESS_OUTPUT, "wb") as f:
        pickle.dump(correctness_dict, f)


def dutch():
    N_POP = 16_655_799  # Netherlands 2011 population
    N = 100
    IDENTIFIER_COUNT = 5

    INPUT_DATASET = "data/nl/data/sample_enc.csv"
    OUTPUT_DATASET = "data/nl/100_profiles.csv"
    CORRECTNESS_OUTPUT = "data/nl/correctness_dict.pkl"

    IDENTIFIER_POOL = [
        "age", "sex", "marstd", "nativity", "bplcountry", "nation",
        "educnl", "empstatd", "labforce", "occisco", "indgen", "dob"
    ]

    df = pd.read_csv(INPUT_DATASET)
    df = df.reset_index(drop=True)
    df["dob"] = df.dob.fillna(6210).astype(int)

    # All columns are pre-encoded as {1,...,K} — no re-encoding needed.
    df_work = df.copy()
    df_new = pd.DataFrame(columns=df.columns.tolist())

    model_dict = {}
    correctness_dict = {}

    for i in tqdm(range(N)):
        identifiers = sorted(random.sample(IDENTIFIER_POOL, k=IDENTIFIER_COUNT))
        print(f"{identifiers=}")

        grouped_df = df_work.groupby(identifiers).size().sort_values()

        selected_df = grouped_df[grouped_df == 1]
        if selected_df.empty:
            selected_df = grouped_df[grouped_df == grouped_df.min()]
        selected_df = (
            pd.DataFrame(selected_df)
            .reset_index()
            .rename({0: "count"}, axis=1)
            .sample(frac=1, replace=False)
        )

        identifiers_str = "/".join(identifiers)
        if identifiers_str in model_dict:
            model = model_dict[identifiers_str]
        else:
            model = correctmatch.fit_model(df[identifiers].to_numpy())
            model_dict[identifiers_str] = model

        current_max = 0
        current_max_row = None
        found_unique_person = False

        for j, row in selected_df.iterrows():
            individual = np.array(list(row[identifiers]), dtype=int)
            correctness_list = []
            for _ in range(10):
                correctness = correctmatch.individual_correctness(model, individual, n=N_POP)
                correctness_list.append(correctness)
                if correctness < 0.1:
                    break

            valid_c = [c for c in correctness_list if not np.isnan(c)]
            mean_c = np.mean(valid_c) if valid_c else 0.0

            if mean_c > current_max:
                current_max = mean_c
                current_max_row = row.copy()

            if mean_c > 0.9:
                selected_row = row
                print(f"Found correctness = {mean_c}")
                found_unique_person = True
                correctness_dict[j] = correctness_list
                break

        if not found_unique_person:
            selected_row = current_max_row
            correctness_dict[j] = correctness_list
            print(f"No unique person found; using row with max correctness = {current_max}")

        assert len(selected_row.shape) == 1

        selected_row = pd.DataFrame(selected_row).transpose().drop("count", axis=1)
        print(selected_row)

        query = " and ".join(
            f"({id}=={selected_row.iloc[0][id]})" for id in identifiers
        )
        matched_df = df_work.query(query)

        df_work = df_work.drop(matched_df.index, axis=0)
        print(matched_df)
        print(f"{len(matched_df)=}")

        matched_df = matched_df.copy()
        matched_df.insert(len(matched_df.columns), "identifiers", ",".join(identifiers))
        df_new = pd.concat([df_new, matched_df])

    df_new.to_csv(OUTPUT_DATASET, index=False)

    with open(CORRECTNESS_OUTPUT, "wb") as f:
        pickle.dump(correctness_dict, f)


if __name__=="__main__":
    dutch()
    # serbian()
    # spanish()
    # english()