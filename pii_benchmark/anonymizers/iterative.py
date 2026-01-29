import json
from random import random
from time import time
from typing import List
from random import sample

from transformers import pipeline

from pii_benchmark.anonymizers.anonymizer import Anonymizer
from pii_benchmark.evaluation import check_correctness, check_guesses_one_profile
from pii_benchmark.prompts import get_staab_prompt, get_staab_prompt_llama
from pii_benchmark.utils import parse_output, parse_output_gpt
from pii_benchmark.credentials import openai_api_key

from openai import APIStatusError, OpenAI, RateLimitError
MAX_DELAY = 60
BASE_DELAY = 2
MAX_RETRIES = 10

ALL_PUMS_ATTRIBUTES = [
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
    "sex",
    "zip code"
]

EXTRA_ATTRIBUTES = [
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
]

PRESIDIO_ATTRIBUTES = [
    "credit card number",
    "crypto wallet number",
    "date or time",
    "date of birth",
    "email address",
    "iban code",
    "ip address",
    "Nationality, religious or political group",
    "Location",
    "state of residence",
    "name",
    "phone number",
    "Medical license",
    "URL",
    "US bank number",
    "US drivers license",
    "US Individual Taxpayer Identification Number",
    "US passport number",
    "SSN"
]


id_map = {
    "phone": "PHONE NUMBER",
    "PHONE NUMBER": "PHONE NUMBER",
    "SSN": "SSN",
    "name": "NAME",
    "email": "PERSONAL EMAIL ADDRESS",
    "address": "RESIDENTIAL ADDRESS",
    "date of birth": "DATE OF BIRTH",
    "DOB": "DATE OF BIRTH",
    "credit card": "CREDIT CARD NUMBER",
    "credit card number": "CREDIT CARD NUMBER",
    "phone number": "PHONE NUMBER",
}

SCENARIO_TEXT = {
    "medical consultation": ["transcript from a medical consultation", "patient", "consultation"],
    "ai chatbot": ["chatbot interaction", "user", "interaction"]
}

ANONYMIZER_SYSTEM_PROMPT = "You are an expert anonymizer that carefully adapts small parts of the text to make it \
                    anonymous. You follow the instructions and format precisely and you try to change as little \
                    as possible, keeping the original text intact as much as possible. Only generalize \
                    information and do not invent new information. \
                    Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my \
                    wife and I' is not. \
                    Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my \
                    husband and I have a dog' -> 'my partner and I have a cat' is not. \
                    Example: 'my husband and I' -> 'I' is also valid as it only removes information."
                    
ANONYMIZER_PROMPT = "Below I give you a {scenario_text} as well as personal inferences made about the \
                    {scenario_subject} that can be made from the {scenario_text_short}. Your task is to anonymize the {scenario_text_short} to prevent \
                    these inferences, while keeping the rest of the text intact. \
                    Conversation: \
                    {text} \
                    Inferences: \
                    {inferences} \
                    In a first step briefly explain the changes you will make. In a new line write a single # and \
                    then return the anonymized text. Only change relevant parts and keep everything else as is. \
                    Make sure you keep the original meaning, i.e. do not invent new information instead \
                    generalize information."

class IterativeAnonymizer(Anonymizer):
    def __init__(self,
                 model_version: str,
                 max_iters: int = 3,
                 attribute_list: str = "ours"):
        super().__init__()
        self.model_version = model_version
        self.max_iters = max_iters
        self.client = OpenAI(api_key=openai_api_key)
        self.attribute_list = attribute_list

        print(f"Initialized iterative anonymizer, attribute list {self.attribute_list}")

    def anonymize(self, profile: dict) -> str:
        i = 0
        ANON_SUCCESSFUL = False
        text = profile["text"]

        full_ground_truth = dict()
        if "direct_identifiers" in profile:
            for id in profile["direct_identifiers"].keys():
                full_ground_truth[id] = profile["direct_identifiers"][id]

        if "indirect_identifiers" in profile:
            for id in profile["indirect_identifiers"].keys():
                full_ground_truth[id] = profile["indirect_identifiers"][id]
        else:
            for id in profile["ground_truth"].keys():
                full_ground_truth[id] = profile["ground_truth"][id]

        if self.attribute_list=="ours":
            attributes_to_pass = ALL_PUMS_ATTRIBUTES
        elif self.attribute_list=="random":
            attributes_to_pass = sample(ALL_PUMS_ATTRIBUTES, 5)
        elif self.attribute_list=="extra":
            attributes_to_pass = ALL_PUMS_ATTRIBUTES + EXTRA_ATTRIBUTES
        elif self.attribute_list=="presidio":
            attributes_to_pass = PRESIDIO_ATTRIBUTES

        # print("features:")
        # print(profile["features"])
        # for f in profile["features"]:
        #     if f in profile["direct_identifiers"]:
        #         print(f"{f}: {profile["direct_identifiers"][f]}")
        #     elif f in profile["indirect_identifiers"]:
        #         print(f"{f}: {profile["indirect_identifiers"][f]}")

        while (i<self.max_iters) and (not ANON_SUCCESSFUL):
            ### predict
            # print(f"iteration {i+1}")
            inferences = self.infer(
                text,
                attributes=attributes_to_pass,
                scenario=profile["scenario"]
            )
            # print("inferences:")
            # print(inferences)
            # correctness = check_guesses_one_profile_local(guesses=inferences, ground_truth=full_ground_truth, attributes=profile["features"])[0]
            correctness = check_guesses_one_profile(guesses=inferences, ground_truth=full_ground_truth, attributes=profile["features"])[0]
            
            correct_inferences = []


            # print(f"correctness: {correctness}")
            # print(f"inferences: {inferences}")

            for k in correctness.keys():
                if correctness[k] is not None and len(correctness[k])>0 and correctness[k][0] == 1:
                    if k!="DOB-Day" and k!="DOB-Month" and k!="DOB-Year":
                        if k in inferences:
                            correct_inferences.append(
                                f"{k}: {inferences[k]}"
                            )
                        else:
                            correct_inferences.append(
                                f"{k}: {inferences[att_code_to_name[k]]}"
                            )
            if len(correct_inferences)==0:
                ANON_SUCCESSFUL = True
            else:
                correct_inferences = "\n".join(correct_inferences)

                ### anonymize
                text = self.anonymize_step(
                    text=text,
                    inferences=correct_inferences,
                )
                i+=1

        return text

    def anonymize_step(self, text: str, inferences: str, scenario: str = "medical consultation") -> str:

        prompt = ANONYMIZER_PROMPT.format(
            scenario_text = SCENARIO_TEXT[scenario][0],
            scenario_subject = SCENARIO_TEXT[scenario][1],
            scenario_text_short = SCENARIO_TEXT[scenario][2],
            text = text,
            inferences = inferences
        )

        chat = [
            {"role": "system", "content": ANONYMIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if self.model_version=="gpt-5":
            response = self.client.responses.create(
                model=self.model_version,
                input=chat,
                # temperature=0.0,
                max_output_tokens=4096,
                reasoning={"effort":"low"},
                text={"verbosity":"low"}
            )
        elif self.model_version=="gpt-4.1":
            response = self.client.responses.create(
                model=self.model_version,
                input=chat,
                max_output_tokens=4096,
            )
        anon_text = response.output_text
        anon_text = anon_text.split("#")[-1]
        return anon_text
    
    def infer(
        self, text: str, attributes: List[str] = None, scenario: str = "reddit"
    ):
        prompt = get_staab_prompt(attributes=attributes, text=text, scenario=scenario)

        if self.model_version=="gpt-5":
            response = self.client.responses.create(
                model=self.model_version,
                input=[
                    {
                        "role": "system",
                        "content": "You are an AI Assistant that specializes in generating synthetic data. Provide the user with a response in the exact format they specify, with no additional details.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=4096,
                # temperature=0.0,
                reasoning={"effort":"low"},
                text={"verbosity":"low"}
            )
        elif self.model_version=="gpt-4.1":
            response = self.client.responses.create(
                model=self.model_version,
                input=[
                    {
                        "role": "system",
                        "content": "You are an AI Assistant that specializes in generating synthetic data. Provide the user with a response in the exact format they specify, with no additional details.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=4096,
            )
        model_guesses = response.output_text#.output[0].content[0].text
        # print("model_guesses")
        # print(model_guesses)
          # response.choices[0].message.content
        # print("parsing output")
        try:
            model_guesses = json.loads(model_guesses)
            # print("json worked")
        except Exception as e:
            # print("json didn't work")
            # print(str(e))
            model_guesses = parse_output_gpt(model_guesses)
        # print(type(model_guesses))
        # print("parsed outqput:")
        # print(model_guesses)
        # print("model_guesses info:")
        # print(f"model_guesses keys: {model_guesses.keys()}")
        # try:
        #     print(f"model guesses for one attribute type: {type(model_guesses[attributes[0]])}")
        # except:
        #     print("no model guesses")
        # print(f"model guesses for one attribute: {model_guesses[attributes[0]].keys()}")
        return model_guesses

class IterativeAnonymizerLlama(Anonymizer):
    def __init__(
        self,
        model_version: str,
        max_iters: int = 3
    ):
        """
        prompt_type defines the anonymization prompt used. It can be the generic anthropic prompt, 
        the modified anthropic prompt that includes attribute types,
        the Clio prompt, or the Rescriber prompt.
        """
        super().__init__()
        self.model_version = model_version
        self.max_iters = max_iters
        self.model = pipeline("text-generation", model=f"meta-llama/Llama-{model_version}")

    def anonymize(self, profile: dict) -> str:
        i = 0
        ANON_SUCCESSFUL = False
        text = profile["text"]

        full_ground_truth = dict()
        if "direct_identifiers" in profile:
            for id in profile["direct_identifiers"].keys():
                full_ground_truth[id] = profile["direct_identifiers"][id]
        for id in profile["ground_truth"].keys():
            full_ground_truth[id] = profile["ground_truth"][id]

        while (i<self.max_iters) and (not ANON_SUCCESSFUL):
            ### predict
            inferences = self.infer(
                text,
                attributes=profile["features"]
            )
            correctness = check_guesses_one_profile_local(guesses=inferences, ground_truth=full_ground_truth, attributes=profile["features"])[0]
            
            correct_inferences = []
            # print(correctness.keys())
            # print(inferences.keys())
            
            for k in correctness.keys():
                if correctness[k] is not None and correctness[k][0] == 1:
                    correct_inferences.append(
                        f"{k}: {inferences[k]}"
                    )
            if len(correct_inferences)==0:
                    ANON_SUCCESSFUL = True
            else:
                correct_inferences = "\n".join(correct_inferences)

                ### anonymize
                text = self.anonymize_step(
                    text=text,
                    inferences=correct_inferences,
                )
                i+=1

        return text

    def anonymize_step(self, text: str, inferences: str, task: str = "medical consultation") -> str:

        prompt = ANONYMIZER_PROMPT.format(
            scenario_text = SCENARIO_TEXT[task][0],
            scenario_subject = SCENARIO_TEXT[task][1],
            scenario_text_short = SCENARIO_TEXT[task][2],
            text = text,
            inferences = inferences
        )

        chat = [
            {"role": "system", "content": ANONYMIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # print(ANONYMIZER_SYSTEM_PROMPT)
        # print(prompt)
        response = self.model(chat, max_new_tokens=4096)
        anon_text = ""
        for r in response[0]["generated_text"]:
            if r["role"] == "assistant":
                anon_text = r["content"]
        return anon_text

    def infer(
        self,
        text: str,
        attributes: List[str] = None,
        task: str = "medical",
    ) -> str:
        system_prompt, prompt = get_staab_prompt_llama(
            attributes=attributes, text=text, scenario=task
        )

        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self.model(chat, max_new_tokens=4096)
        model_guesses = ""
        for r in response[0]["generated_text"]:
            if r["role"] == "assistant":
                model_guesses = r["content"]
        model_guesses = parse_output(model_guesses)
        return model_guesses


att_name_to_code = {
    "age": "AGEP",
    "sex": "SEX",
    "race": "RAC2P",
    "citizenship status": "CIT",
    "education level": "SCHL",
    "educational attainment": "SCHL",
    "state of place of work": "POWSP",
    "time of departure for work": "JWDP",
    "occupation": "OCCP",
    "class of worker": "COW",
    "marital status": "MAR",
    "military service": "MIL",
    "means of transportation to work": "JWTRNS",
    "phone": "phone",  # 'PHONE NUMBER',
    # "name": "name",  # 'NAME',
    # "NAME": "name",
    "name": "NAME",
    "email": "PERSONAL EMAIL ADDRESS",
    "address": "RESIDENTIAL ADDRESS",
    "SSN": "SSN",
    "DOB": "DOB",
    "date of birth": "DOB",
    "credit card": "CREDIT CARD NUMBER",
    "credit card number": "CREDIT CARD NUMBER",
    "employment status": "ESR",
    "state of residence": "ST",
    "number of people in carpool when driving to work": "DRIVESP",
    "phone number": "PHONE NUMBER",
    "Day of Birth": "DOB-Day",
    "Month of Birth": "DOB-Month",
    "Year of Birth": "DOB-Year",
}

att_code_to_name = dict((v,k) for k,v in att_name_to_code.items())

def check_guesses_one_profile_local(
    guesses: dict, ground_truth: dict, attributes: List[str]
) -> dict:
    correctness = dict()
    correctness_llm = dict()
    for att in attributes:
        if att in id_map and att != "DOB":
            gt = str(ground_truth[id_map[att]])
        else:
            gt = str(ground_truth[att])

        att_ = att
        # print("in the correctness function")
        # print(f"att name: {att_}")
        # print(f"guesses keys: {guesses.keys()}")
        # print(f"att_ in att_code_to_name: {att_ in att_code_to_name}")
        # if att_ in att_code_to_name:
        #     print(f"att_ in att_code_to_name: {att_code_to_name[att_]}")

        if att_ in guesses:
            if "Guess" not in guesses[att_]:
                model_guess = ""
            elif not isinstance(guesses[att_]["Guess"], list):
                model_guess = (
                    guesses[att_]["Guess"].strip("[").strip("]").split(";")
                )
                model_guess = [
                    m
                    for m in model_guess
                    if m != "" and m != ", " and m != "],"
                ]
            else:
                print(f"guesses are a list: {guesses[att_]["Guess"]}")
                print(f"len(guesses): {len(guesses[att_]["Guess"])}")
                model_guess = guesses[att_]["Guess"]
        elif att_ in att_code_to_name and att_code_to_name[att_] in guesses:
            att_name = att_code_to_name[att_]
            if "Guess" in guesses[att_name]:
                if isinstance(guesses[att_name]["Guess"], list):
                    print(f"model guess is list:{guesses[att_name]["Guess"]}")
                    model_guess = guesses[att_name]["Guess"]
                else:
                    model_guess = guesses[att_name]["Guess"].strip("[").strip("]").split(";")
                    model_guess = [
                            m
                            for m in model_guess
                            if m != "" and m != ", " and m != "],"
                        ]
            else:
                model_guess = ""
        else:
            model_guess = ""

        is_correct_llm = None

        # print(f"model guesses: {model_guess}")
        # print(f"ground truth: {gt}")

        is_correct = check_correctness(
            gt=gt,
            model_guess=model_guess,
            pii_type=att,
        )
        correctness[att] = is_correct
        correctness_llm[att] = is_correct_llm

    return correctness, correctness_llm
