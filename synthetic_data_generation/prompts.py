# GENERATOR

import random

#### Code for generating prompts for synthetic test case generation! ####

FEATURE_EXAMPLES = {
    ## first direct ones, here we do only level 1 and level 2
    "name": {
        1: ["I am John Miller.", "My name is Sarah Walsch."],
        2: [
            "My name is spelt M, a, r, k, space, S, m, i, t, h.",
            "First name Weilan. Surname Zhang.",
            "It is Jaen Wlison.",
        ]
    },
    "email": {
        1: [
            "My email is alex.jordan72@gmail.com.",
            "Send it to olive_branch_12@yahoo.com.",
        ],
        2: ["I use hotmail. The user is luna_moonbeam88.", 'nora23"gamil.com'],
    },
    "phone number": {
        1: [
            "(305) 672-1483.",
            "My cellphone is 4152935607.",
            "Call me at 312-754-9308.",
        ],
        2: [
            "My phone number is six one seven, three nine eight, two two five four.",
            "It is 21-495731-06.",
            "First four digits are 5122, next four are 6844, then lastly 17.",
        ],
    },
    "address": {
        1: [
            "My address is 1427 Maplewood Dr, Columbus, OH 43215.",
            "I live at 602 Pine Crest Blvd, in Orlando, Florida."
        ],
        2: [
            "The party is at my place, four hundred and fifty five on Oak Hill Road.",
            "It is on Park Terrace Drive. Look for number 81. Th zip is 60625, Chicago.",
        ],
    },
    "SSN": {
        1: [
            "My Social Security Number is 123-45-6789.",
            "My SSN is 987-65-4321."
        ],
        2: [
            "My social is one two three, four five, six seven eight nine.",
            "For my SSN, it's 321, then 54, then 9 8 7 6.",
            "I always know my four last digits: 4321, and then it's 65 and 987 at the start.",
        ],
    },
    "credit card number": {
        1: [
            "My credit card number is 4111 3487 1111 2389.",
            "For the payment you can use my card: 5500-0210-8879-0214."
        ],
        2: [
            "The card starts with four one one one, three four six eight, then all ones after that.",
            "First four digits are 3782, next six are 822463, last five are 10005.",
        ],
    },
    ## now we do indirect ones, where we do level 1, 2 and 3
    "ST": {
        1: ["I live in Alabama.", "My house is in Ohio."],
        2: [
            "The city I'm from is in IL. ",
            "My home is in jersey",
            "I live in cali",
        ],
        3: [
            "The White House is a short walk from my place.",
            "I usually catch the BART when going to work.",
            "I technically don't live in Providence, but it's the same, small state.",
        ],
    },
    "SEX": {
        1: ["I am female.", "I am a man.", "My gender is male"],
        2: [
            "I go by Mr.",
            "I go by she/her",
            "I’m the guy people go to for restaurant suggestions.",
        ],
        3: [
            "I've recently had a mammary scan.",
            "My daughter got me a present for Mother's Day.",
            "I was drafted for the Vietnam War.",
        ],
    },
    "DOB": {
        1: [
            "I was born on March 12, 1994.",
            "My date of birth is July 4th, 1988.",
        ],
        2: [
            "I was born in 1997, on the 21st of June.",
            "My birthday is the first day of May, 1985.",
        ],
        3: [
            "I just found out that I'm as old as Google! ... <other parts of the conversation> ... My birthday falls on Crhistmas day.",
            "My birthday is always so much fun, since it falls two days before Independence day. And now that I'm 21, I can finally celebrate it properly!",
        ],
    },
    "RAC2P": {
        1: ["My race is black.", "I am asian."],
        2: [
            "I'm of european descent.",
            "I am caucasian",
            "Im wite",
            "I am African-American",
        ],
        3: [
            "I've visited my parents' home town in Ethiopia.",
            "I grew up hearing Punjabi spoken at home.",
            "We celebrate Saint Nicholas’ Day on December 6th, it’s a family tradition",
        ],
    },
    "MAR": {
        1: [
            "I am married.",
            "I have never been married.",
            "I am legally single."
        ],
        2: [
            "I have a spouse, we got married five years ago.",
            "I just got divorced."
        ],
        3: [
            "I’m Mrs. A now, no longer Miss B.",
            "My wife’s funeral was almost exactly a year ago.",
            "I negotiated a prenup before the wedding, which made things much simpler later.",
            "I’m going to a speed dating event next week hopefully I’ll finally meet someone I like.",
        ],
    },
    "SCHL": {
        1: [
            "I have a regular high school diploma.",
            "I completed my bachelor's degree.",
            "I hold a master's degree.",
        ],
        2: [
            "I finished 12th grade and got a job afterwards.",
            "I dropped out of college after one years.",
            "I completed law school.",
            "I have a PhD in physics.",
        ],
        3: [
            "I left school at sixteen to work full time.",
            "Everyone in my office has at least a bachelor's. I'm the odd one out.",
            "I spent years taking college classes but never walked away with a degree.",
            "I defended my dissertation last spring.",
        ],
    },
    "ESR": {
        1: [
            "I am currently employed and working.",
            "I am unemployed and looking for work.",
            "I am not in the labor force.",
            "I am serving in the armed forces.",
        ],
        2: [
            "I work full time at a marketing firm.",
            "I’m active duty military.",
            "I'm between jobs at the moment.",
            "I’m retired now.",
        ],
        3: [
            "I stay home to take care of my kids.",
            "The company shut down my department, so I’ve been job hunting.",
            "I'm turning 12 next month.",
            "I’m deployed overseas with my unit.",
            "I’m still on the payroll, but I haven’t been into the office in weeks.",
        ],
    },
    "OCCP": {
        1: [
            "I’m an elementary school teacher.",
            "I am a chef.",
            "I work as a mechanical engineer.",
        ],
        2: [
            "I lead a team of IT specialists and make decisions on software systems for the company.",
            "I work in academia as a prof.",
            "I’m a column writer.",
            "I’m a doctor/surgeon.",
        ],
        3: [
            "Most mornings I’m answering emails from my software teams spread across time zones.",
            "Every April I’m swamped double-checking forms for people who can’t figure out their deductions.",
            "I’m usually on my feet all day, moving between patients’ rooms with a cart of meds.",
            "I’m constantly grading stacks of math exams until late in the evening.",
            "I try to work with a variety of ingredients to really surprise the guests.",
        ],
    },
    "CIT": {
        1: [
            "I was born in the United States.",
            "I became a U.S. citizen through naturalization.",
            "I am not a citizen of the United States.",
            "I was born in Puerto Rico."
        ],
        2: [
            "I was born overseas, but my parents are both American.",
            "I grew up in Texas and have been a citizen since birth.",
            "I’m here on a work visa.",
            "I was born in Guam."
        ],
        3: [
            "I vote in U.S. elections every four years.",
            "My passport is from another country.",
            "I didn’t need to apply for citizenship since my parents already had it.",
            "I can’t vote in federal elections here."
        ],
    }
}


PROMPT_HEADER = """<SCENARIO> The TARGET ATTRIBUTES provided for the individual must appear in the text. 

It is important that each attribute value should only be expressed according to the specified DIFFICULTY LEVEL, which determines how easy or hard it is to infer the attribute value. The three levels we consider are listed below. 

(Level 1) In this level, the attribute values are explicitly mentioned in the text exactly as they are written in the TARGET ATTRIBUTES, in a clean, direct, and standard way. Any reader or text anonymization method should be able to immediately identify the attribute values.

(Level 2) In this level, the attribute values are still explicitly present in the text (a reader could identify it without advanced inference or reasoning), but it is nonstandard, obfuscated, or unusual enough that standard text anonymization methods might miss it. The difficulty could arise for instance from slang or colloquial expressions, alternative spellings, nonstandard formatting, partial obfuscation or phonetic forms. It is important that the attribute values, even if obfuscated, remain mentioned explicitly, e.g. a full phone number, address, name or credit card number still needs to be present.

(Level 3) In this level, the attribute values are not explicitly stated in the text. Instead, they are only implied through contextual cues, cultural references, or indirect descriptions. A human reader might infer the attribute with background knowledge or reasoning, but a text anonymization method relying on explicit detection would likely miss it. It is important that the attribute values must be mentioned only in this indirect manner and should never appear as explicit statements. Also try to make this indirect implication as natural as possible. For instance, if the attribute is date of birth, you can subtly mention the age at one point in the conversation and the exact day and month somewhere else. 

It is important that you only mention the attribute values according to the selected DIFFICULTY LEVEL, clearly distinguishing from other levels. 

You will also be provided with example sentences for each attribute. Use these examples as inspiration to understand the task, but do not copy them verbatim. Ensure the generated text is consistent with the user profile and scenario provided, while remaining realistic and varied.\n"""


PROMPT_INFORMATION = """TARGET ATTRIBUTES:
<PROFILE>

DIFFICULTY LEVEL: <DIFFICULTY LEVEL>

EXAMPLES: 
<EXAMPLES>
"""

PROMPT_FOOTER = "<SCENARIO_FORMATTING> The exchange should contain <WORDLIMIT> words. The generated texts will be used for testing purposes. It is of utmost importance that you leak all TARGET ATTRIBUTES at the desired level in a realistic manner consistent with the provided scenario and profile.\n"


SCENARIO = {
    "Medical consultation": """
        You are an AI assistant tasked with generating a fictional, yet plausible transcript of a consultation visit between an individual and a GP doctor. 
        The consultation may cover any topic of interest to the patient, and should feel natural and engaging. 
        Possible topics include, but are not limited to, routine check-ups, recent symptoms, chronic condition management, mental health concerns, medication questions or lifestyle advice.
        This transcript only records the verbal component of the visit. 
        You will be given a set of demographical attributes from the patient PROFILE, for which the consultation transcript should be realistic and consistent.
        """,
    "Chatbot conversation": """
        You are an AI assistant tasked with generating a fictional, yet plausible conversation between an individual and an AI chatbot.
        The conversation may cover any topic of interest to the individual, and should feel natural and engaging. 
        Possible topics include, but are not limited to technical questions, planning or decision-making, creative brainstorming, personal advice or reflections on daily life.
        You will be given a set of demographical attributes from the individual's PROFILE, for which the conversation should be realistic and consistent. 
        The chatbot is not necessarily aware of the demographic attributes of the individual.
        """,
    "Meeting transcript": """
        You are an AI assistant tasked with generating a fictional, yet plausible transcript of a virtual meeting between two individuals, one called 'TARGET' and one called 'OTHER'. 
        The meeting may take place in any professional or semi-professional context and should feel natural and engaging. 
        Possible settings include, but are not limited to, a work-related meeting between two colleagues, a business meeting, a legal consultation, a class or tutoring session, a therapy or coaching meeting, a sales call, or a job interview.
        The transcript should consist solely of spoken dialogue between the two participants.
        You will be given a set of demographical attributes from the TARGET individual's PROFILE, for which the conversation should be realistic and consistent. 
        """
}

SCENARIO_FORMATTING = {
    "Medical consultation": """Format the output exactly as alternating dialogue lines exactly prefixed with 'Patient:' and 'Doctor:' (do not replace these with their respective names), with no scene descriptions. I.e.
     
     [START OF TRANSCRIPT]
     Patient: <PATIENT'S WORDS>
     Doctor: <DOCTOR'S WORDS>
     Patient: <PATIENT'S WORDS>
     Doctor: <DOCTOR'S WORDS>
     etc.
     [END OF TRANSCRIPT]
     
     Do not deviate from this format. Do not include non-spoken components and actions in the transcript.""",
    "Chatbot conversation": """Format the output exactly as alternating dialogue lines exactly prefixed with 'Person:' and 'Chatbot:' (do not replace these with their respective names), with no scene descriptions. I.e.
     
     [START OF TRANSCRIPT]
     Person: <PERSON'S WORDS>
     Chatbot: <CHATBOT'S WORDS>
     Person: <PERSON'S WORDS>
     Chatbot: <CHATBOT'S WORDS>
     etc.
     [END OF TRANSCRIPT]
     
     Do not deviate from this format.
     """,
    "Meeting transcript": """Format the output exactly as alternating dialogue lines exactly prefixed with 'Target:' and 'Other:' (do not replace these with their respective names), with no scene descriptions. I.e.
     
     [START OF TRANSCRIPT]
     Target: <TARGET'S WORDS>
     Other: <OTHER'S WORDS>
     Target: <TARGET'S WORDS>
     Other: <OTHER'S WORDS>
     etc.
     [END OF TRANSCRIPT]
     
     Do not deviate from this format.
     """,
}

MIN_WORDS = {1: 750, 2: 750, 3: 1500} # minimum words per difficulty level
MAX_WORDS = {1: 1000, 2: 1000, 3: 2000} # maximum words per difficulty level

TARGET_ATTRIBUTES_MAP = { # map the PUMS code to a descriptive name if necessary
     "ST": "state of residence",
     "SEX": "sex",
     "DOB": "date of birth",
     "RAC2P": "race",
     "MAR": "marital status",
     "SCHL": "educational attainment",
     "ESR": "employment status",
     "OCCP": "occupation",
     "CIT": "citizenship status",
}

OTHER_LANGUAGE_FOOTER = """
    Importantly, the conversation must be generated in <TARGET_LANGUAGE>. 
    Although the target attributes, their values, and examples are provided above in English, they should be fully translated and adapted to <TARGET_LANGUAGE>. 
    This includes the level of difficulty, which should be expressed in a way that is natural and consistent within the linguistic context of <TARGET_LANGUAGE>."""

### Prompt creation helper functions ###
def get_scenario(scenario) -> str:
    return SCENARIO[scenario]


def get_scenario_output(scenario) -> str:
    return SCENARIO_FORMATTING[scenario]


def get_features(features) -> str:
    featurestr = ""
    for feature in features:
        featurestr = featurestr + feature
        if feature == features[-2]:
            featurestr = featurestr + " and "
        elif feature != features[-1]:
            featurestr = featurestr + ", "
    return featurestr


def get_examples(difficulty, features):
    i = 0
    examplestr = ""
    for feature in features:
        if feature in FEATURE_EXAMPLES:
            if difficulty in FEATURE_EXAMPLES[feature]:
                
                if feature in TARGET_ATTRIBUTES_MAP:
                    feature_name = TARGET_ATTRIBUTES_MAP[feature]
                else:
                    feature_name = feature
                
                examplestr = examplestr + f"Examples for attribute '{feature_name}' mentioned at difficulty level {difficulty}: \n"
                
                for example in FEATURE_EXAMPLES[feature][difficulty]:
                    i = i + 1
                    examplestr = (
                        examplestr
                        + "- Example "
                        + str(i)
                        + ": "
                        + example
                        + "\n"
                    )
    return examplestr


def get_word_limit(difficulty):
    return f"between {MIN_WORDS[difficulty]} and {MAX_WORDS[difficulty]}"

def check_attribute_uppercase(line):
    attribute_list = ["name", "state", "date", "email"]
    for attribute in attribute_list:
        if attribute in line:
            return True
    return False

def prepare_dataentry(dataentry):
    dataentry = dataentry.splitlines()
    outentry = ""
    for line in dataentry:
        if not check_attribute_uppercase(line):
            line = line.lower()
        line = line.capitalize()
        outentry = outentry + line + "\n"
        
    # some cleaning for the profile if needed
    outentry = outentry.replace('type:', '')
    outentry = outentry.replace('description:', '')
    outentry = outentry.replace('n/a:', 'not applicable') 
        
    return outentry

# Main function for prompt creation
def create_generative_prompt(
    scenario: str, dataset: str, features: list, difficulty: int, dataentry: str, language: str = "English"
) -> str:
    
    # sample scenario if requested
    if scenario == "random":
        selected_scenario = random.choice(list(SCENARIO.keys())) 
    else:
        selected_scenario = scenario
    
    prompt_header = PROMPT_HEADER
    prompt_header = prompt_header.replace("<SCENARIO>", get_scenario(selected_scenario))
    prompt_header = prompt_header.replace("<DATASET>", dataset)
    prompt_information = PROMPT_INFORMATION

    prompt_information = prompt_information.replace("<PROFILE>", prepare_dataentry(dataentry))
    prompt_information = prompt_information.replace(
        "<DIFFICULTY LEVEL>", str(difficulty)
    )

    prompt_information = prompt_information.replace(
        "<EXAMPLES>", get_examples(difficulty, features)
    )
    prompt_footer = PROMPT_FOOTER
    prompt_footer = prompt_footer.replace(
        "<SCENARIO_FORMATTING>", get_scenario_output(selected_scenario)
    )
    prompt_footer = prompt_footer.replace(
        "<WORDLIMIT>", get_word_limit(difficulty)
    )

    prompt = prompt_header + "\n" + prompt_information + "\n" +  prompt_footer
    
    if language != "English":
        prompt = prompt + "\n" + OTHER_LANGUAGE_FOOTER.replace("<TARGET_LANGUAGE>", language)
    
    return prompt, selected_scenario


#### Code for generating prompts for obtaining direct identifiers for a data record ####

DIRECT_IDENTIFIER_PROMPT = """You are an AI assistant tasked with generating a list of dummy attributes for a person residing in the United States. The PROFILE of the person is provided below in the format similar to <DATASET>. Also provided is the list of ATTRIBUTES you are to generate.

PROFILE: 
<PROFILE>

ATTRIBUTES:
<DIRECT IDENFITIERS>

Format the output exactly in the following format with no additional text:
<IDENTIFIER_OUTPUT_FORMAT> 

The generated dummy attributes should be consistent with the provided profile, and free of extraneous detail.
"""

IDENTIFIER_LIST = {
    "name": "name: Generate a realistic name for the individual. Use common patterns that reflect the person's gender and origins.",
    "email": "email: Generate a realistic personal email address. Use common patterns that reflect how people actually make emails, such as different first/last name combinations (susan.miller@gmail.com), initials with numbers (shm54@yahoo.com), nicknames (sue_m54@outlook.com), numbers that could be related to birth year, age, or random digits (susanH1971@gmail.com). Make sure the domain is from a popular provider (e.g., Gmail, Yahoo, Outlook, Hotmail).",
    "phone number": "phone number: Generate a realistic US mobile number in a number range used by a common US mobile provider. This number should be consistent with the state of residence of the individual (if available).",
    "address": "address: Generate a realistic US Residential address. This address should be from the same state of residence as their mobile number and correspond to a real town/city in that state. This address should be reasonable for the individual given their background and occupation. This address should be of the appropriate type given the location (e.g. house for a suburb, apartment for a city centre). Include an accurate ZIP code in the address that reflects the location of the residence. Use an appropriate random number for the street number (don't use 123). ",
}

IDENTIFIER_LIST_OUTPUT_PUMS = {
    "name": "name: <NAME>",
    "email": "email: <EMAIL>",
    "phone number": "phone number: <PHONE NUMBER>",
    "address": "address: <RESIDENTIAL ADDRESS>",
}

def get_identifiers(identifiers: list) -> str:
    identifier_str = ""
    for identifier in identifiers:
        identifier_str = identifier_str + IDENTIFIER_LIST[identifier] + "\n"
    return identifier_str


def get_identifier_output(identifiers: list, dataset: str) -> str:
    identifier_str = ""
    for identifier in identifiers:
        identifier_str = (
            identifier_str + IDENTIFIER_LIST_OUTPUT_PUMS[identifier] + "\n"
        )
    return identifier_str


# Main function for creating prompts for generating direct identifiers
def create_direct_identifiers_prompt(
    dataset: str, identifiers: list, dataentry: str
) -> str:
    prompt = DIRECT_IDENTIFIER_PROMPT
    prompt = prompt.replace("<DATASET>", dataset)
    prompt = prompt.replace("<PROFILE>", dataentry)
    prompt = prompt.replace(
        "<DIRECT IDENFITIERS>", get_identifiers(identifiers)
    )
    prompt = prompt.replace(
        "<IDENTIFIER_OUTPUT_FORMAT>",
        get_identifier_output(identifiers, dataset),
    )
    return prompt
