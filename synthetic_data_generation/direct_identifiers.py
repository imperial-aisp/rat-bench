import argparse
import json
import random
from datetime import date
import numpy as np
import pandas as pd
import os

## everything for name generation
# let's always load the data just once
PATH_TO_DATA = "./data/"

FIRST_NAME_DF = pd.read_csv(os.path.join(PATH_TO_DATA, "first_name_all_years.csv"))
LAST_NAME_DF = pd.read_csv(os.path.join(PATH_TO_DATA, "last_name.csv"))

# first_name_all_years.csv has gaps (e.g. no freq_1918, no freq_2018 column at
# all) and doesn't cover every year down to get_full_name's min_year=1880
# default (data starts at 1900) -- get_full_name() snaps to the nearest year
# in this list rather than assuming freq_{yob} always exists.
_AVAILABLE_NAME_YEARS = sorted(
    int(c.split("_", 1)[1]) for c in FIRST_NAME_DF.columns if c.startswith("freq_")
)


def _nearest_available_name_year(yob):
    return min(_AVAILABLE_NAME_YEARS, key=lambda y: abs(y - yob))

# Mexican name data — loaded lazily so PUMS-only runs are unaffected
_MEX_FIRST_NAME_DF = None
_MEX_LAST_NAME_DF = None

# Serbian name data — hardcoded (all SRB profiles are from a women's survey)
_SRB_FEMALE_FIRST_NAMES = [
    "Ana", "Marija", "Jelena", "Milica", "Jovana", "Ivana", "Katarina",
    "Dragana", "Slavica", "Vesna", "Snežana", "Biljana", "Maja", "Sandra",
    "Tijana", "Nina", "Aleksandra", "Zorica", "Ljubica", "Gordana",
    "Natalija", "Tamara", "Sanja", "Svetlana", "Jasmina", "Mirjana",
    "Radmila", "Danijela", "Nevena", "Kristina",
]
_SRB_LAST_NAMES = [
    "Jovanović", "Petrović", "Nikolić", "Marković", "Đorđević",
    "Stojanović", "Ilić", "Stanković", "Popović", "Lazarević",
    "Simić", "Savić", "Milošević", "Radovanović", "Stefanović",
    "Filipović", "Đurić", "Vasić", "Ristić", "Pavlović",
    "Kostić", "Bogdanović", "Todorović", "Kovačević", "Živković",
    "Arsić", "Vuković", "Ninković", "Milenković", "Lukić",
]

def _load_mex_names():
    global _MEX_FIRST_NAME_DF, _MEX_LAST_NAME_DF
    if _MEX_FIRST_NAME_DF is None:
        _MEX_FIRST_NAME_DF = pd.read_csv(os.path.join(PATH_TO_DATA, "es/MEX_names.csv"))
        _MEX_LAST_NAME_DF = pd.read_csv(os.path.join(PATH_TO_DATA, "es/mexico_surnames.csv"))


def get_full_name_mex(sex: str) -> str:
    """Sample a full Mexican name conditioned on sex ('Mujer' or 'Hombre')."""
    _load_mex_names()
    gender = "Female" if sex == "Mujer" else "Male"
    sub = _MEX_FIRST_NAME_DF[_MEX_FIRST_NAME_DF["gender"] == gender].copy()
    total = sub["frequency"].sum()
    first = np.random.choice(sub["name"].values, p=sub["frequency"].values / total)
    first = first.strip().capitalize()
    total_s = _MEX_LAST_NAME_DF["incidence"].sum()
    last = np.random.choice(
        _MEX_LAST_NAME_DF["surname"].values,
        p=_MEX_LAST_NAME_DF["incidence"].values / total_s,
    )
    return first + " " + last.strip().capitalize()


_CURP_VOWELS = "AEIOU"
_CURP_CONSONANTS = "BCDFGHJKLMNÑPQRSTVWXYZ"
_MEX_STATE_CODES = [
    "AS", "BC", "BS", "CC", "CL", "CM", "CS", "CH", "DF", "DG",
    "GT", "GR", "HG", "JC", "MC", "MN", "MS", "NT", "NL", "OC",
    "PL", "QT", "QR", "SP", "SL", "SR", "TC", "TS", "TL", "VZ",
    "YN", "ZS",
]

def generate_curp(birth_year: int | None = None) -> str:
    """Generate a plausible but fake 18-character CURP.

    CURP encodes the holder's real birth date in date_part (YYMMDD) and
    disambiguates century via a differentiating character at position 17
    (a digit 0-9 for pre-2000 births, a letter for 2000+ -- real CURPs vary
    which letter by a fuller homoclave rule, "A" here is just one valid
    example). Pass the profile's actual birth_year so this doesn't
    contradict the age/DOB already shown elsewhere in the profile; falls
    back to an independently-random year and century marker if not given.
    """
    letters = (
        random.choice("BCDFGHJKLMNPQRSTVWXYZ")
        + random.choice(_CURP_VOWELS)
        + random.choice("BCDFGHJKLMNPQRSTVWXYZ")
        + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )
    if birth_year is not None:
        yy = birth_year % 100
        century = "0" if birth_year < 2000 else "A"
    else:
        yy = random.randint(0, 99)
        century = random.choice("0123456789A")
    mm = random.randint(1, 12)
    dd = random.randint(1, 28)
    date_part = f"{yy:02d}{mm:02d}{dd:02d}"
    sex_char = random.choice(["H", "M"])
    state = random.choice(_MEX_STATE_CODES)
    consonants = "".join(random.choice(_CURP_CONSONANTS) for _ in range(3))
    check = str(random.randint(0, 9))
    return letters + date_part + sex_char + state + consonants + century + check


_MEX_LADAS = ["55", "33", "81", "222", "229", "477", "442", "614", "667", "998"]

def generate_mexican_phone() -> str:
    """Generate a realistic 10-digit Mexican mobile number."""
    lada = random.choice(_MEX_LADAS)
    remaining = 10 - len(lada)
    digits = "".join(str(random.randint(0, 9)) for _ in range(remaining))
    return lada + digits


_MEX_STREETS = [
    "Insurgentes", "Reforma", "Hidalgo", "Juárez", "Morelos", "Revolución",
    "Independencia", "Benito Juárez", "Miguel Hidalgo", "Francisco Madero",
    "Venustiano Carranza", "Emiliano Zapata", "Lázaro Cárdenas", "Álvaro Obregón",
    "Cinco de Mayo", "16 de Septiembre", "Constitución", "República",
]
_MEX_COLONIAS = [
    "Centro", "Roma Norte", "Condesa", "Polanco", "Del Valle", "Narvarte",
    "Doctores", "Santa María la Ribera", "San Rafael", "Coyoacán",
    "Tlalpan", "Pedregal", "Ecatepec", "Lindavista", "Portales",
]
_MEX_CITY_STATE = [
    ("Ciudad de México", "CDMX"), ("Guadalajara", "Jalisco"), ("Monterrey", "Nuevo León"),
    ("Puebla", "Puebla"), ("Tijuana", "Baja California"), ("León", "Guanajuato"),
    ("Mérida", "Yucatán"), ("Cancún", "Quintana Roo"), ("San Luis Potosí", "San Luis Potosí"),
    ("Querétaro", "Querétaro"), ("Culiacán", "Sinaloa"), ("Hermosillo", "Sonora"),
    ("Chihuahua", "Chihuahua"), ("Zapopan", "Jalisco"), ("Ecatepec", "Estado de México"),
]

def get_full_name_srb() -> str:
    """Sample a full Serbian female name (all SRB profiles are from a women's survey)."""
    first = random.choice(_SRB_FEMALE_FIRST_NAMES)
    last = random.choice(_SRB_LAST_NAMES)
    return f"{first} {last}"


_SRB_REGION_CODES = ["71", "72", "73", "74", "75"]


def _jmbg_check(digits12: str) -> int:
    """Compute JMBG check digit from first 12 digits. Returns 10 if invalid."""
    d = [int(c) for c in digits12]
    s = (7*(d[0]+d[6]) + 6*(d[1]+d[7]) + 5*(d[2]+d[8]) +
         4*(d[3]+d[9]) + 3*(d[4]+d[10]) + 2*(d[5]+d[11]))
    k = 11 - (s % 11)
    return 0 if k == 11 else k


def generate_jmbg(birth_year: int | None = None) -> str:
    """Generate a plausible but fake 13-digit Serbian JMBG.

    JMBG encodes the holder's real birth date (DDMMGGG, where GGG is the
    last 3 digits of the year -- a leading 9 means 1900s, a leading 0 means
    2000s, which falls out automatically from taking the last 3 digits of
    any 4-digit year). Pass the profile's actual birth_year so this doesn't
    contradict the age/DOB already shown elsewhere in the profile; falls
    back to a random 1970-2004 year if not given.
    """
    while True:
        dd = random.randint(1, 28)
        mm = random.randint(1, 12)
        yy = birth_year if birth_year is not None else random.randint(1970, 2004)
        yyy = str(yy)[-3:]
        rr = random.choice(_SRB_REGION_CODES)
        bbb = str(random.randint(500, 999))  # 500–999 = female range
        base = f"{dd:02d}{mm:02d}{yyy}{rr}{bbb}"
        k = _jmbg_check(base)
        if k != 10:
            return base + str(k)


_SRB_MOBILE_PREFIXES = ["060", "061", "062", "063", "064", "065", "066", "069"]


def generate_serbian_phone() -> str:
    """Generate a realistic Serbian mobile phone number."""
    prefix = random.choice(_SRB_MOBILE_PREFIXES)
    digits = "".join(str(random.randint(0, 9)) for _ in range(7))
    return f"{prefix}/{digits[:3]}-{digits[3:]}"


_SRB_STREETS = [
    "Knez Mihailova", "Kralja Aleksandra", "Makedonska", "Nemanjina",
    "Terazije", "Savska", "Vojvode Stepe", "Bulevar oslobođenja",
    "Cara Dušana", "Svetogorska", "Francuska", "Nušićeva",
    "Zmaj Jovina", "Obilićev venac", "Jurija Gagarina",
]
_SRB_CITIES = [
    ("Beograd", "11000"), ("Novi Sad", "21000"), ("Niš", "18000"),
    ("Kragujevac", "34000"), ("Subotica", "24000"), ("Zrenjanin", "23000"),
    ("Pančevo", "26000"), ("Čačak", "32000"), ("Leskovac", "16000"),
    ("Smederevo", "11300"),
]


def generate_serbian_address() -> str:
    """Generate a plausible Serbian residential address."""
    street = random.choice(_SRB_STREETS)
    number = random.randint(2, 150)
    city, postal = random.choice(_SRB_CITIES)
    return f"{street} {number}, {postal} {city}, Srbija"


def generate_mexican_address() -> str:
    """Generate a plausible Mexican residential address."""
    street = random.choice(_MEX_STREETS)
    number = random.randint(2, 999)
    colonia = random.choice(_MEX_COLONIAS)
    city, state = random.choice(_MEX_CITY_STATE)
    return f"Calle {street} #{number}, Col. {colonia}, {city}, {state}, México"


# ── Flemish/Belgian (NL) name and identifier generators ───────────────────────

_NL_MALE_FIRST_NAMES = [
    "Liam", "Ruben", "Finn", "Lars", "Mathis", "Pieter", "Thomas", "Luca",
    "Noel", "Arne", "Wout", "Bram", "Jens", "Jonas", "Sander", "Kobe",
    "Niels", "Joris", "Stef", "Maarten", "Wouter", "Bert", "Tim", "Kevin",
    "Alexander", "Nicolas", "Simon", "Michiel", "Kristof", "Dieter",
]
_NL_FEMALE_FIRST_NAMES = [
    "Emma", "Olivia", "Nora", "Lena", "Fien", "Julie", "Laura", "Sara",
    "Elien", "Amber", "Sofie", "Lisa", "Lore", "An", "Ines",
    "Charlotte", "Hannah", "Katrien", "Lies", "Nathalie", "Elke", "Silke",
    "Annelies", "Manon", "Ilse", "Griet", "Lien", "Ellen", "Karen", "Hailey",
]
_NL_LAST_NAMES = [
    "De Smedt", "Janssen", "Maes", "Claes", "Willems", "Peeters", "De Backer",
    "Hermans", "Wouters", "Smeets", "Mertens", "Jacobs", "Van den Berg",
    "De Graef", "Pieters", "Stevens", "Dubois", "Lambert", "Leclercq",
    "Desmet", "Vermeersch", "Van Acker", "Bogaert", "Cools", "Nijs",
    "De Wolf", "Claessens", "Goossens", "Hendrickx", "Martens",
]


def get_full_name_nl(sex: str) -> str:
    """Sample a full Flemish name conditioned on sex ('Male' or 'Female')."""
    if sex == "Female":
        first = random.choice(_NL_FEMALE_FIRST_NAMES)
    else:
        first = random.choice(_NL_MALE_FIRST_NAMES)
    last = random.choice(_NL_LAST_NAMES)
    return f"{first} {last}"


def generate_rrn(birth_year: int | None = None) -> str:
    """Generate a plausible but fake Belgian rijksregisternummer (YY.MM.DD-NNN.CC).

    RRN encodes the holder's real birth date, and its checksum depends on
    century: for 2000+ births the 9-digit base is conceptually prefixed
    with "2" (i.e. treated as 2,000,000,000 + base) before the mod-97
    check, while pre-2000 births use the 9-digit base directly. Pass the
    profile's actual birth_year so this doesn't contradict the age/DOB
    already shown elsewhere in the profile -- this matters here more than
    for JMBG/CURP, since real NL survey ages in this dataset can be single
    digits, i.e. genuinely born in the 2000s, not just theoretically so.
    Falls back to a random pre-2000 year (simpler checksum case) if not given.
    """
    if birth_year is not None:
        yy = birth_year % 100
        post_2000 = birth_year >= 2000
    else:
        yy = random.randint(40, 99)
        post_2000 = False
    mm = random.randint(1, 12)
    dd = random.randint(1, 28)
    nnn = random.randint(1, 998)
    base = int(f"{yy:02d}{mm:02d}{dd:02d}{nnn:03d}")
    checksum_base = base + 2_000_000_000 if post_2000 else base
    cc = 97 - (checksum_base % 97)
    if cc == 0:
        cc = 97
    return f"{yy:02d}.{mm:02d}.{dd:02d}-{nnn:03d}.{cc:02d}"


_NL_MOBILE_PREFIXES = [
    "0470", "0471", "0472", "0473", "0474", "0475",
    "0476", "0477", "0478", "0479", "0485", "0486",
    "0487", "0488", "0489", "0494", "0495", "0496",
]


def generate_nl_phone() -> str:
    """Generate a realistic Belgian mobile phone number."""
    prefix = random.choice(_NL_MOBILE_PREFIXES)
    digits = "".join(str(random.randint(0, 9)) for _ in range(6))
    return f"{prefix} {digits[:2]} {digits[2:4]} {digits[4:]}"


_NL_STREETS = [
    "Kerkstraat", "Stationstraat", "Schoolstraat", "Dorpsstraat", "Molenstraat",
    "Nieuwstraat", "Kasteeldreef", "Lindenlaan", "Bosstraat", "Veldstraat",
    "Antwerpsestraat", "Gentsestraat", "Brugsestraat", "Kapelstraat",
    "Vrijheidslaan", "Mechelsesteenweg", "Leuvensesteenweg", "Ringlaan",
]
_NL_CITIES = [
    ("Gent", "9000"), ("Antwerpen", "2000"), ("Brugge", "8000"),
    ("Leuven", "3000"), ("Hasselt", "3500"), ("Mechelen", "2800"),
    ("Kortrijk", "8500"), ("Aalst", "9300"), ("Sint-Niklaas", "9100"),
    ("Genk", "3600"), ("Roeselare", "8800"), ("Turnhout", "2300"),
]


def generate_nl_address() -> str:
    """Generate a plausible Flemish residential address."""
    street = random.choice(_NL_STREETS)
    number = random.randint(2, 150)
    city, postal = random.choice(_NL_CITIES)
    return f"{street} {number}, {postal} {city}, België"


def get_full_name(gender, age, min_year=1880, max_year=2024):
    '''
    Generate a full name based on gender and age.
    Input: 
        gender: 'M' or 'F'
        age: integer
              
    The first name is sampled from the actual distribution of baby names, conditioned on both year of birth and gender. 
    Source: https://www.ssa.gov/oact/babynames/limits.html
    
    The last name is sampled from the actual distribution for last names more frequent than 1000 occurrences from the US Census 2010.
    This is not dependent on gender, nor on year of birth.
    Source: https://www.census.gov/topics/population/genealogy/data.html
    '''
    
    year_today = date.today().year
    yob = year_today - int(age)

    yob = max(yob, min_year)
    yob = min(yob, max_year)

    if f"freq_{yob}" not in FIRST_NAME_DF.columns:
        yob = _nearest_available_name_year(yob)

    # sample first name
    sub_df = FIRST_NAME_DF[(FIRST_NAME_DF['gender'] == gender) & (FIRST_NAME_DF[f"freq_{yob}"] > 0)]
    first_name = np.random.choice(sub_df['first_name'].values, p=sub_df[f"freq_{yob}"].values)
    
    # sample last name
    last_name = np.random.choice(LAST_NAME_DF['last_name'].values, p=LAST_NAME_DF['last_name_frequency'].values)
    
    return first_name + ' ' + last_name

def checkSSNvalid(SSN):
    # Check if all digits are same
    firstdigit = SSN[0]
    digit_all_same_flag = True
    for c in SSN:
        if c != firstdigit:
            digit_all_same_flag = False

    if digit_all_same_flag:
        return False

    return True


def generate_SSN():
    # SSNs are comprised of 3 parts, Area Number, Group Number, Serial Number
    SSN = ""

    # Generate Area Number, Area number cannot be 000, 900-999 or 666
    AreaNumber = 666
    while AreaNumber == 666:
        AreaNumber = random.randint(1, 899)
    GroupNumber = random.randint(1, 99)
    SerialNumber = random.randint(1, 9999)
    if AreaNumber < 100:
        SSN = SSN + "0"
        if AreaNumber < 10:
            SSN = SSN + "0"
    SSN = SSN + str(AreaNumber) + "-"

    # Generate Group Number, Group number cannot be 00
    if GroupNumber < 10:
        SSN = SSN + "0"
    SSN = SSN + str(GroupNumber) + "-"

    # Generate Serial Number, Serial number cannot be 00
    if SerialNumber < 1000:
        SSN = SSN + "0"
        if SerialNumber < 100:
            SSN = SSN + "0"
            if SerialNumber < 10:
                SSN = SSN + "0"
    SSN = SSN + str(SerialNumber)

    # SSNs cannot have all digits the same
    if checkSSNvalid(SSN) == False:
        SSN = generate_SSN()

    return SSN


def luhn_checksum(card_number: str) -> int:
    """Calculate the Luhn checksum for validation."""

    def digits_of(n):
        return [int(d) for d in str(n)]

    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for d in even_digits:
        total += sum(digits_of(d * 2))
    return total % 10


def generate_card_number(prefix: str, length: int) -> str:
    """Generate a card number with given prefix and length that passes Luhn check."""
    number = prefix
    while len(number) < (length - 1):
        number += str(random.randint(0, 9))

    # calculate check digit
    check_digit = [
        str(d) for d in range(10) if luhn_checksum(number + str(d)) == 0
    ][0]
    return number + check_digit


def generate_card():
    issuer = random.choice(
        ["visa", "mastercard", "amex", "discover", "diners", "jcb"]
    )
    """Generate dummy card numbers by issuer."""
    issuers = {
        "visa": ("4", 16),
        "mastercard": (str(random.choice(range(51, 56))), 16),
        "amex": (str(random.choice(["34", "37"])), 15),
        "discover": ("6011", 16),
        "diners": (
            str(
                random.choice(
                    ["300", "301", "302", "303", "304", "305", "36", "38"]
                )
            ),
            14,
        ),
        "jcb": ("35", 16),
    }

    if issuer.lower() not in issuers:
        raise ValueError(
            "Unknown issuer. Choose from: " + ", ".join(issuers.keys())
        )

    prefix, length = issuers[issuer.lower()]
    card = generate_card_number(prefix, length)
    return card


MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}



def generate_birthday(age: int) -> str:
    today = date.today()
    today = str(today).split("-")
    year = int(today[0])
    month = int(today[1])
    day = int(today[2])

    # Get Year and Month of birth
    year_of_birth = year - int(age)
    month_of_birth = random.randint(1, 12)

    # Compute Leap year correctly by checking current day
    if (month > 2) or (month == 2 and day == 29):
        # If current day is Feb 29th or later, we wont subtract one year later if we randomly sample Feb 29th
        if year_of_birth % 4 == 0 and (
            year_of_birth % 100 != 0 or year_of_birth % 400 == 0
        ):
            is_leap_year = True
        else:
            is_leap_year = False
    else:
        # If current day is Feb 28th or earlier, we will subtract one year later
        if (year_of_birth-1) % 4 == 0 and (
            (year_of_birth-1) % 100 != 0 or (year_of_birth-1) % 400 == 0
        ):
            is_leap_year = True
        else:
            is_leap_year = False

    # Get Day of Birth - factoring in month length and leap year
    if month_of_birth in [1, 3, 5, 7, 8, 10, 12]:
        day_of_birth = random.randint(1, 31)
    elif month_of_birth == 2:
        if is_leap_year:
            day_of_birth = random.randint(1, 29)
        else:
            day_of_birth = random.randint(1, 28)
    else:
        day_of_birth = random.randint(1, 30)

    # Subtract an additional year from YOB if the date chosen is after today's date. 
    # If today is Jan 23rd 2026, someone born on December 25th 2000 would be 25 years old, not 26.
    if month_of_birth > month or (
        month_of_birth == month and day_of_birth > day
    ):
        year_of_birth = year_of_birth - 1

    DOB = (
        str(day_of_birth)
        + " "
        + MONTHS[month_of_birth]
        + " "
        + str(year_of_birth)
    )
    return DOB


# ── Batch direct-identifier generation for the multilingual "new profiles" ────
# Mirrors extend_seed_profiles.py::add_direct_identifiers for English: adds
# name/national-ID/credit-card/phone/address columns to a raw profile
# dataframe (one row per profile), using the age-consistent generators above
# so the embedded birth date in JMBG/CURP/RRN matches each row's real age.
#
# These read the tiny sex-encoding maps directly (rather than importing the
# decoding helpers from data.py) since data.py imports FROM this module --
# importing back would be circular.

def _invert_json_map(path):
    with open(path, "r", encoding="utf-8") as f:
        return {v: k for k, v in json.load(f).items()}


def add_direct_identifiers_srb(df: pd.DataFrame) -> pd.DataFrame:
    """Add name/JMBG/credit card/phone/address columns to a Serbian raw
    profile dataframe (must have an 'age' column; one row per profile).
    All SRB profiles are from a women's survey, so name/JMBG generation
    doesn't need a decoded sex value.
    """
    df = df.copy()
    today_year = date.today().year
    names, jmbgs, cards, phones, addresses = [], [], [], [], []
    for _, row in df.iterrows():
        birth_year = today_year - int(row["age"])
        names.append(get_full_name_srb())
        jmbgs.append(generate_jmbg(birth_year=birth_year))
        cards.append(generate_card())
        phones.append(generate_serbian_phone())
        addresses.append(generate_serbian_address())
    df["name"] = names
    df["JMBG"] = jmbgs
    df["credit card number"] = cards
    df["phone number"] = phones
    df["address"] = addresses
    return df


def add_direct_identifiers_mex(df: pd.DataFrame) -> pd.DataFrame:
    """Add name/CURP/credit card/phone/address columns to a Mexican raw
    profile dataframe (must have 'EDAD' and 'SEXO' columns; one row per profile).
    """
    df = df.copy()
    today_year = date.today().year
    sexo_map = _invert_json_map("./data/es/maps/SEXO_map.json")  # {1: "Mujer", 2: "Hombre"}
    names, curps, cards, phones, addresses = [], [], [], [], []
    for _, row in df.iterrows():
        birth_year = today_year - int(row["EDAD"])
        sexo = sexo_map.get(int(row["SEXO"]), "Hombre")
        names.append(get_full_name_mex(sexo))
        curps.append(generate_curp(birth_year=birth_year))
        cards.append(generate_card())
        phones.append(generate_mexican_phone())
        addresses.append(generate_mexican_address())
    df["name"] = names
    df["CURP"] = curps
    df["credit card number"] = cards
    df["phone number"] = phones
    df["address"] = addresses
    return df


def add_direct_identifiers_nl(df: pd.DataFrame) -> pd.DataFrame:
    """Add name/RRN/credit card/phone/address columns to a Dutch raw
    profile dataframe (must have 'age' and 'sex' columns; one row per profile).
    """
    df = df.copy()
    today_year = date.today().year
    sex_map = _invert_json_map("./data/nl/maps/sex_map.json")  # {1: "Female", 2: "Male"}
    names, rrns, cards, phones, addresses = [], [], [], [], []
    for _, row in df.iterrows():
        birth_year = today_year - int(row["age"])
        sex = sex_map.get(int(row["sex"]), "Male")
        names.append(get_full_name_nl(sex))
        rrns.append(generate_rrn(birth_year=birth_year))
        cards.append(generate_card())
        phones.append(generate_nl_phone())
        addresses.append(generate_nl_address())
    df["name"] = names
    df["RRN"] = rrns
    df["credit card number"] = cards
    df["phone number"] = phones
    df["address"] = addresses
    return df


# JMBG is a pure-digit string that can start with 0 (day 01-09, or a 2000s
# birth year's "0YY" GGG field) -- a plain pd.read_csv() infers that whole
# column as int64 and silently drops the leading digit (verified: strips it
# on ~1/3 of a real 200-row batch). CURP/RRN are safe (they contain letters
# or punctuation), so only JMBG needs this. Always load via
# load_new_profiles_with_ids() below, not a bare pd.read_csv().
_ID_COLUMN_DTYPES = {"srb": {"JMBG": str}, "mex": None, "nl": None}

LANG_CONFIG = {
    "srb": {
        "input_csv": "data/srb/200_new_profiles.csv",
        "output_csv": "data/srb/200_new_profiles_with_ids.csv",
        "add_fn": add_direct_identifiers_srb,
    },
    "mex": {
        "input_csv": "data/es/200_new_profiles.csv",
        "output_csv": "data/es/200_new_profiles_with_ids.csv",
        "add_fn": add_direct_identifiers_mex,
    },
    "nl": {
        "input_csv": "data/nl/200_new_profiles.csv",
        "output_csv": "data/nl/200_new_profiles_with_ids.csv",
        "add_fn": add_direct_identifiers_nl,
    },
}


def load_new_profiles_with_ids(lang: str) -> pd.DataFrame:
    """Load a language's *_new_profiles_with_ids.csv with the correct dtype
    for its national-ID column, so JMBG's leading zeros survive the round
    trip. Use this instead of a bare pd.read_csv(LANG_CONFIG[lang]["output_csv"]).
    """
    cfg = LANG_CONFIG[lang]
    return pd.read_csv(cfg["output_csv"], dtype=_ID_COLUMN_DTYPES[lang])


def main(languages=("srb", "mex", "nl"), seed=None):
    """Generate direct identifiers (name, national ID, credit card, phone,
    address) for each language's newly-selected profiles (the *_new_profiles.csv
    files produced by weighted_sampling_.py's n_new extension) and write them
    back out as a new CSV alongside the input, with the identifier columns
    appended. Never touches the original 100_profiles.csv for any language.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for lang in languages:
        cfg = LANG_CONFIG[lang]
        print(f"[{lang}] Loading {cfg['input_csv']}...")
        df = pd.read_csv(cfg["input_csv"])
        print(f"[{lang}] Generating direct identifiers for {len(df)} profiles...")
        df = cfg["add_fn"](df)
        df.to_csv(cfg["output_csv"], index=False)
        print(f"[{lang}] Wrote {len(df)} profiles with identifiers to {cfg['output_csv']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", type=str, default="srb,mex,nl",
                        help="Comma-separated subset of srb,mex,nl to process.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(languages=[l.strip() for l in args.languages.split(",")], seed=args.seed)
