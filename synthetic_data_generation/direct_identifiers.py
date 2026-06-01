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

def generate_curp() -> str:
    """Generate a plausible but fake 18-character CURP."""
    letters = (
        random.choice("BCDFGHJKLMNPQRSTVWXYZ")
        + random.choice(_CURP_VOWELS)
        + random.choice("BCDFGHJKLMNPQRSTVWXYZ")
        + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )
    yy = random.randint(0, 99)
    mm = random.randint(1, 12)
    dd = random.randint(1, 28)
    date_part = f"{yy:02d}{mm:02d}{dd:02d}"
    sex_char = random.choice(["H", "M"])
    state = random.choice(_MEX_STATE_CODES)
    consonants = "".join(random.choice(_CURP_CONSONANTS) for _ in range(3))
    century = random.choice("0123456789A")
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


def generate_jmbg() -> str:
    """Generate a plausible but fake 13-digit Serbian JMBG."""
    while True:
        dd = random.randint(1, 28)
        mm = random.randint(1, 12)
        yy = random.randint(1970, 2004)
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


def generate_rrn() -> str:
    """Generate a plausible but fake Belgian rijksregisternummer (YY.MM.DD-NNN.CC)."""
    yy = random.randint(40, 99)
    mm = random.randint(1, 12)
    dd = random.randint(1, 28)
    nnn = random.randint(1, 998)
    base = int(f"{yy:02d}{mm:02d}{dd:02d}{nnn:03d}")
    cc = 97 - (base % 97)
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
