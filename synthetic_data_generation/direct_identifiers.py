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
