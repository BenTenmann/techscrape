import re
import json
import requests
import numpy as np
import pandas as pd


class TerminalColors:
    """
    Object class for terminal text colors
    """

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_log(**log):
    """
    Helper function to pint logs, newline delimited
    :param log: a dictionary of logs
    :return: string representation
    """
    s = 'LOG: \n'
    for key, value in log.items():
        s += f'     {key} : {value}\n'
    return s


def is_str(type_name='method'):
    """
    Checks type on function or method input; function can only have a single input
    :param type_name: defines whether it is a function or method to be wrapped
    :return: wrapped function
    """
    if type_name == 'function':
        def type_check(function):
            def checked(text):
                if type(text) == str:
                    result = function(text)
                    return result
                raise TypeError(f'input `str` -- currently {type(text)}')

            return checked
    elif type_name == 'method':
        def type_check(function):
            def checked(cls, text):
                if type(text) == str:
                    result = function(cls, text)
                    return result
                raise TypeError(f'input `str` -- currently {type(text)}')

            return checked
    else:
        raise ValueError(f'`type_name` must be `function` or `method` -- currently {type_name}')
    return type_check


def empty(container: list) -> list:
    """
    Create a `NA` string to replace an empty list
    :param container: the container
    :return: either the original container or [`NA`]
    """
    if len(container) == 0:
        return ['NA']
    return container


def geolocation(location: str, axis: str) -> float:
    """
    Get the longitudinal or latitudinal coordinates for a location
    :param location: location string
    :param axis: `lon` or `lat`
    :return:
    """
    location = location.lower().strip().replace(' ', '+')
    query = f'https://nominatim.openstreetmap.org/search?q={location}' \
            f'&format=json'
    result = json.loads(requests.get(query).text)
    if type(result) == list:
        if len(result) == 0:
            return np.nan
        return float(result[0][axis])
    return float(result[axis])


def to_json(data: dict, filename: str, mode: str = 'w') -> None:
    """
    Write a data dictionary to a .json file
    :param data: data dictionary
    :param filename: filename to be saved to
    :param mode: the writing mode (default=`w`)
    :return: None
    """
    with open(filename, mode) as file:
        file.write(
            json.dumps(data, indent=4)
        )
    file.close()
    return None


def read_json(filename: str) -> dict:
    """
    Read a .json files into a dictionary
    :param filename: path to the file
    :return: the dictionary from the file
    """
    with open(filename, 'r') as file:
        j = file.read()
    file.close()
    return json.loads(j)


def is_deprecated(function):
    """
    Wrapper for deprecation warning
    :param function: function to be wrapped
    :return: wrapped function
    """
    def deprecated_function(*args, **kwargs):
        print(
            f'{TerminalColors.WARNING}'
            f'The method or function {function.__name__} is deprecated. '
            f'Please do not use in the future.'
            f'{TerminalColors.ENDC}'
        )
        result = function(*args, **kwargs)
        return result
    return deprecated_function


def get_roc(prediction: list, target: list, threshold: float) -> tuple:
    """
    Get the true-positive and false-positive rates from a set of predictions
    and targets with a given threshold
    :param prediction: list of predictions
    :param target: list of targets
    :param threshold: the threshod for the predictions
    :return: a tuple of floats (tp, fp)
    """
    tp = fp = 0
    p = n = 0
    for pred, targ in [*zip(prediction, target)]:
        classification = pred >= threshold
        if classification and targ:
            tp += 1
        if classification and not targ:
            fp += 1
        p += targ
        n += not targ
    return tp / p, fp / n


def base_clean(df_: pd.Series) -> pd.Series:
    """
    Base clean of company strings to feed into the model
    :param df_: pd.Series of company names
    :return: cleaned pd.Series of company names
    """
    REGEX = [
        (  # remove non-ws delimiters and special characters
            re.compile(
                r"[.,\-&'+™]"
            ), ""),
        (  # remove company jargon
            re.compile(
                r"(inc|ltd|pvt|gmbh|llc|ag|co|limited)(?!\w)"
            ), ""),
        (  # remove bracket-comments
            re.compile(
                r"(\(.*\))"
            ), ""),
        (  # remove . after @ symbol
            re.compile(
                r"@.*"
            ), ""),
        (  # strip ws
            re.compile(
                r"(^ +| +$)"
            ), ""),
        (  # replace >= 1 ws
            re.compile(
                r"[ ]+"
            ), "_")
    ]

    for pattern, replace in REGEX:
        df_ = df_.apply(lambda x: pattern.sub(replace, x))

    return df_


def clean(filename: str) -> pd.DataFrame:
    """
    Clean the company names
    :param filename: csv file to load
    :return: pandas dataframe
    """
    df = pd.read_csv(filename)
    df.dropna(inplace=True)

    # basic clean
    df_ = df['company_name'].apply(
        lambda x:
        x.strip()
    )

    df['company_name'] = base_clean(df_)
    return df


class DummyMatchObject:
    """
    A dummy match object; used for packing
    """
    def __init__(self, start, end):
        self.s = start
        self.e = end

    def start(self, n):
        if n == 0:
            return self.s
        raise ValueError

    def end(self, n):
        if n == 0:
            return self.e
        raise ValueError


def pack(all_matches: list) -> list:
    """
    Pack the matches for further processing
    :param all_matches: list of re.Match objects
    :return: list of tuples of shifted re.Match objects
    """
    zipped = [*zip(all_matches[:-1], all_matches[1:]),
              (all_matches[-1], DummyMatchObject(all_matches[-1].end(0) + 500,
                                                 all_matches[-1].end(0) + 1000))]
    return zipped


class WaitingTime:
    """
    WaitingTime object for CrunchBase scraping
    """
    def __init__(self, constant, noise):
        self.constant = constant
        self.noise = noise

    def __call__(self):
        return self.constant + self.noise.rvs()
