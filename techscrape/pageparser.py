import nltk
import torch
import pandas as pd
from .utils.helpers import base_clean


def name_to_tensor(name, vocab):
    """
    Make a company into a tensor of integers -- class indices
    :param name: name of the company
    :param vocab: vocabulary of the model
    :return: tensor of len(name) and type torch.int32
    """
    return torch.tensor([[vocab[key]] for key in name], dtype=torch.int32)


def update():
    # TODO: how to make this be executed once in a while or at least upon installation?
    """
    Check if the following elements for nltk are installed or up-to-date
    :return: none
    """
    for element in ['words', 'maxent_ne_chunker',
                    'averaged_perceptron_tagger', 'stopwords',
                    'punkt']:
        nltk.download(element)
    return None


def parse(page: str) -> set:
    """
    Returns the most likely company names from a page
    :param page: a cleaned string representation of the page text
    :return: a set of company names
    """
    words = nltk.word_tokenize(page, language='english')
    tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE")


def clean_parsed(model, vocabulary: dict, companies: list, threshold: float = 0.46) -> list:
    """
    Use a machine learning model to filter a list of company names based on a binary classification
    :param model: a PyTorch model
    :param vocabulary: dictionary of vocabulary
    :param companies: list of company names
    :param threshold: the classification threshold
    :return: the filtered list of company names
    """
    companies = base_clean(pd.Series([company.strip().lower() for company in companies]))
    filtered = []
    for idx, company in enumerate(companies):
        pred = model(name_to_tensor(company, vocabulary))
        if pred.item() >= threshold:
            filtered.append(companies[idx])
    return filtered
