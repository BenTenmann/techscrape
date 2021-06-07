import os
import time
import json
import torch
import logging
from scipy.stats import expon
from .crawlers import Crawler
from .pagescrub import PageScrubber
from .pageparser import parse, clean_parsed
from .crunchy import CrunchyPy, BotBlockError
from .models import LSTMClassifier
from .utils.helpers import WaitingTime

logging.basicConfig(level=logging.INFO)


class TechScraper:
    """
    The TechScraper class
    """

    def __init__(self, crawlers=None):
        if crawlers is None:
            raise ValueError('`crawlers` must be given')
        if not (
                type(crawlers) == list
                and
                all(crawler.__bases__[0] == Crawler for crawler in crawlers)
        ):
            raise TypeError('`crawlers` must be a list of Crawler objects')
        self.crawlers = crawlers
        with open(f'{os.path.dirname(__file__)}/models/vocab.json', 'r') as file:
            f_ = file.read()
        file.close()
        self.vocab = json.loads(f_)
        self.model = LSTMClassifier(embedding_dim=80,
                                    hidden_dim=200,
                                    vocab_size=len(self.vocab),
                                    label_size=1,
                                    learning_rate=0.001,
                                    batch_size=1)
        self.model.load_state_dict(torch.load(f'{os.path.dirname(__file__)}/models/model.pt'))

    def _get_pages(self, query):
        pages = []
        for crawler in self.crawlers:
            pages += crawler.search(query)
        return pages

    def get(self, query: str) -> list:
        """
        Get a list of company names from a given query using the list of crawlers
        provided at initialization
        :param query: a space delimited query
        :return: a list of company names
        """
        pages = self._get_pages(query)
        companies = set()
        for page in pages:
            cleaned = PageScrubber.scrub(page)
            companies |= parse(cleaned)
        companies = list(companies)
        return clean_parsed(self.model, self.vocab, companies)

    @staticmethod
    def extend_data(data: dict, companies: list, step_size: int = 1,
                    wait: WaitingTime = WaitingTime(60, expon(scale=20))) -> dict:
        """
        Extend a data dictionary with a list of company names. It will only extend the data
        based on the company names not in the data.
        This function scrapes CrunchBase for these company names (slowly). The `aggressiveness`
        of the scraping (i.e. the step-size and the waiting times) can be set, however; it is
        recommended to use the default settings.
        :param data: a dictionary of data; can be empty
        :param companies: a list of company names to be searched on CrunchBase
        :param step_size: the number of companies to be searched in any one step -- i.e. how many
                          company names should be looked up before the scraper starts waiting again
        :param wait: a WaitTime object, which takes a constant value + a distribution to set the
                     the amount of time the scraper should wait between steps; necessary to avoid
                     a `BotBlockError`, i.e. ot to get flagged as a web-crawler and blocked from the
                     CrunchBase
        :return: the updated data dictionary
        """
        ls = [company for company in companies if company not in data.keys()]
        total = len(ls)
        data_to_add = (company for company in ls)
        cb = CrunchyPy(suppress_warning=True)
        while True:
            try:
                data = cb.manual_download(data, [next(data_to_add) for _ in range(step_size)])
            except (StopIteration, BotBlockError) as e:
                if e == BotBlockError:
                    print(e)
                break
            t = wait()
            total -= step_size
            logging.info(
                f' {step_size} companies loaded -- {total} left; waiting for {t:.2f}s'
            )
            time.sleep(t)
        return data
