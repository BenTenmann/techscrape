import re
import requests
import logging
from .utils import (is_str,
                    pack)

logging.basicConfig(level=logging.INFO)


def process_query(query: str, replace: str) -> str:
    return query.strip().replace(" ", replace)


class Crawler:
    """
    Base crawler class
    """
    BASE_URL: str
    BRANCH_URL: str
    AGG_SEARCH: re.Pattern

    @classmethod
    def _response(cls, query: str) -> str:
        """
        Get the html string response from the query page
        :param query: query string
        :return: html
        """
        return requests.get(f'{cls.BASE_URL}{query}').text

    @classmethod
    def _collect_links(cls, page: str) -> list:
        """
        Collect the branching links from a given query
        :param page: the query page
        :return: list of links
        """
        return cls.AGG_SEARCH.findall(page)

    @classmethod
    def _branching(cls, link_list: list) -> list:
        """
        Returns the query page branches in the form of a list of html strings
        :param link_list: list of branch links
        :return: list of html page responses
        """
        responses = []
        for link in link_list:
            article_url = f'{cls.BRANCH_URL}{link}'
            responses.append(requests.get(article_url).text)
            logging.info(f'{article_url} connected')
        return responses

    @classmethod
    def _base_search(cls, query: str):
        """
        The base search function -- combines query and branching
        :param query: the formatted, page-specific query
        :return: list of html page responses
        """
        page = cls._response(query)
        link_list = cls._collect_links(page)
        branches = cls._branching(link_list)
        return branches


class Sifted(Crawler):
    """
    Crawl sifted.eu for a given query
    """
    BASE_URL = 'https://sifted.eu/?s='
    BRANCH_URL = 'https://sifted.eu/articles/'
    AGG_SEARCH = re.compile(
        r"<a href=\"https://sifted\.eu/articles/([a-z0-9\-]+/)\" "
        r"class=\"hover:text-(?:[a-z\-]+) sifted__analytics__latest-from-sifted\""
        r">(?:[:'\",a-zA-Z0-9$£€? \-]+)</a>"
    )

    @classmethod
    @is_str(type_name='method')
    def search(cls, query):
        """
        Search sifted.eu for a given query
        :param query: space delimited string -- a certain search sentence e.g
                      `machine learning in drug discovery`
        :return: branch page responses
        """
        query = process_query(query, '+')
        return cls._base_search(query)


class TechCrunch(Crawler):
    """
    Crawl TechCrunch for a given query
    """
    BASE_URL = 'https://techcrunch.com/'
    pass


class BioCentury(Crawler):
    """
    Crawl BioCentury for a given query
    """
    BASE_URL = 'https://www.biocentury.com/search?q='
    BRANCH_URL = 'https://www.biocentury.com/article/'

    @classmethod
    @is_str(type_name='method')
    def search(cls, query):
        """
        Search BioCentury for a given query
        :param query: space delimited query string
        :return: list of branch page responses
        """
        query = process_query(query, '%20')
        return cls._base_search(query)


class TheInformation(Crawler):
    """
    Crawl TheInformation for a given query
    """
    BASE_URL = 'https://www.theinformation.com/search?query='
    pass


class BioPharmGuy(Crawler):
    """
    Crawl BioPharmGuy for a given query
    """
    BASE_URL = 'https://biopharmguy.com/'
    BRANCH_URL = '/links/company-by-location'

    @classmethod
    def search(cls, query):
        pass


class MLBlog(Crawler):
    """
    Crawl the blog at the link below
    """
    BASE_URL = (
        'https://blog.benchsci.com/startups-using-artificial-intelligence'
        '-in-drug-discovery#generate_novel_drug_candidates'
    )
    AGG_SEARCH = [
        re.compile(
            r"h3 style=\"clear: both;\">(.+)</h3>"
        ),
        re.compile(
            r"<a target=\"_blank\" href=\"(?:.+)\" rel=\"noopener\">(.+)</a>.*?</h4>"
        ),
        re.compile(
            r"<p><strong>Uses AI to</strong>: (.*?)\..*?"
            r"<strong>Allows researchers to</strong>: (.*?)\..*?"
            r"<strong>Founded</strong>: ([0-9]*?)\..*?"
            r"<strong>Headquarters</strong>: (.*?)\..*?[</p>]*?"
        )]
    KEYS = ['uses_ai_to', 'allows_researchers_to', 'founded', 'hq']

    @classmethod
    def search(cls, *query):
        """
        Scrapes the blog for category, company name, and descriptors
        :param query: *optional
        :return: data dictionary
        """
        page = cls._response('')
        cat_iter = cls.AGG_SEARCH[0].finditer(page)
        categories = list(cat_iter)
        matches = pack(categories)

        desc = cls.AGG_SEARCH[-1].finditer(page)
        data = {
            match_a.group(1):
                {
                    company: {
                        key: value
                        for key, value in
                        [*zip(cls.KEYS, next(desc).groups())]
                    } for company in
                    cls.AGG_SEARCH[1].findall(page[match_a.end(0):match_b.start(0)])
                }
            for match_a, match_b in matches
        }

        return data
