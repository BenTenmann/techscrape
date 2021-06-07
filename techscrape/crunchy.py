import logging
import time
from .utils.helpers import *
from scipy.stats import expon

logging.basicConfig(level=logging.INFO)


class Scraper:
    # TODO: Fix the scraping -- returns too many empties; improve logging.
    ENDPOINT = 'https://www.crunchbase.com/organization/'
    FINANCIALS = re.compile(  # get the total funding for a company
        r"<span class=\"component--field-formatter field-type-money ng-star-inserted\""
        r" title=\"\$([0-9.MB]+)\">\$(?:[0-9.MB]+)</span>"
    )
    N_EMPLOYEES = re.compile(  # gets number of employees for a company
        r"<a class=\"component--field-formatter field-type-enum link-accent ng-star-inserted\""
        r" href=\"/search/people/field/organizations/num_employees_enum/(?:[a-z0-9_\-]+)\">([0-9\-]+)</a>"
    )
    SERIES = re.compile(  # get the last funding type
        r"<a class=\"component--field-formatter field-type-enum link-accent ng-star-inserted\""
        r" href=\"/search/funding_rounds/field/organizations/last_funding_type/(?:[a-z0-9_\-]+)\">([A-Za-z\s]+)</a>"
    )
    LOCATION = re.compile(  # get the location of the company
        r"<a _ngcontent-sc240=\"\" title=\"(?:[A-Za-z\s]+)\" class=\"link-accent ng-star-inserted\""
        r" href=\"/search/organizations/field/organizations/location_identifiers/(?:[a-z0-9_\-]+)\"> "
        r"([A-Za-z\s]+)</a>"
    )
    WEBSITE = re.compile(
        r"href=\"(?:[htps:/]*www\.[a-z\-]+\.[a-z]+/?)\" target=\"_blank\" "
        r"title=\"(?:[htps:/]*www\.[a-z\-]+\.[a-z]+/?)\" "
        r"aria-label=\"([htps:/]*www\.[a-z\-]+\.[a-z]+/?)\"> "
    )
    RECOMBINATION = [  # regex name change + sub string
        (
            re.compile(  # e.g. benevolentai -> benevolent-ai
                r"^([a-z0-9\-]+)(?:ai)$"
            ),
            r"\1-ai"
        ),
        (
            re.compile(  # e.g. antidote.me -> antidote-me
                r"^([a-z0-9\-]+)(?:\.)([a-z0-9\-]+)$"
            ),
            r"\1-\2"
        )
    ]

    def __init__(self,  suppress_warning=False, **client):
        if all([key in client.keys() for key in ['user',
                                                 'password',
                                                 'host']]):
            self.proxies = {  # VPN
                'http': f'{client["user"]}:{client["password"]}@{client["host"]}',
                'https': f'{client["user"]}:{client["password"]}@{client["host"]}'
            }
        else:
            if not suppress_warning:
                print(
                    f'{TerminalColors.WARNING}[WARNING]: no proxy currently activated{TerminalColors.ENDC}'
                )
        self.logger = {  # log the connection attempts per company
            'attempts': 0,
            'att_names': [],
            'responses': []
        }

    def _get_total_funding(self, page: str) -> list:
        """
        Get the total funding to date of the company
        :param page: the html-format page response
        :return: list of financials
        """
        return self.FINANCIALS.findall(page)

    def _get_employees(self, page: str) -> list:
        """
        Get the number of employees of the company
        :param page: the html-string
        :return: list of number of employees
        """
        return self.N_EMPLOYEES.findall(page)

    def _get_series(self, page: str) -> list:
        """
        Get the status of the last funding received
        :param page: html-string
        :return: list of series
        """
        return self.SERIES.findall(page)[:1]

    def _get_loc(self, page: str) -> list:
        """
        Get the location of the company
        :param page: the html-string
        :return: list of locations
        """
        return self.LOCATION.findall(page)

    def _get_website(self, page: str) -> list:
        """
        Get the website link of the company
        :param page: html-string
        :return: list of link
        """
        return self.WEBSITE.findall(page)

    def _connect(self, name: str) -> str:
        """
        Establish a connection to the company of interest on CrunchBase
        :param name: the company name
        :return: the page response as a html string
        """
        try:
            self.logger["attempts"] += 1
            self.logger["att_names"].append(name)

            response = requests.get(
                f'{self.ENDPOINT}{name}',
                headers={
                    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
                    'referer': 'https://www.crunchbase.com/',
                    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,'
                              'image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                    'accept-encoding': 'gzip, deflate, br',
                    'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
                    'content-type': 'text/html; charset=utf-8'
                },
                # proxies=self.proxies
            )
        except requests.ConnectionError:
            raise RuntimeError(
                f'{TerminalColors.WARNING}There was an error with the web-request{TerminalColors.ENDC}'
            )

        self.logger["responses"].append(response.status_code)
        if response.status_code != 200:
            if response.status_code == 403:
                raise BotBlockError(
                    f'{TerminalColors.FAIL}[403] FORBIDDEN{TerminalColors.ENDC}'
                )
            if response.status_code == 404:
                print(
                    f'{TerminalColors.WARNING}||---[WARNING: 404]--||\n'
                    f'{self.ENDPOINT}{name} not found{TerminalColors.ENDC}'
                )
                if self.logger["attempts"] <= self.RECOMBINATION.__len__():
                    regex, sub = self.RECOMBINATION[
                        self.logger["attempts"] - 1
                        ]
                    name = regex.sub(sub, name)
                    page = self._connect(name)
                else:
                    raise TimeoutError(
                        f'{TerminalColors.FAIL}||---TIMEOUT---||\n'
                        f'{print_log(**self.logger)}{TerminalColors.ENDC}'
                    )
            else:
                raise RuntimeError(
                    '||---FATAL---||\n'
                    'Something, somewhere, went badly wrong\n'
                    f'Response code: {response.status_code}\n'
                    'LOG: \n'
                    f'{print_log(**self.logger)}'
                )
        else:
            page = response.text
        return page

    def get_all(self, name: str) -> dict:
        """
        Get all data from a given company
        :param name: The name of a company
        :return: a dictionary of data
        """
        name = name.lower().strip().replace(' ', '-')

        self.logger["attempts"] = 0
        try:
            page = self._connect(name)
        except TimeoutError:
            return {
                'total_funding': [],
                'n_employees': [],
                'series': [],
                'location': [],
                'website': []
            }
        # get the relevant data from the page
        fin = self._get_total_funding(page)
        emp = self._get_employees(page)
        ser = self._get_series(page)
        loc = self._get_loc(page)
        web = self._get_website(page)

        return {
            'total_funding': fin,
            'n_employees': emp,
            'series': ser,
            'location': loc,
            'website': web
        }


class CrunchyPy(Scraper):
    def get(self, company_dict: dict, to_file: bool = True) -> dict:
        """
        Get the company data over a dictionary of categories
        :param company_dict: dictionary of categories, each associated with a
                             list of companies
        :param to_file: whether to save to a json file
        :return: a dictionary of company data
        """
        companies = []
        metadata = {}
        for category in company_dict.values():
            companies += category
        for company in companies:
            metadata[company] = self.get_all(company)
            if to_file:
                with open('../depr/test_2.json', 'w') as file:
                    file.write(
                        json.dumps(metadata, indent=4)
                    )
                file.close()
            t = expon.rvs(scale=20)
            logging.info(f'{company} processed -- waiting {t}s')
            time.sleep(t)
        return metadata

    def manual_download(self, data: dict, companies: list) -> dict:
        """
        Manual download gives greater outside control of data collection from CrunchBase
        as compared to .get()
        :param data: a data dictionary
        :param companies: a list of company names
        :return: the updated data dictionary
        """
        for company in companies:
            data[company] = self.get_all(company)
        return data


def to_dataframe(data: dict) -> pd.DataFrame:
    """
    Convert the dictionary format of data returned by CrunchyPy into a data_frame
    format
    :param data: dictionary of data
    :return: a dataframe version of said data
    """
    loc, emp, ser, fun, web = [], [], [], [], []
    companies = []
    for company, attributes in data.items():
        loc += [';'.join(empty(attributes["location"]))]
        emp += empty(attributes["n_employees"])[:1]
        ser += empty(attributes["series"])[:1]
        fun += empty(attributes["total_funding"])[:1]
        web += empty(attributes["website"])[:1]
        companies.append(company)
    dframe = pd.DataFrame({
        'company': companies,
        'location': loc,
        'n_employees': emp,
        'last_funding': ser,
        'total_funding': fun,
        'website': web
    })
    return dframe


class BotBlockError(Exception):
    """
    Custom exception for 403 page response
    """
    pass
