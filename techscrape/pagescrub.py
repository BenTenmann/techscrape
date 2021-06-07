import re
from .utils import is_str


class PageScrubber:
    """
    PageScrubber object
    """
    PARAGRAPH = re.compile(  # works for sifted.eu for now
        r"<span style=\"font-weight: 400;\">(.+?)</span>"
    )

    @classmethod
    @is_str(type_name='method')
    def scrub(cls, page: str) -> str:
        """
        Cleans the page response to get the text body
        :param page: page html-response
        :return: Text body
        """
        return ' '.join(cls.PARAGRAPH.findall(page))
