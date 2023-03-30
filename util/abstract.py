disallowed = ['\n', '\t', '\r']

class Abstract():
    def __init__(self, pmid: int, year: int, title: str, text: str):
        self.pmid = pmid
        self.pub_year = year
        self.title = title
        self.text = text

        if not self.title or str.isspace(self.title):
            self.title = ' '
        if not self.text or str.isspace(self.text):
            self.text = ' '

    def __str__(self) -> str:
        str_pmid = str(self.pmid)
        str_year = str(self.pub_year)
        str_title = self.title
        str_text = self.text

        for item in disallowed:
            if item in str_title:
                str_title = str_title.replace(item, '')
            if item in str_text:
                str_text = str_text.replace(item, '')

        return str_pmid + '\t' + str_year + '\t' + str_title + '\t' + str_text