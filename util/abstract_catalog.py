import pickle
import gzip
import os
import glob
import xml.etree.ElementTree as ET
import util.km_util as km_util
from util.abstract import Abstract

delim = '\t'

class AbstractCatalog():
    def __init__(self, pubmed_path) -> None:
        self.catalog = dict()
        self.abstract_files = list()
        self.path_to_pubmed_abstracts = pubmed_path

    def catalog_abstracts(self, dump_rate = 150):
        abstract_files_to_catalog = self._get_files_to_catalog()
        abstract_files_to_catalog = sorted(abstract_files_to_catalog)

        if not abstract_files_to_catalog:
            return

        path = km_util.get_abstract_catalog(self.path_to_pubmed_abstracts)
        self.load_existing_catalog(path)

        km_util.report_progress(0, len(abstract_files_to_catalog))

        for i, gzip_file in enumerate(abstract_files_to_catalog):
            with gzip.open(gzip_file, 'rb') as xml_file:
                abstracts = _parse_xml(xml_file.read())
                filename = os.path.basename(gzip_file)

                for abstract in abstracts:
                    self.add_or_update_abstract(abstract)

                self.abstract_files.append(filename)
                km_util.report_progress(i + 1, len(abstract_files_to_catalog))

                if i % dump_rate == 0:
                    self.write_catalog_to_disk(path)
                
        self.write_catalog_to_disk(path)

    def add_or_update_abstract(self, abstract: Abstract) -> None:
        if abstract.pmid in self.catalog:
            # if PMID is already in catalog, update it w/ new info

            # get the old abstract
            old = pickle.loads(self.catalog[abstract.pmid])

            # use the earlier of the two years, in case the new one is a correction
            year = min(abstract.pub_year, old.pub_year)

            # use the new text, unless it is unavailable
            if str.isspace(abstract.text):
                text = old.text
            else:
                text = abstract.text
            
            # use the new title, unless it is unavailable
            if str.isspace(abstract.title):
                title = old.title
            else:
                title = abstract.title

            # create the merged abstract object
            abstract = Abstract(abstract.pmid, year, title, text)

        self.catalog[abstract.pmid] = pickle.dumps(abstract)

    def write_catalog_to_disk(self, path: str) -> None:
        dir = os.path.dirname(path)
        
        if not os.path.exists(dir):
            os.mkdir(dir)

        with gzip.open(path, 'wt', encoding=km_util.encoding) as gzip_file:
            for pmid in self.catalog:
                abs = self.catalog[pmid]
                abs = pickle.loads(abs)
                line = str(abs) + '\n'
                gzip_file.write(line)

        km_util.write_all_lines(km_util.get_cataloged_files(self.path_to_pubmed_abstracts), self.abstract_files)

    def load_existing_catalog(self, path: str) -> None:
        '''Used to update an existing catalog'''
        if not os.path.exists(path):
            return

        with gzip.open(path, 'rt', encoding=km_util.encoding) as file:
            for line in file:
                abs = self._parse_abstract(line)
                self.catalog[abs.pmid] = pickle.dumps(abs)

    def stream_existing_catalog(self, path: str) -> 'list[Abstract]':
        '''Used to index the abstracts' tokens in the completed catalog'''
        with gzip.open(path, 'rt', encoding=km_util.encoding) as file:
            for line in file:
                yield self._parse_abstract(line)

    def _parse_abstract(self, line: str):
        split = line.strip('\n').split('\t')
        abs = Abstract(int(split[0]), int(split[1]), split[2], split[3])
        return abs

    def _get_files_to_catalog(self) -> 'list[str]':
        cataloged_files_path = km_util.get_cataloged_files(self.path_to_pubmed_abstracts)
        if os.path.exists(cataloged_files_path):
            self.abstract_files = km_util.read_all_lines(cataloged_files_path)

        all_abstracts_files = glob.glob(os.path.join(self.path_to_pubmed_abstracts, "*.xml.gz"))
        not_indexed_yet = []

        for file in all_abstracts_files:
            if os.path.basename(file) not in self.abstract_files:
                not_indexed_yet.append(file)

        return not_indexed_yet

def _parse_xml(xml_content: str) -> 'list[Abstract]':
    """"""
    root = ET.fromstring(xml_content)
    abstracts = []

    for pubmed_article in root.findall('PubmedArticle'):
        pmid = None
        year = None
        title = None
        abstract_text = None

        # get PMID
        try:
            medline_citation = pubmed_article.find('MedlineCitation')
            pmid = medline_citation.find('PMID').text
        except AttributeError:
            continue

        # get publication year
        try:
            article = medline_citation.find('Article')
            journal = article.find('Journal')
            journal_issue = journal.find('JournalIssue')
            pub_date = journal_issue.find('PubDate')
            year = pub_date.find('Year').text
        except AttributeError:
            year = 99999 # TODO: kind of hacky...
            pass

        # get article title
        try:
            title = article.find('ArticleTitle').text
        except AttributeError:
            pass

        # get article abstract text
        try:
            abstract = article.find('Abstract')
            abs_text_nodes = abstract.findall('AbstractText')
            if len(abs_text_nodes) == 1:
                abstract_text = "".join(abs_text_nodes[0].itertext())
            else:
                node_texts = []

                for node in abs_text_nodes:
                    node_texts.append("".join(node.itertext()))
                
                abstract_text = " ".join(node_texts)
        except AttributeError:
            pass

        if pmid:
            if type(year) is str:
                year = int(year)

            abstract = Abstract(int(pmid), year, title, abstract_text)
            abstracts.append(abstract)

            if str.isspace(abstract.text) and str.isspace(abstract.title):
                continue

    return abstracts