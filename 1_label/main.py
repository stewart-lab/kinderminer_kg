import json
import random
import spacy
import sys
import glob
from tokenizers import BertWordPieceTokenizer

from os.path import dirname, join, abspath
parent_path = abspath(join(dirname(__file__), '..'))
parent_path = abspath(join(parent_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

import kinderminer_query as km
from util.abstract_catalog import AbstractCatalog
import util.km_util as km_util

def main():
    genes = './targets/genes.txt'
    drugs = './targets/curatedDrugsOneSynWithHighKMHitsRM2LetterOnesV2.txt'
    phenotypes = './targets/phenotypesBestOneSynBasedOnHighKMHitsRM2letterOnesV2.txt'
    drug_drug = './km_results/drug_interactions.txt'
    out_folder = './km_results/'
    

    # perform the KM query
    km.textfile_km_query(genes, genes, out_folder + 'genes_genes.tsv')
    km.textfile_km_query(genes, phenotypes, out_folder + 'genes_phenotypes.tsv')
    km.textfile_km_query(genes, drugs, out_folder + 'genes_drugs.tsv')

    with open(drug_drug, 'r') as f:
        drug_drug_interactions = [x for x in f.readlines() if '#' not in x]
    query = [{'a_term': x.split('\t')[0].strip(), 'b_term': x.split('\t')[1].strip(), 'return_pmids': True} for x in drug_drug_interactions]
    result = km.run_km_query(query, True)
    km.write_km_result(result, out_folder + 'drugs_drugs.tsv')

    # km.textfile_km_query(drugs, drugs, out_folder + 'drugs_drugs.tsv')
    km.textfile_km_query(drugs, phenotypes, out_folder + 'drugs_phenotypes.tsv')

    # pull PMIDs of significant relationships out
    pmids = _sample_km_pmids(out_folder + 'manually_curated/')

    # pmids = set()
    # jsonl_items = '/Users/rmillikin/My Drive/Postdoc/Projects/kinderminer_kg/label/unlabeled_data/unlabeled_training_data.jsonl'
    # with open(jsonl_items, 'r') as f:
    #     for item in f.readlines():
    #         item = json.loads(item)
    #         pmid = item['meta']['pmid']
    #         pmids.add(pmid)

    # turn PMIDs of significant relationships into formatted abstract text for Prodigy
    _create_unlabeled_training_data('./label/unlabeled_data/unlabeled_training_data.jsonl', pmids, './label/vocab.txt')

def _sample_km_pmids(km_data_path: str):

    return_pmid_list = set()

    data_files = glob.glob(km_data_path + '*.tsv')

    for file in data_files:
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                spl = line.strip().split('\t')

                if i == 0:
                    index_dict = { label : spl.index(label) for label in spl}
                    km_header = line
                else:
                    pvalue = float(spl[index_dict['pvalue']])
                    article_count = int(spl[index_dict['len(a_b_intersect)']])
                    b_term = spl[index_dict['b_term']]

                    if pvalue < 0.01 and article_count > 5:
                        take = 3
                        pmid_intersect = spl[index_dict['pmid_intersection']].strip('{}\"')
                        pmid_intersect = [int(x.strip()) for x in pmid_intersect.split(',')][:take]

                        for pmid in pmid_intersect:
                            return_pmid_list.add(pmid)

    return return_pmid_list

def _create_unlabeled_training_data(out_path: str, pmid_set: set, tokenizer_vocab: str):
    lowercase = True
    hide_special = True
    hide_wp_prefix = True
    tokenizer = BertWordPieceTokenizer(tokenizer_vocab, lowercase=lowercase)
    sep_token = tokenizer._parameters.get("sep_token")
    cls_token = tokenizer._parameters.get("cls_token")
    special_tokens = (sep_token, cls_token)
    wp_prefix = tokenizer._parameters.get("wordpieces_prefix")

    pubmed_path = '/Users/rmillikin/PubmedAbstracts'
    catalog = AbstractCatalog(pubmed_path)

    selected_abstracts = []

    for abs in catalog.stream_existing_catalog(km_util.get_abstract_catalog(pubmed_path)):
        if not abs.text or str.isspace(abs.text):
            continue

        if abs.pmid in pmid_set:
            selected_abstracts.append(abs)

    selected_abstracts = sorted([x for x in selected_abstracts], key=lambda x: random.random())

    with open(out_path, 'w') as f:
        for i, abstract in enumerate(selected_abstracts):
            doc_text = abstract.text
            if '(ABSTRACT TRUNCATED' in doc_text:
                doc_text = doc_text.split('(ABSTRACT TRUNCATED', 1)[0]

            training_item = {}
            training_item["text"] = doc_text
            training_item["tokens"] = _get_tokens(doc_text, tokenizer, hide_special, hide_wp_prefix, special_tokens, wp_prefix)
            training_item["meta"] = {'pmid': abstract.pmid}
            
            f.write(json.dumps(training_item))
            f.write('\n')

def _get_tokens(the_text, tokenizer, hide_special, hide_wp_prefix, special_tokens, wp_prefix):
    tokens = tokenizer.encode(the_text)
    eg_tokens = []
    idx = 0
    for (text, (start, end), tid) in zip(
        tokens.tokens, tokens.offsets, tokens.ids
    ):
        # If we don't want to see special tokens, don't add them
        if hide_special and text in special_tokens:
            continue
        # If we want to strip out word piece prefix, remove it from text
        if hide_wp_prefix and wp_prefix is not None:
            if text.startswith(wp_prefix):
                text = text[len(wp_prefix) :]
        token = {
            "text": text,
            "id": idx,
            "start": start,
            "end": end,
            # This is the encoded ID returned by the tokenizer
            "tokenizer_id": tid,
            # Don't allow selecting special SEP/CLS tokens
            "disabled": text in special_tokens,
        }
        eg_tokens.append(token)
        idx += 1
    for i, token in enumerate(eg_tokens):
        # If the next start offset != the current end offset, we
        # assume there's whitespace in between
        if i < len(eg_tokens) - 1 and token["text"] not in special_tokens:
            next_token = eg_tokens[i + 1]
            token["ws"] = (
                next_token["start"] > token["end"]
                or next_token["text"] in special_tokens
            )
        else:
            token["ws"] = True
    
    return eg_tokens

if __name__ == '__main__':
    main()