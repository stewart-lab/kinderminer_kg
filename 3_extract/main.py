#!/bin python3
import os
os.environ["CUPY_CACHE_DIR"] = os.getcwd()

import sys
import spacy
import json
import argparse
import re
import thinc
import torch
import gzip
from spacy.tokens import DocBin, Doc, SpanGroup
from spacy.language import Language
from rel_pipe import make_relation_extractor, score_relations
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors

sentencizer_regex = r"(.*?)(?:(?<!\bvs|\bSt|\b\.g)\.\s{1,}|\?\s{1,}|$)(?=[a-z]{0,}[A-Z]|$)"

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--json_in')
parser.add_argument('-o', '--tsv_out')
args = parser.parse_args()

def extract_relations_from_json(in_path: str, out_path: str, spans_model: str, rel_model: str):
    # load spancat model
    spancat_model = spacy.load(spans_model)

    # load rel model
    rel_model = spacy.load(rel_model)

    with gzip.open(in_path, 'rt', encoding='utf-8') as file:
        lines = file.readlines()

    with gzip.open(out_path, 'wt', encoding='utf-8') as out_file:
        out_file.write('name\ttag\trel\tname\ttag\tpmid')
        out_file.write('\n')

        out_item = []
        for i, line in enumerate(lines):
            pmid = 0

            try:
                out_item.clear()
                parsed_json = json.loads(line)
                abstract_title = parsed_json['title']
                abstract_text = parsed_json['text'].strip()
                pmid = parsed_json['pmid']

                if not abstract_text:
                    continue

                if '(ABSTRACT TRUNCATED' in abstract_text:
                    abstract_text = abstract_text.split('(ABSTRACT TRUNCATED', 1)[0].strip()

                # tokenize, get spans
                paragraph_doc = spancat_model(abstract_text)
                # paragraph_doc = merge_subspans(paragraph_doc)
                paragraph_tokens = [x.text for x in paragraph_doc[0:len(paragraph_doc)]]
                
                # sentencize, extract relations
                matches = re.finditer(sentencizer_regex, abstract_text, re.MULTILINE)
                for matchNum, match in enumerate(matches, start=1):
                    sentence_text = match.group().strip()

                    if not sentence_text:
                        continue

                    sent_doc = spancat_model(sentence_text) # TODO: only user tokenizer here, not spancat
                    
                    # match up spans from paragraph to sentences
                    sent_tokens = [x.text for x in sent_doc[0:len(sent_doc)]]
                    test = subfinder(paragraph_tokens, sent_tokens)
                    tok_offset = test[0]

                    sent_spans = SpanGroup(sent_doc)
                    for paragraph_span in paragraph_doc.spans['sc']:
                        sent_span_start = paragraph_span.start - tok_offset
                        sent_span_end = paragraph_span.end - tok_offset

                        if sent_span_start < 0 or sent_span_start > len(sent_doc):
                            continue
                        if sent_span_end < 0 or sent_span_end > len(sent_doc):
                            continue

                        sent_span = sent_doc[sent_span_start : sent_span_end]
                        sent_span.label_ = paragraph_span.label_
                        sent_spans.append(sent_span)
                    sent_doc.spans['sc'] = sent_spans
                    sent_doc.spans['spans'] = sent_spans

                    # extract relations
                    for name, proc in rel_model.pipeline:
                        sent_doc = proc(sent_doc)

                    for rel_spans, rel_dict in sent_doc._.rel.items():
                        spans = sent_doc.spans['sc']

                        for rel_label, rel_score in rel_dict.items():
                            if rel_score > 0.5:
                                for span_a in spans:
                                    for span_b in spans:
                                        if span_a.start == rel_spans[0] and span_a.end == rel_spans[1] and span_a.label_ == rel_spans[2] and span_b.start == rel_spans[3] and span_b.end == rel_spans[4] and span_b.label_ == rel_spans[5]:
                                            out_item.append(str(span_a.text).strip())
                                            out_item.append(str(span_a.label_))
                                            out_item.append(str(rel_label))
                                            out_item.append(str(span_b.text).strip())
                                            out_item.append(str(span_b.label_))
                                            out_item.append(str(pmid))

                                            # write rel data to tsv
                                            out_file.write(str.join('\t', out_item))
                                            out_file.write('\n')
                                            out_item.clear()
                                            
            except Exception as e:
                print('error extracting pmid: ' + str(pmid))
                print(str(e), file=sys.stderr)

    # delete *.cubin files
    dir_name = os.getcwd()
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".cubin"):
            os.remove(os.path.join(dir_name, item))

# https://stackoverflow.com/questions/10106901/elegant-find-sub-list-in-list
def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append(i)
    return matches

# def merge_subspans(span_doc):
#     spans = span_doc.spans['sc']
#     new_spans = SpanGroup()

#     for span in spans:

#         pass

#     span_doc.spans['sc'] = new_spans
#     span_doc.spans['spans'] = new_spans

def main():
    gpu_name = os.environ["CUDA_VISIBLE_DEVICES"]
    gpu_number = torch.cuda.current_device()
    spacy.require_gpu(gpu_id=gpu_number)
    print('using gpu: ' + str(gpu_number) + ' (' + gpu_name + ')')

    thinc.api.use_pytorch_for_gpu_memory()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    extract_relations_from_json(args.json_in, args.tsv_out, '/models/spans/model-best', '/models/rel/model-best')

    # DEBUG
    # extract_relations_from_json(
    #     '/Users/rmillikin/Desktop/abstract_chunks/chunk_0.jsonl.gzip', 
    #     '/Users/rmillikin/Desktop/abstract_chunks/chunk_0_rels_debug.tsv.gzip', 
    #     './extract/models/spans/model-best', 
    #     './extract/models/rel/model-best')

if __name__ == '__main__':
    main()