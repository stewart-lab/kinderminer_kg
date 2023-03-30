import json
import random
import typer
from pathlib import Path

from spacy.tokens import Span, SpanGroup, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer
#import rel_pipe
#import rel_model

msg = Printer()

SYMM_LABELS = ["POS_ASSOCIATION", "NEG_ASSOCIATION", "BINDS", "DRUG_INTERACTION_WITH", "COREF"]
MAP_LABELS = {}
# MAP_LABELS = {
#     "Pos-Reg": "Regulates",
#     "Neg-Reg": "Regulates",
#     "Reg": "Regulates",
#     "No-rel": "Regulates",
#     "Binds": "Binds",
# }

def main(json_loc: Path, train_file: Path, dev_file: Path, test_file: Path):
    """Creating the corpus from the Prodigy annotations."""
    Doc.set_extension("rel", default={})
    vocab = Vocab()

    docs = {"train": [], "dev": [], "test": []}
    ids = {"train": set(), "dev": set(), "test": set()}
    count_all = {"train": 0, "dev": 0, "test": 0}
    count_pos = {"train": 0, "dev": 0, "test": 0}

    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            # span_starts = set()
            if example["answer"] == "accept":
                neg = 0
                pos = 0
                try:
                    # Parse the tokens
                    words = [t["text"] for t in example["tokens"]]
                    spaces = [t["ws"] for t in example["tokens"]]
                    doc = Doc(vocab, words=words, spaces=spaces)

                    # Parse the entities
                    prodigy_spans = example["spans"]
                    spacy_spans = []
                    for span in prodigy_spans:
                        entity = _convert_prodigy_span_to_spacy_span(doc, span)
                        spacy_spans.append(entity)

                    doc.spans['spans'] = spacy_spans

                    # Parse the relations
                    rels = {}
                    for x1 in prodigy_spans:
                        for x2 in prodigy_spans:
                            rels[get_combo_tuple(x1, x2)] = {}
                    relations = example["relations"]
                    for relation in relations:
                        head_span = relation["head_span"]
                        tail_span = relation["child_span"]
                        rel_label = relation["label"]
                        combo_tuple = get_combo_tuple(head_span, tail_span)
                        
                        if rel_label not in MAP_LABELS:
                            MAP_LABELS[rel_label] = rel_label

                        if rel_label not in rels[combo_tuple]:
                            rels[combo_tuple][rel_label] = 1.0
                            pos += 1
                        if rel_label in SYMM_LABELS:
                            symm_tuple = get_combo_tuple(tail_span, head_span)

                            if rel_label not in rels[symm_tuple]:
                                rels[symm_tuple][rel_label] = 1.0
                                pos += 1

                    # The annotation is complete, so fill in zero's where the data is missing
                    for x1 in prodigy_spans:
                        for x2 in prodigy_spans:
                            combo_tuple = get_combo_tuple(x1, x2)

                            for rel_label in MAP_LABELS.values():
                                if rel_label not in rels[combo_tuple]:
                                    neg += 1
                                    rels[combo_tuple][rel_label] = 0.0
                    doc._.rel = rels

                    # only keeping documents with at least 1 positive case
                    if pos > 0:
                        # use the original PMID/PMCID to decide on train/dev/test split
                        article_id = str(example["meta"]["pmid"])

                        if article_id.endswith("1"):
                            ids["dev"].add(article_id)
                            docs["dev"].append(doc)
                            count_pos["dev"] += pos
                            count_all["dev"] += pos + neg
                        # elif article_id.endswith("3"):
                        #     ids["test"].add(article_id)
                        #     docs["test"].append(doc)
                        #     count_pos["test"] += pos
                        #     count_all["test"] += pos + neg
                        else:
                            ids["train"].add(article_id)
                            docs["train"].append(doc)
                            count_pos["train"] += pos
                            count_all["train"] += pos + neg
                except KeyError as e:
                    msg.fail(f"Skipping doc because of key error: {e} in {example['meta']['pmid']}")
                    #print(str(example))

    docbin = DocBin(docs=docs["train"], store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(
        f"{len(docs['train'])} training sentences from {len(ids['train'])} articles, "
        f"{count_pos['train']}/{count_all['train']} pos instances."
    )

    docbin = DocBin(docs=docs["dev"], store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(
        f"{len(docs['dev'])} dev sentences from {len(ids['dev'])} articles, "
        f"{count_pos['dev']}/{count_all['dev']} pos instances."
    )

    docbin = DocBin(docs=docs["test"], store_user_data=True)
    docbin.to_disk(test_file)
    msg.info(
        f"{len(docs['test'])} test sentences from {len(ids['test'])} articles, "
        f"{count_pos['test']}/{count_all['test']} pos instances."
    )


def _convert_prodigy_span_to_spacy_span(doc, prodigy_span):
    spacy_span = doc[prodigy_span['token_start'] : prodigy_span['token_end'] + 1]
    spacy_span.label_ = prodigy_span['label']
    return spacy_span

def get_combo_tuple(head_span, tail_span):
    combo_tuple = (head_span["token_start"], head_span["token_end"] + 1, head_span["label"], tail_span["token_start"], tail_span["token_end"] + 1, tail_span["label"])
    return combo_tuple

if __name__ == "__main__":
    typer.run(main)
