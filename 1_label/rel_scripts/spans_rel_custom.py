import copy
from typing import List, Optional
import spacy
from spacy.training import Example
import prodigy
from typing import List, Optional, Union, Iterable, Callable
import spacy
from spacy.tokens import Doc, Span, SpanGroup
from spacy.training import Example
from spacy.language import Language
import copy
from prodigy.components.loaders import JSONL, get_stream
from prodigy.components.preprocess import add_tokens, split_sentences, make_raw_doc
from prodigy.models.matcher import PatternMatcher
from prodigy.core import recipe
from prodigy.util import log, split_string, get_labels, msg, set_hashes, INPUT_HASH_ATTR
from prodigy.types import TaskType, RecipeSettingsType, StreamType
import json
from rel_pipe import make_relation_extractor, score_relations
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


@recipe(
    "spans_rel_custom",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy pipeline with a span categorizer", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated relation label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    span_label=("Comma-separated span label(s) to annotate or text file with one label per line", "option", "sl", get_labels),
    wrap=("Wrap lines in the UI by default (instead of showing tokens in one row)", "flag", "W", bool),
    component=("Name of spancat component in the pipeline", "option", "c", str),
    # fmt: on
)
def manual(
    dataset: str,
    spacy_model: str,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    span_label: Optional[List[str]] = None,
    component: str = "spancat",
    wrap: bool = False,
    hide_arrow_heads: bool = False,
) -> RecipeSettingsType:
    """
    Manually label relations between spans predicted by a spancat model.
    """
    log("RECIPE: Starting recipe spans_rel_custom", locals())
    nlp = spacy.load(spacy_model)

    if component not in nlp.pipe_names:
        msg.fail(
            f"Can't find component '{component}' in pipeline. Make sure that the "
            f"pipeline you're using includes a trained span categorizer that you "
            f"can correct. If your component has a different name, you can use "
            f"the --component option to specify it.",
            exits=1,
        )
    labels = span_label
    model_labels = nlp.pipe_labels.get(component, [])
    if not labels:
        labels = model_labels
        if not labels:
            msg.fail("No --label argument set and no labels found in model", exits=1)
        msg.text(f"Using {len(labels)} labels from model: {', '.join(labels)}")
    log(f"RECIPE: Annotating with {len(labels)} labels", labels)
    key = nlp.get_pipe(component).key
    msg.text(f"""Reading spans from key '{key}': doc.spans["{key}"]""")
    stream = get_stream(source, loader=loader, rehash=True, dedup=True, input_key="text")

    def make_tasks(nlp: Language, stream: StreamType) -> StreamType:
        """Add a 'spans' key to each example, with predicted spans."""
        texts = ((eg["text"], eg) for eg in stream)

        for text, eg in texts:
            sentence_docs = make_rel_tasks_from_abstract_doc(eg, nlp)

            for task in sentence_docs:
                yield task

    stream = make_tasks(nlp, stream)
    stream = add_tokens(nlp, stream)

    return {
        "view_id": "relations",
        "dataset": dataset,
        "stream": stream,
        "update": None,
        "exclude": None,
        "config": {
            "lang": nlp.lang,
            "labels": label,
            "relations_span_labels": span_label,
            "exclude_by": "input",
            "wrap_relations": wrap,
            "custom_theme": {"cardMaxWidth": "90%"},
            "hide_relation_arrow": hide_arrow_heads,
            "auto_count_stream": True,
        },
    }

@recipe(
    "spans_rel_custom.correct",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy pipeline with a span categorizer", "positional", None, str),
    spacy_rel_model=("Loadable spaCy pipeline with a relation extractor", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated relation label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    span_label=("Comma-separated span label(s) to annotate or text file with one label per line", "option", "sl", get_labels),
    wrap=("Wrap lines in the UI by default (instead of showing tokens in one row)", "flag", "W", bool),
    component=("Name of spancat component in the pipeline", "option", "c", str),
    rel_component=("Name of relation extractor component in the pipeline", "option", "rc", str),
    min_rel_score=("Minimum rel score needed from model", "option", "minsc", float),
    max_rel_score=("Minimum rel score needed from model", "option", "maxsc", float),
    # fmt: on
)
def correct(
    dataset: str,
    spacy_model: str,
    spacy_rel_model: str,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    span_label: Optional[List[str]] = None,
    component: str = "spancat",
    rel_component: str = "relation_extractor",
    min_rel_score: float = 0,
    max_rel_score: float = 1,
    wrap: bool = False,
    hide_arrow_heads: bool = False,
) -> RecipeSettingsType:
    """
    Correct relations between spans predicted by a rel/spancat models.
    """
    log("RECIPE: Starting recipe spans_rel_custom.correct", locals())
    span_model = spacy.load(spacy_model)
    rel_model = spacy.load(spacy_rel_model)

    if component not in span_model.pipe_names:
        msg.fail(
            f"Can't find component '{component}' in pipeline. Make sure that the "
            f"pipeline you're using includes a trained span categorizer that you "
            f"can correct. If your component has a different name, you can use "
            f"the --component option to specify it.",
            exits=1,
        )

    if rel_component not in rel_model.pipe_names:
        msg.fail(
            f"Can't find component '{rel_component}' in pipeline. Make sure that the "
            f"pipeline you're using includes a trained relation extractor that you "
            f"can correct. If your component has a different name, you can use "
            f"the --rel_component option to specify it.",
            exits=1,
        )

    labels = span_label
    model_labels = span_model.pipe_labels.get(component, [])
    if not labels:
        labels = model_labels
        if not labels:
            msg.fail("No --label argument set and no labels found in model", exits=1)
        msg.text(f"Using {len(labels)} labels from model: {', '.join(labels)}")
    log(f"RECIPE: Annotating with {len(labels)} labels", labels)

    key = span_model.get_pipe(component).key
    msg.text(f"""Reading spans from key '{key}': doc.spans["{key}"]""")
    # rel_key = rel_model.get_pipe(rel_component).key
    # msg.text(f"""Reading rels from key '{rel_key}': doc.rel["{rel_key}"]""")

    # add the rel component to the pipe
    #nlp.add_pipe("relation_extraction", source=nlp_rel)

    stream = get_stream(source, loader=loader, rehash=True, dedup=True, input_key="text")

    def make_tasks(span_model: Language, rel_model: Language, stream: StreamType) -> StreamType:
        """Add a 'spans' key to each example, with predicted spans."""
        texts = ((eg["text"], eg) for eg in stream)

        for text, eg in texts:
            try:
                sentence_docs = make_rel_tasks_from_abstract_doc(eg, span_model, rel_model, min_rel_score)

                for task in [x for x in sentence_docs if len(x['spans']) > 1 and len(x['relations']) == 0]:
                    #print(str(task))
                    #task['meta'] = {}
                    yield task
            except:
                print('error in PMID' + str(eg['meta']['pmid']))

    stream = make_tasks(span_model, rel_model, stream)
    stream = add_tokens(span_model, stream)

    return {
        "view_id": "relations",
        "dataset": dataset,
        "stream": stream,
        "update": None,
        "exclude": None,
        "config": {
            "lang": span_model.lang,
            "labels": label,
            "relations_span_labels": span_label,
            "exclude_by": "input",
            "wrap_relations": wrap,
            "custom_theme": {"cardMaxWidth": "90%"},
            "hide_relation_arrow": hide_arrow_heads,
            "auto_count_stream": True,
        },
    }

@recipe(
    "spans_rel_custom.review",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy pipeline with a span categorizer", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated relation label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    span_label=("Comma-separated span label(s) to annotate or text file with one label per line", "option", "sl", get_labels),
    wrap=("Wrap lines in the UI by default (instead of showing tokens in one row)", "flag", "W", bool),
    component=("Name of spancat component in the pipeline", "option", "c", str),
    # fmt: on
)
def review(
    dataset: str,
    spacy_model: str,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    span_label: Optional[List[str]] = None,
    component: str = "spancat",
    # rel_component: str = "relation_extractor",
    wrap: bool = False,
    hide_arrow_heads: bool = False,
) -> RecipeSettingsType:
    """
    Review relations in already-annotated data.
    """
    log("RECIPE: Starting recipe spans_rel_custom.review", locals())
    span_model = spacy.load(spacy_model)

    labels = span_label
    log(f"RECIPE: Annotating with {len(labels)} labels", labels)

    stream = get_stream(source, loader=loader, rehash=True, dedup=True, input_key="text")

    stream = add_tokens(span_model, stream)

    return {
        "view_id": "relations",
        "dataset": dataset,
        "stream": stream,
        "update": None,
        "exclude": None,
        "config": {
            "lang": span_model.lang,
            "labels": label,
            "relations_span_labels": span_label,
            "exclude_by": "input",
            "wrap_relations": wrap,
            "custom_theme": {"cardMaxWidth": "90%"},
            "hide_relation_arrow": hide_arrow_heads,
            "auto_count_stream": True,
        },
    }

# takes unlabeled data and annotates spans, then breaks into sentences
def make_rel_tasks_from_abstract_doc(paragraph_json: dict, spancat_model: Language, rel_model: Language = None, min_rel_score = 0):
    # tokenize, get spans
    paragraph_doc = spancat_model(paragraph_json['text'])
    paragraph_tokens = [x.text for x in paragraph_doc[0:len(paragraph_doc)]]
    
    # break doc text into sentences
    import re
    regex = r"(.*?)(?:(?<!\bvs|\bSt|\b\.g)\.\s{1,}|\?\s{1,}|$)(?=[a-z]{0,}[A-Z]|$)"  # r"(.*?)(?:\.|\?)(?=\s{1,}[A-Z]|$)"

    tasks = []
    matches = re.finditer(regex, paragraph_json['text'].strip(), re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        #task = dict()
        sentence_text = match.group().strip()
        if not sentence_text:
            continue
        sent_doc = spancat_model(sentence_text)
        sent_tokens = [x.text for x in sent_doc[0:len(sent_doc)]]
        #test = paragraph_tokens.index(sent_tokens)
        test = subfinder(paragraph_tokens, sent_tokens)

        # match up spans from paragraph to sentences
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
        sent_doc.spans['spans'] = sent_spans
        sent_doc.spans['sc'] = sent_spans

        # if rel_model:
        #     # extract relations
        #     for name, proc in rel_model.pipeline:
        #         sent_doc = proc(sent_doc)

        # create the new item to write out
        task = {'text': sent_doc.text}

        # task_spans = []
        # for span in sent_doc.spans['spans']:
        #     task_spans.append(
        #         {
        #             "token_start": span.start,
        #             "token_end": span.end - 1,
        #             "start": span.start_char,
        #             "end": span.end_char,
        #             "text": span.text,
        #             "label": span.label_
        #         }
        #     )
        task['spans'] = convert_spacy_spans_to_prodigy_spans(sent_doc)

        if rel_model:
            # rels = []
            # for value, rel_dict in sent_doc._.rel.items():
            #     for rel_label, rel_score in rel_dict.items():
            #         if min_rel_score == 0 or rel_score >= min_rel_score:
            #             skip_due_to_low_score = False
            #         if rel_score > 0.5:
            #             for e in task_spans:
            #                 for b in task_spans:
            #                     if e['token_start'] == value[0] and e['token_end'] + 1 == value[1] and e['label'] == value[2] and b['token_start'] == value[3] and b['token_end'] + 1 == value[4] and b['label'] == value[5]:
            #                         full_rel_dict = dict()
            #                         full_rel_dict['head'] = e['token_start']
            #                         full_rel_dict['child'] = b['token_start']
            #                         full_rel_dict['head_span'] = {'start': e['start'], 'end': e['end'], 'token_start': e['token_start'], 'token_end': e['token_end'], 'label': e['label']}
            #                         full_rel_dict['child_span'] = {'start': b['start'], 'end': b['end'], 'token_start': b['token_start'], 'token_end': b['token_end'], 'label': b['label']}
            #                         full_rel_dict['label'] = str(rel_label)
            #                         rels.append(full_rel_dict)

            rels = get_rels(sent_doc, rel_model)
            task["_view_id"] = "relations"
            task["relations"] = rels

        task = set_hashes(task)

        # save PMID and return the task
        task['meta'] = paragraph_json['meta']
        #task['meta']['pmid'] = int(task['meta']['pmid'])

        if type(task['meta']) is not dict:
            print('wonky pmid: ' + task['text'])
            continue
        tasks.append(task)

    return tasks

def get_rels(sent_doc, rel_model, score_cutoff = 0.5):
    for name, proc in rel_model.pipeline:
        sent_doc = proc(sent_doc)

    #prodigy_spans = sent_doc.spans['spans']
    prodigy_spans = convert_spacy_spans_to_prodigy_spans(sent_doc)
    rels = []
    for value, rel_dict in sent_doc._.rel.items():
        for rel_label, rel_score in rel_dict.items():
            if rel_score >= score_cutoff:
                for e in prodigy_spans:
                    for b in prodigy_spans:
                        if e['token_start'] == value[0] and e['token_end'] + 1 == value[1] and e['label'] == value[2] and b['token_start'] == value[3] and b['token_end'] + 1 == value[4] and b['label'] == value[5]:
                            full_rel_dict = dict()
                            full_rel_dict['head'] = e['token_start']
                            full_rel_dict['child'] = b['token_start']
                            full_rel_dict['head_span'] = {'start': e['start'], 'end': e['end'], 'token_start': e['token_start'], 'token_end': e['token_end'], 'label': e['label']}
                            full_rel_dict['child_span'] = {'start': b['start'], 'end': b['end'], 'token_start': b['token_start'], 'token_end': b['token_end'], 'label': b['label']}
                            full_rel_dict['label'] = str(rel_label)
                            #full_rel_dict['rel_score'] = rel_score
                            rels.append(full_rel_dict)

    return rels

def convert_spacy_spans_to_prodigy_spans(doc):
    task_spans = []
    for span in doc.spans['spans']:
        task_spans.append(spacy_span_to_prodigy_span(span))
    return task_spans

def spacy_span_to_prodigy_span(spacy_span):
    return {
                "token_start": spacy_span.start,
                "token_end": spacy_span.end - 1,
                "start": spacy_span.start_char,
                "end": spacy_span.end_char,
                "text": spacy_span.text,
                "label": spacy_span.label_
            }

def prodigy_span_to_spacy_span(doc, prodigy_span):
    spacy_span = doc[prodigy_span['token_start'] : prodigy_span['token_end'] + 1]
    spacy_span.label_ = prodigy_span['label']
    return spacy_span

# https://stackoverflow.com/questions/10106901/elegant-find-sub-list-in-list
def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append(i)
    return matches

def rel_model_info(spans_model, rel_model, labeled_data):
    spancat_model = spacy.load(spans_model)
    rel_model = spacy.load(rel_model)

    hits = {}
    misses = {}
    hit_scores = {}

    with open(labeled_data, 'r') as f:
        for line in f:
            data = json.loads(line)
            data["true_relations"] = data["relations"]
            pmid = str(data['meta']['pmid'])
            dev_set = pmid.endswith('1')

            if not dev_set:
                continue
            
            sent_doc = spancat_model(data["text"])
            spangroup = SpanGroup(sent_doc)
            for prodigy_span in data['spans']:
                spacy_span = prodigy_span_to_spacy_span(sent_doc, prodigy_span)
                spangroup.append(spacy_span)
            sent_doc.spans['spans'] = spangroup
            sent_doc.spans['sc'] = spangroup
            DEBUG_prodigy_spans = convert_spacy_spans_to_prodigy_spans(sent_doc)
            
            model_rels = get_rels(sent_doc, rel_model, 0)
            true_rels = data["true_relations"]

            for true_rel in true_rels:
                if 'color' in true_rel:
                    del true_rel['color']

                rel_type = true_rel["label"]

                if rel_type not in misses:
                    misses[rel_type] = 0
                    hits[rel_type] = 0
                    hit_scores[rel_type] = []
                

                matched_model_rel = find_matched_dict(true_rel, model_rels, ['color', 'rel_score', 'head', 'child'])

                if not matched_model_rel:
                    misses[rel_type] = misses[rel_type] + 1

                    if abs(true_rel['head_span']['token_start'] - true_rel['child_span']['token_start']) < 40:
                        print('text:' + data['text'])
                        print('looking for rel: ' + str(true_rel) + '\n')
                        print('in this set: ' + str.join('\n', [str(x) for x in model_rels if x['label'] == true_rel['label']]))
                        print('\n\n')
                        yea = 0
                else:
                    hits[rel_type] = hits[rel_type] + 1
                    hit_scores[rel_type].append(matched_model_rel['rel_score'])

    for key in misses:
        miss_count = misses[key]
        hit_count = hits[key]
        recall = hit_count / (hit_count + miss_count)
        print(key + ": " + str(recall))

        scores = hit_scores[key]
        with open(key + '.tsv', 'w') as f:
            for score in scores:
                f.write(str(score))
                f.write('\n')

def find_matched_dict(the_dict, list_of_dicts, keys_to_ignore):
    for item in list_of_dicts:
        return_item = True

        for key in the_dict:
            if key in keys_to_ignore:
                continue
            if key in item and the_dict[key] != item[key]:
                return_item = False

        if return_item:
            return item

    return None



def main():
    rel_model_info("./extract/models/spans/model-best", "./extract/models/rel/model-best", './label/labeled_data/spans_rel_labeled.jsonl')
    # import os
    # wd = os.getcwd()
    # spancat_model = spacy.load("./extract/models/spans/model-best")
    # rel_model = spacy.load("./extract/models/rel/model-best")

    # json_items = []
    # with open('/Users/rmillikin/My Drive/Postdoc/Projects/kinderminer_kg/label/unlabeled_data/unlabeled_training_data.jsonl', 'r') as f:
    #     for line in f:
    #         json_item = json.loads(line)
    #         json_items.append(json_item)

    # for doc in json_items:
    #     task = make_rel_tasks_from_abstract_doc(doc, spancat_model, rel_model)

if __name__ == '__main__':
    main()