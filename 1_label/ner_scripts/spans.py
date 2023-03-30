from typing import List, Optional, Union, Iterable, Callable
import spacy
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.language import Language
import copy
import json
from tokenizers import BertWordPieceTokenizer
from spacy.vocab import Vocab

from spacy.matcher import Matcher

from prodigy.models.matcher import PatternMatcher
from prodigy.components.preprocess import add_tokens, make_raw_doc
from prodigy.components.loaders import get_stream
from prodigy.core import recipe
from prodigy.util import log, split_string, get_labels, msg, set_hashes, INPUT_HASH_ATTR, read_jsonl
from prodigy.types import TaskType, RecipeSettingsType, StreamType

class BertTokenizer:
    def __init__(self, vocab_file, lowercase=True):
        self.vocab = Vocab()
        self._tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=lowercase)

    def __call__(self, text):
        tokens = self._tokenizer.encode(text)
        words = []
        spaces = []
        for i, (text, (start, end)) in enumerate(zip(tokens.tokens, tokens.offsets)):
            words.append(text)
            if i < len(tokens.tokens) - 1:
                # If next start != current end we assume a space in between
                next_start, next_end = tokens.offsets[i + 1]
                spaces.append(next_start > end)
            else:
                spaces.append(True)
        the_doc = Doc(self.vocab, words=words, spaces=spaces)
        # the_doc.ents = tokens.tokens
        return the_doc

def add_tokens_bert(stream, tokenizer, hide_special, hide_wp_prefix, special_tokens, wp_prefix):
    for eg in stream:
        tokens = tokenizer.encode(eg["text"])
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
                # Don't allow selecting spacial SEP/CLS tokens
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
        eg["tokens"] = eg_tokens
        yield eg

@recipe(
    "custom.spans.manual",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    tokenizer_vocab=("Tokenizer vocab file", "option", "tv", str),
    # fmt: on
)
def manual(
    dataset: str,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    tokenizer_vocab: Optional[str] = None,
) -> RecipeSettingsType:
    """
    Annotate potentially overlapping and nested spans in the data. If
    patterns are provided, their matches are highlighted in the example, if
    available. The tokenizer is used to tokenize the incoming texts so the
    selection can snap to token boundaries. You can also set --highlight-chars
    for character-based highlighting.
    """
    log("RECIPE: Starting recipe custom.spans.manual", locals())
    
    # lowercase = True
    # hide_special = True
    # hide_wp_prefix = True

    labels = label  # comma-separated list or path to text file
    if not labels:
        msg.fail("No --label argument set", exits=1)
    log(f"RECIPE: Annotating with {len(labels)} labels", labels)
    
    stream = get_stream(source, loader=loader, input_key="text")
    # tokenizer = BertWordPieceTokenizer(tokenizer_vocab, lowercase=lowercase)
    # sep_token = tokenizer._parameters.get("sep_token")
    # cls_token = tokenizer._parameters.get("cls_token")
    # special_tokens = (sep_token, cls_token)
    # wp_prefix = tokenizer._parameters.get("wordpieces_prefix")

    # stream = add_tokens(stream, tokenizer, hide_special, hide_wp_prefix, special_tokens, wp_prefix)
    
    return {
        "view_id": "spans_manual",
        "dataset": dataset,
        "stream": stream,
        "validate_answer": None,
        "config": {
            "labels": labels,
            "exclude_by": "input",
            "auto_count_stream": True,
        },
    }

@recipe(
    "custom.spans.correct",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy pipeline with a span categorizer", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    tokenizer_vocab=("Tokenizer vocab file", "option", "tv", str),
    component=("Name of spancat component in the pipeline", "option", "c", str),
    # fmt: on
)
def correct(
    dataset: str,
    spacy_model: str,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    tokenizer_vocab: Optional[str] = None,
    component: str = "spancat",
) -> RecipeSettingsType:
    """
    Correct a span categorizer predicting potentially overlapping and nested
    spans. Requires a spaCy pipeline with a trained SpanCategorizer and will
    show all spans in the given group.
    """
    log("RECIPE: Starting recipe custom.spans.correct", locals())

    nlp = spacy.load(spacy_model)

    if component not in nlp.pipe_names:
        msg.fail(
            f"Can't find component '{component}' in pipeline. Make sure that the "
            f"pipeline you're using includes a trained span categorizer that you "
            f"can correct. If your component has a different name, you can use "
            f"the --component option to specify it.",
            exits=1,
        )
    labels = label
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

    stream = add_tokens(nlp, stream)

    def make_tasks(nlp: Language, stream: StreamType) -> StreamType:
        """Add a 'spans' key to each example, with predicted spans."""
        texts = ((eg["text"], eg) for eg in stream)
        for doc, eg in nlp.pipe(texts, as_tuples=True, batch_size=5):
            
            task = copy.deepcopy(eg)
            spans = []

            for span in doc.spans[key]:
                if labels and span.label_ not in labels:
                    continue

                spans.append(
                    {
                        "token_start": span.start,
                        "token_end": span.end - 1,
                        "start": span.start_char,
                        "end": span.end_char,
                        "text": span.text,
                        "label": span.label_,
                        "source": spacy_model,
                        "input_hash": eg[INPUT_HASH_ATTR],
                    }
                )

            _spans = []
            [_spans.append(x) for x in spans if x not in _spans]
            spans = _spans

            task["spans"] = spans
            task = set_hashes(task)
            yield task

    stream = make_tasks(nlp, stream)
    
    return {
        "view_id": "spans_manual",
        "dataset": dataset,
        "stream": stream,
        "config": {
            "lang": nlp.lang,
            "labels": labels,
            "exclude_by": "input",
            "auto_count_stream": True,
        },
    }
