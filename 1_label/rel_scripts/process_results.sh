prodigy db-out pubmed_ner_rel_sentences > ../labeled_data/spans_rel_labeled.jsonl
python ./parse_data.py ../labeled_data/spans_rel_labeled.jsonl ../../train/rel/data/train.spacy ../../train/rel/data/dev.spacy ../../train/rel/data/test.spacy
cp ../labeled_data/spans_rel_labeled.jsonl ../../train/rel/data/spans_rel_labeled.jsonl
cp ./rel_model.py ../../extract/rel_model.py
cp ./rel_pipe.py ../../extract/rel_pipe.py
cp ./rel_model.py ../../train/rel/rel_model.py
cp ./rel_pipe.py ../../train/rel/rel_pipe.py