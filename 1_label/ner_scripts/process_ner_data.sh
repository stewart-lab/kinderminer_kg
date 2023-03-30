prodigy db-out pubmed_spancat > ../labeled_data/spans_labeled.jsonl
prodigy data-to-spacy ../../train/spans/data --spancat pubmed_spancat -es 0.2