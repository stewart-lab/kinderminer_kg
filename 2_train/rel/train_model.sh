# create venv
python3 -m venv ./spans-venv
python -m venv ./spans-venv
pip install --upgrade pip

# install spacy
pip install spacy==3.3.1
pip install spacy-transformers

# train the model
spacy project run train_gpu