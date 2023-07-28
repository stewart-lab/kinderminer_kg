# create venv
#python3 -m venv ./spans-venv
#python -m venv ./spans-venv
#pip install --upgrade pip

# install spacy
pip install spacy==3.3.1
pip install spacy-transformers==1.1.7

# train the model
spacy project run train_gpu