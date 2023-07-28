# create venv
#python3 -m venv ./spans-venv
#python -m venv ./spans-venv
#pip install --upgrade pip

# install spacy
pip install spacy==3.3.1
pip install spacy-transformers==1.1.7

# fill config file (need to download base cfg file first from https://spacy.io/usage/training )
python -m spacy init fill-config ./config.cfg ./filled_config.cfg

# debug the filled config file
python -m spacy debug data ./filled_config.cfg

# train the model
python -m spacy train ./filled_config.cfg --gpu-id 0 --output ./model/