FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

RUN apt-get -y update && apt-get -y install python3.9
RUN apt-get -y install python3-pip

# install python package requirements
RUN pip install --upgrade pip \
    pip install pip setuptools wheel \
    pip install cupy-cuda114 \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 \
    pip install -U 'spacy[transformers,cuda114]'

COPY . .