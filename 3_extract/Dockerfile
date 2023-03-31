FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

RUN apt-get -y update && apt-get -y install python3.9
RUN apt-get -y install python3-pip

# install python package requirements
RUN pip install --upgrade pip \
    pip install pip setuptools wheel \
    pip install cupy-cuda114==10.6.0 \
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 

COPY . .

RUN pip install -r requirements.txt