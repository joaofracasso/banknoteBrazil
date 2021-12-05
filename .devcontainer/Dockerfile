# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster

FROM nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update \
     && apt-get -y install python3.8 wget python3.8-distutils git

RUN cd /tmp && wget https://bootstrap.pypa.io/get-pip.py && \
    python3.8 /tmp/get-pip.py


RUN python3.8 -m pip install pip -U

RUN python3.8 -m pip install pylint pytest

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY app/requirements.txt /tmp/pip-tmp/
RUN python3.8 -m pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

