FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && \
	DEBIAN_FRONTEND="noninteractive" apt-get install -y \
	tzdata \
	vim \
	git \
	python3 \
	python3-pip

RUN pip3 install -U pip setuptools wheel

RUN pip3 install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'

RUN pip3 uninstall ninja -y && pip install ninja -U
RUN pip3 install -v -U "git+https://github.com/facebookresearch/xformers.git@main#egg=xformers"

RUN pip3 install packaging
RUN git clone https://github.com/Dao-AILab/flash-attention \
	&& cd flash-attention && python3 setup.py install \
	&& cd csrc/rotary && pip3 install . \
	&& cd ../layer_norm && pip3 install . \
	&& cd ../xentropy && pip3 install . \
	&& cd ../.. && rm -rf flash-attention
COPY ./tiny_llama_requirements.txt ./tiny_llama_requirements.txt
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r tiny_llama_requirements.txt
RUN pip3 install -r requirements.txt

ENV PYTHONPATH=/lib/python3.8/site-packages/flash_attn-2.3.6-py3.8-linux-x86_64.egg/
RUN ln -s /usr/bin/python3 /usr/bin/python

# matplotlib fonts
RUN apt install -y font-manager && rm ~/.cache/matplotlib -fr
