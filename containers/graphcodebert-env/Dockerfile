FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda

RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
	

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=CONDA_DIR/bin:PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*
	
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install -y mamba -c conda-forge

COPY environment.yml .
RUN conda env create --file ./environment.yml &&\
    conda clean -tipy
	
COPY entrypoint.sh .
RUN chmod +x ./entrypoint.sh

# Set entrypoint to bash
ENTRYPOINT ["./entrypoint.sh"]