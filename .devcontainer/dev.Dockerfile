FROM python:3.10.14-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC

# Install apt packages.
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends \
        vim \
        tmux \
        build-essential \
        libosmesa6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

RUN pip install -U pip \
    && pip install nvitop
