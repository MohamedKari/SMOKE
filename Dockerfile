FROM nvidia/cuda:10.2-devel-ubuntu16.04

##### CONDA #####
RUN apt-get update -y && \
    apt-get install -y \
        wget  
RUN wget --progress=dot:mega https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV PATH="/root/miniconda3/condabin:${PATH}"

##### PROJECT DEPENDENCIES #####
WORKDIR /app
COPY env.yml env.yml
RUN conda env update -f env.yml --name base

# required for smoke/data/datasets/evaluation/kitti/kitti_eval
RUN apt-get -y install libboost-all-dev

# required by evaluate_object_offline.cpp (gnuplot, ps2pdf, pdfcrop)
RUN apt-get -y install \
        gnuplot gnuplot-x11 gnuplot-doc \
        ghostscript \
        texlive-extra-utils  

##### APPLICATION #####
ENV PYTHONUNBUFFERED=.
COPY . .

RUN cd /app/smoke/data/datasets/evaluation/kitti/kitti_eval && \
    g++ -O3 -DNDEBUG -o evaluate_object_offline evaluate_object_offline.cpp
# RUN python setup.py build develop

ENTRYPOINT [ "bash" ]

