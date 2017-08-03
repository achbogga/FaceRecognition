FROM ov-tf-new
ADD . /FR
WORKDIR /FR/Codes/facenet
RUN pip2 install -Ur requirements.txt
RUN apt-get update -y && \
    apt-get install apt-utils -y
ENV PYTHONPATH=/FR/Codes/facenet/src:/FR/Codes/facenet/src/align:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=1
RUN git config --global push.default simple && \
    git config --global user.email "aboggaram@objectvideo.com" && \
    git config --global user.name "Achyut Boggaram"
RUN apt-get install vim -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
WORKDIR /FR

