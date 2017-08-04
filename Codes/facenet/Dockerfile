FROM ov-tf
RUN git clone https://achbogga:Achyut%401994@github.com/achbogga/facenet-ovi.git /facenet
WORKDIR /facenet
ADD . /facenet/
RUN git config --global url.git@github.com:achbogga/facenet-ovi.git.insteadOf https://achbogga:Achyut%401994@github.com/achbogga/facenet-ovi.git
RUN pip2 install -Ur requirements.txt
RUN apt-get install apt-utils -y
ENV PYTHONPATH=/facenet/src:/facenet/src/align:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=1
RUN git config --global push.default simple && \
    git config --global user.email "aboggaram@objectvideo.com" && \
    git config --global user.name "Achyut Boggaram"
RUN apt-get install vim -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

