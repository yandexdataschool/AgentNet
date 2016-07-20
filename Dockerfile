FROM yandex/rep:0.6.5
MAINTAINER Alexander Panin <justheuristic@gmail.com>

RUN apt-get -qq update
RUN apt-get install -y libopenblas-dev
#RUN apt-get install -y cmake
RUN apt-get install -y zlib1g-dev
RUN apt-get install -y libjpeg-dev 
RUN apt-get install -y xvfb libav-tools xorg-dev python-opengl
RUN apt-get -y install swig #!This won't work with Box2D!


RUN /bin/bash --login -c "\
    source activate rep_py2 && \
    pip install --upgrade pip && \
    pip install --upgrade https://github.com/Theano/Theano/archive/master.zip &&\
    pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip &&\
    pip install --no-deps --upgrade https://github.com/yandexdataschool/AgentNet/archive/master.zip \
    "

RUN /bin/bash --login -c "\
    source activate jupyterhub_py3 && \ 
    pip install --upgrade pip && \
    pip install --upgrade https://github.com/Theano/Theano/archive/master.zip &&\
    pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip &&\
    pip install --no-deps --upgrade https://github.com/yandexdataschool/AgentNet/archive/master.zip \
    "

RUN /bin/bash --login -c "\
    git clone https://github.com/yandexdataschool/AgentNet -b master &&\
    sed -i -e '3iln -s ~/AgentNet/examples /notebooks/agentnet_examples\' /root/install_modules.sh \
    "
    
    
    

