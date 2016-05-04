FROM yandex/rep:0.6.5
MAINTAINER Alexander Panin <justheuristic@gmail.com>
RUN apt-get -qq update

USER root
RUN source activate rep_py2 && \ 
    pip install --upgrade pip && \
    pip install --upgrade https://github.com/Theano/Theano/archive/master.zip\
    pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip\
    
RUN source activate rep_py2 && \
    cd $HOME && git clone https://github.com/yandexdataschool/AgentNet -b develop &&\
    cd AgentNet && python setup.py install


