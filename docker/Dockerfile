FROM pytorch/pytorch
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install wget python3-pip libcairo2-dev pkg-config python3-dev libgirepository1.0-dev vim nano xorg openbox --assume-yes 
RUN pip install pycairo PyGObject
ENV CUDA_HOME /usr/local/cuda

ADD . /home/pxai/

WORKDIR /home/pxai/
RUN conda install --file requirements.txt
RUN conda install -y -c pytorch \
    cudatoolkit=10.0 \
    "pytorch=1.2.0=py3.6_cuda10.0.130_cudnn7.6.2_0" \
    "torchvision=0.4.0=py36_cu100" \
    "scikit-learn" \
 && conda clean -ya

WORKDIR /home/pxai/weights/
RUN chmod +x download_realtime_saliency.sh download_patterns.sh
RUN ./download_realtime_saliency.sh
RUN ./download_patterns.sh

WORKDIR /home/pxai/
RUN useradd -ms /bin/bash ubuntu
RUN python3 -c "import torchvision.models as models; vgg16 = models.vgg16(pretrained=True)"
RUN /bin/bash
