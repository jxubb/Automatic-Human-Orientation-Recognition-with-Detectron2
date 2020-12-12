FROM nvidia/cuda:10.1-cudnn7-devel

COPY . .

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
  python3-pip python3-dev \
  nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev \
  'ffmpeg' \
   locales

RUN pip3 install setuptools --upgrade
RUN pip3 install --upgrade pip
RUN pip3 install cython
RUN pip3 install fastapi==0.60.1
RUN pip3 install uvicorn
RUN pip3 install opencv-python
RUN pip3 install numpy==1.17.2
RUN pip3 install python-multipart
RUN pip3 install pillow
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

# Detectron2 - CUDA 10.1 copy
RUN pip3 install torch==1.7 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html
RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

# Set the locale (required for uvicorn)
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
  dpkg-reconfigure --frontend=noninteractive locales && \
  update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

WORKDIR .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8860"]
