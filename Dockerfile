FROM python:3.8-slim-buster

COPY . .


RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev

# Detectron2 prerequisites
RUN pip install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip install fastapi==0.60.1
RUN pip install uvicorn
RUN pip install opencv-python
RUN pip install numpy==1.17.2
RUN pip install python-multipart
RUN pip install pillow
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

# Detectron2 - CPU copy
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html

WORKDIR .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
