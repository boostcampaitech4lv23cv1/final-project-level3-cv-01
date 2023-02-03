FROM python:3.8
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install libgl1-mesa-glx
RUN pip3 install -r requirements.txt
