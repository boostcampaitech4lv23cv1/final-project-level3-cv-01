FROM python:3.8.7-slim-buster

# LABEL maintainer="dwybaek7@gmail.com"

WORKDIR /app

COPY . /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

RUN apt-get update && apt-get install -y \
    install --upgrade pip && \
    pip install -r requirements.txt
    #apt-get -y install python-pip python-dev libgl1-mesa-glx \
    #apt-get -y install build-essential cmake \
    #apt-get -y install libgtk-3-dev \
    #apt-get -y install libboost-all-dev \

ENTRYPOINT ["streamlit", "run", "streamlit/HEY-I.py", "--server.address=0.0.0.0", "--server.port 8080"]