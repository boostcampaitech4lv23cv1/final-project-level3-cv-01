FROM python:3.8.7-slim-buster

# LABEL maintainer="dwybaek7@gmail.com"

WORKDIR /app

COPY . /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

RUN apt-get update && apt-get install -y \
    apt-get -y install python-pip python-dev libgl1-mesa-glx \
    apt-get install build-essential cmake \
    apt-get install libgtk-3-dev \
    apt-get install libboost-all-dev \
    install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "streamlit/HEY-I.py", "--server.address=0.0.0.0", "--server.port 8080"]