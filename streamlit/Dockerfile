FROM python:3.8
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install libgl1-mesa-glx
EXPOSE 8080
COPY . /app
CMD streamlit run --server.port 8080 --server.enableCORS false HEY-I.py