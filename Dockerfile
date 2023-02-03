FROM gcr.io/hey-i-375802/base-image:heyi
WORKDIR /app
COPY . /app
EXPOSE 8080
CMD streamlit run --server.port 8080 --server.enableCORS false /app/streamlit/HEY-I.py
