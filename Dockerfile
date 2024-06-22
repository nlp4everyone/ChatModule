FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt requirements.txt

# install
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn","chat_api:app","--port","8080"]
