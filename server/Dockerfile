
FROM python:3.9

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -y fastapi uvicorn python-multipart nltk
# На неправильных версиях ничего не заработает
RUN pip install tensorflow==2.15.1 tensorflow-hub tensorflow-text==2.15.0

RUN python -m nltk.downloader stopwords

EXPOSE 5051

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5051", "--reload"]
