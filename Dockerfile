FROM python:3.11

RUN mkdir -p /app
WORKDIR /app
COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /app/requirements.txt

VOLUME /app
COPY . /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]