FROM python:3.12

RUN mkdir /app

COPY . /app

RUN pip install /app
WORKDIR /app
CMD gunicorn -w 4 -b '0.0.0.0:8080' 'proxy_test:create_app()'