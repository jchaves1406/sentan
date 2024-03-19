FROM python:3.10.12

RUN mkdir /app
WORKDIR /app

COPY ./ /app
RUN pip install .