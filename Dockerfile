# FROM python:3.8-slim
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install \
    python3 \
    python3-pip \
    git \
    -y

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install --no-cache-dir -r requirements.txt
