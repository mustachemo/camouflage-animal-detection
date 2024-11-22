FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "run.py"]
