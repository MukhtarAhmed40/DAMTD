FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /workspace/requirements.txt

COPY . /workspace

CMD ["bash", "-lc", "echo 'Container ready. To run reproduce: python code/preprocess_general.py --dataset LSNM2024; python code/train.py --config configs/damtd_LSNM2024.yaml'"]
