FROM jupyter/datascience-notebook:latest

COPY MLOps/app/api/requirements.txt /tmp/MLOps/app/api/requirements.txt
COPY MLOps/test-requirements.txt /tmp/MLOps/test-requirements.txt

WORKDIR /tmp/MLOps
RUN pip install --no-cache-dir -r /tmp/MLOps/test-requirements.txt
