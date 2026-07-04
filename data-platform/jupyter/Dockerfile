FROM jupyter/datascience-notebook:latest

COPY Model/api/requirements.txt /tmp/Model/api/requirements.txt
COPY Model/test-requirements.txt /tmp/Model/test-requirements.txt

RUN pip install --no-cache-dir -r /tmp/Model/test-requirements.txt
