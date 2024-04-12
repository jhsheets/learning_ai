FROM python:3.10-slim-bullseye

EXPOSE 8080

WORKDIR /app

# Build and install llamacpp
RUN apt-get update && apt-get install -y git && apt-get install -y build-essential
RUN pip install llama-cpp-python

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY app.py .
COPY --from=shared measure.py .

CMD ["python", "/app/app.py"]