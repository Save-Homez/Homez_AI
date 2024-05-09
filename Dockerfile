FROM python:3.9.17-slim
RUN pip install pipenv
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ["src/*.py", "data/*.csv", "model/*.pkl", "./"]
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "homez_ai:app"]
