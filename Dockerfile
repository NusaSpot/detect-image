FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir gunicorn

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "--timeout", "500", "app:app"]
