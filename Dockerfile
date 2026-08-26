FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ENV LOG_LEVEL=INFO
ENV SECRET_KEY=change-me-in-production
ENV API_KEY=

CMD ["python", "app.py"]
