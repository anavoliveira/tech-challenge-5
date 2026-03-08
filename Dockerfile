FROM python:3.11-slim

WORKDIR /app

# Instalação de Dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY database/ ./database/

RUN python src/train.py

COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

EXPOSE 8080

CMD ["serve"]
