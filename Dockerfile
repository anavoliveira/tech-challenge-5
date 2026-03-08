FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY database/ ./database/

# Train the model at build time so it's baked into the image
RUN python src/train.py

# SageMaker requires a "serve" executable in PATH that starts the server on port 8080
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

EXPOSE 8080

CMD ["serve"]
