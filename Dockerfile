# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for pdfplumber and chromadb
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# We will handle the actual command in docker-compose