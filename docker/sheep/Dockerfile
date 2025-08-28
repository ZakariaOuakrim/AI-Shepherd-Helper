# Producer Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy producer script and CSV file
COPY data-gen.py .
COPY sheep_gps_data.csv .

# Default command
CMD ["python", "data-gen.py"]
