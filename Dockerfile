FROM python:3.11-slim

# Install Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python deps
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Bundle tessdata
COPY tessdata /app/tessdata
ENV TESSDATA_PREFIX=/app/tessdata

# Copy app code
COPY . /app

# Gradio/Spaces/Railway port
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

# Run app
CMD ["python", "app.py"]
