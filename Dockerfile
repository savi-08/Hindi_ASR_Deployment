# Use a slim version of Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn python-multipart numpy onnxruntime soundfile librosa

# Expose port 8000
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
