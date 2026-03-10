# =============================================================================
# FRAUD DETECTION API - DOCKERFILE
# =============================================================================
# This Dockerfile defines the environment to run the FastAPI application
# in an isolated container. It uses a lightweight Python 3.9 image.
# =============================================================================

# Base image - lightweight Python 3.9
FROM python:3.9-slim
# Set the working directory inside the container
WORKDIR /app
# Copy all project files into the container
COPY . .
# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Expose port 80 to allow external access
EXPOSE 80
# Launch the FastAPI application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0","--port","80"]