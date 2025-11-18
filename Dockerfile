# Use an official Python runtime matching your pyproject.toml requirement
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system build dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project into the container
COPY . .

# Install the project and its dependencies
# This looks at pyproject.toml and installs 'neuro-ranker-distill' as a package
RUN pip install --no-cache-dir .

# Expose the port defined in docker-compose.yaml
EXPOSE 8000

# Run the server using the 'neurorank' CLI defined in your pyproject.toml
# We set PYTHONPATH to ensure it can find the manage module in the root
ENV PYTHONPATH=/app
CMD ["neurorank", "runserver", "--host", "0.0.0.0", "--port", "8000"]