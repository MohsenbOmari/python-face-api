# Use the official lightweight Python image.
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Tell deepface where to store models in a writable directory
ENV DEEPFACE_HOME /tmp/.deepface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# This command pre-downloads the AI model during the build process
RUN python -c "from deepface import DeepFace; DeepFace.build_model('VGG-Face')"

# Copy the main application file into the container
COPY main.py .

# Expose the port that the application will run on
EXPOSE 10000

# --- تم التعديل هنا ---
# Run the application using gunicorn with an increased timeout of 120 seconds
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "120", "main:app"]
