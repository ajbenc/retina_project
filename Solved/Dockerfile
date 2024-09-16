# Use the official Python 3.9.6 image from DockerHub
FROM python:3.9.6-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install necessary system packages for h5py and TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    zlib1g-dev \
    libjpeg-dev \
    liblapack-dev \
    libblas-dev \
    gfortran

# Install pip 21.2.3
RUN pip install --upgrade pip==21.2.3

RUN pip install -r requirements.txt

# Install Jupyter Notebook
RUN pip install jupyter

# Copy the entire project into the container
COPY . .

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Set environment variable to prevent Python from buffering output
ENV PYTHONUNBUFFERED=1

# Set the default command to start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
