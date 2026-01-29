# Minimal environment to run the notebook + evaluation (Linux container)
FROM python:3.13-slim

WORKDIR /app

# LightGBM can require OpenMP runtime; build tools are useful if wheels are unavailable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install jupyter

COPY . .

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
