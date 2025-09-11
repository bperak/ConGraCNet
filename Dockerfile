# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# System deps for scientific stack and igraph
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      g++ \
      curl \
      git \
      pkg-config \
      libxml2-dev \
      libz-dev \
      libssl-dev \
      libffi-dev \
      libglib2.0-0 \
      libgl1 \
      libgomp1 \
      libstdc++6 \
      graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install igraph dependencies separately
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libigraph0-dev \
      libglpk-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* || echo "Some packages not available, continuing..."

# Copy only requirements first for layer caching
COPY requirements.txt /app/requirements.txt

# Install Python deps. Prefer CPU-only by default.
RUN pip install --upgrade pip

# Install core packages first
RUN pip install python-dotenv numpy pandas scikit-learn

# Install PyTorch CPU-only version (smaller and more reliable)
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining packages from requirements.txt
RUN pip install -r /app/requirements.txt || echo "Some packages failed, continuing..."

# Install spaCy model separately
RUN python -m spacy download en_core_web_sm || echo "spaCy model download failed, continuing..."

# Copy the app
COPY . /app

# Ensure authSettings.py exists in the image (copy from example if missing)
RUN if [ ! -f /app/authSettings.py ]; then \
      if [ -f /app/authSettings.py.example ]; then \
        cp /app/authSettings.py.example /app/authSettings.py; \
      else \
        printf "graphUser=\"neo4j\"\ngraphPass=\"change-me\"\ngraphURL=\"bolt://localhost:7687\"\nuserName=\"user\"\napiKey=\"key\"\n" > /app/authSettings.py; \
      fi; \
    fi

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Default environment for Streamlit to be accessible externally
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Healthcheck: check the TCP port is open
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/8501 || exit 1'

# Run the Streamlit app. Allow overriding via CMD.
CMD ["streamlit", "run", "cgcnStream_0_3_6_withSBBLabel.py"]


