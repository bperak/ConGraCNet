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

# Install Streamlit explicitly (critical for the app)
RUN pip install streamlit==1.27.0

# Install PyTorch CPU-only version (smaller and more reliable)
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining packages from requirements.txt (fail on error)
RUN pip install -r /app/requirements.txt

# Ensure py2neo is installed explicitly (safety) and verify
RUN pip install --no-deps py2neo==2021.2.3 && \
    python -c "import py2neo; print('py2neo version:', py2neo.__version__)"

# Install spaCy model separately (non-fatal if it fails due to network)
RUN python -m spacy download en_core_web_sm || true

# Install NLTK data needed by wordnet/vader and make it available to appuser
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN mkdir -p /usr/local/share/nltk_data && \
    python -c "import nltk; nltk.download('wordnet', download_dir='/usr/local/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/local/share/nltk_data'); nltk.download('vader_lexicon', download_dir='/usr/local/share/nltk_data')"

# Verify critical packages are installed
RUN python -c "import streamlit; print('Streamlit version:', streamlit.__version__)" && \
    python -c "import pandas; print('Pandas version:', pandas.__version__)" && \
    python -c "import numpy; print('NumPy version:', numpy.__version__)" && \
    python -c "import py2neo; print('py2neo import OK')" && \
    python -c "from nltk.corpus import wordnet as wn; from nltk.sentiment.vader import SentimentIntensityAnalyzer; print('NLTK corpora OK')" && \
    python -c "import igraph, louvain, leidenalg, networkx; print('Graph libs OK')" && \
    python -c "import plotly_resampler as pr; print('plotly_resampler OK')" && \
    python -c "import spacy, spacy_wordnet; print('spaCy + spacy_wordnet OK')"

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


