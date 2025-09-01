# ConGraCNet Installation Guide

> **Complete step-by-step installation instructions for ConGraCNet**

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Installation Methods](#installation-methods)
4. [Environment Setup](#environment-setup)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Setup](#advanced-setup)
9. [Production Deployment](#production-deployment)
10. [Uninstallation](#uninstallation)

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.9 or higher
- **Memory**: 8GB RAM
- **Storage**: 10GB free space
- **Network**: Internet connection for package installation and API access

### Recommended Requirements

- **Operating System**: Windows 11+, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python**: 3.11 or higher
- **Memory**: 16GB RAM or higher
- **Storage**: 50GB free space (for large corpora)
- **Network**: Stable broadband connection
- **Graphics**: Hardware acceleration support for 3D visualizations

### Hardware Considerations

#### For Large-Scale Analysis
- **Memory**: 32GB+ RAM for networks with 100,000+ nodes
- **Storage**: SSD with 100GB+ free space
- **CPU**: Multi-core processor (8+ cores recommended)
- **GPU**: CUDA-compatible GPU for accelerated computations

#### For Development
- **Memory**: 16GB+ RAM
- **Storage**: 25GB+ free space
- **CPU**: 4+ cores
- **Development Tools**: Git, code editor, terminal

## Prerequisites

### Required Software

#### 1. Python Installation

**Windows:**
```bash
# Download from python.org
# Or use Microsoft Store
# Or use Chocolatey
choco install python

# Verify installation
python --version
pip --version
```

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.11

# Or download from python.org
# Verify installation
python3 --version
pip3 --version
```

**Linux (Ubuntu/Debian):**
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3.11 python3.11-venv python3.11-dev python3-pip

# Verify installation
python3.11 --version
pip3 --version
```

**Linux (CentOS/RHEL):**
```bash
# Install EPEL repository
sudo yum install epel-release

# Install Python
sudo yum install python3.11 python3.11-pip python3.11-devel

# Verify installation
python3.11 --version
pip3.11 --version
```

#### 2. Git Installation

**Windows:**
```bash
# Download from git-scm.com
# Or use Chocolatey
choco install git

# Verify installation
git --version
```

**macOS:**
```bash
# Using Homebrew
brew install git

# Or download from git-scm.com
# Verify installation
git --version
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install git

# CentOS/RHEL
sudo yum install git

# Verify installation
git --version
```

#### 3. Database Setup

**Neo4j Desktop (Recommended for Development):**
1. Download from [neo4j.com](https://neo4j.com/download/)
2. Install and create a new database
3. Note the connection details (URL, username, password)

**Neo4j Community Edition (Linux/Production):**
```bash
# Ubuntu/Debian
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 4.4' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j

# Start Neo4j service
sudo systemctl start neo4j
sudo systemctl enable neo4j

# Set initial password
sudo neo4j-admin set-initial-password your_password
```

**Neo4j Docker (Alternative):**
```bash
# Pull Neo4j image
docker pull neo4j:4.4

# Run Neo4j container
docker run \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  -e NEO4J_PLUGINS='["apoc"]' \
  -v $HOME/neo4j/data:/data \
  -v $HOME/neo4j/logs:/logs \
  -v $HOME/neo4j/import:/var/lib/neo4j/import \
  -v $HOME/neo4j/plugins:/plugins \
  neo4j:4.4
```

#### 4. Sketch Engine API Access

1. **Create Account**: Visit [sketchengine.eu](https://www.sketchengine.eu/)
2. **Request API Access**: Contact support for API credentials
3. **Get Credentials**: Username and API key
4. **Verify Access**: Test API connectivity

### System Dependencies

#### Windows Dependencies
```bash
# Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or install using Chocolatey
choco install visualstudio2019buildtools
choco install visualstudio2019-workload-vctools
```

#### macOS Dependencies
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Linux Dependencies
```bash
# Ubuntu/Debian
sudo apt install build-essential libssl-dev libffi-dev python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install openssl-devel libffi-devel python3-devel
```

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### 1. Clone Repository
```bash
# Clone the repository
git clone https://github.com/bperak/ConGraCNet.git
cd ConGraCNet

# Verify clone
ls -la
```

#### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Verify activation
which python  # Should point to venv directory
pip list      # Should show minimal packages
```

#### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Verify installation
pip list
python -c "import streamlit, pandas, networkx; print('Dependencies installed successfully')"
```

#### 4. Install Development Dependencies (Optional)
```bash
# Install development tools
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Method 2: Conda Installation

#### 1. Create Conda Environment
```bash
# Create new conda environment
conda create --name congracnet python=3.11

# Activate environment
conda activate congracnet

# Verify activation
conda info --envs
```

#### 2. Install Dependencies
```bash
# Install core packages
conda install -c conda-forge streamlit pandas networkx plotly

# Install remaining packages via pip
pip install -r requirements.txt
```

### Method 3: Docker Installation

#### 1. Create Dockerfile
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "cgcnStream_0_3_6_withSBBLabel.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### 2. Build and Run Docker Container
```bash
# Build image
docker build -t congracnet .

# Run container
docker run -p 8501:8501 congracnet

# Or run with volume mounting for development
docker run -p 8501:8501 -v $(pwd):/app congracnet
```

## Environment Setup

### Virtual Environment Management

#### Activating Virtual Environment
```bash
# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Verify activation
echo $VIRTUAL_ENV  # Should show venv path
which python       # Should point to venv
```

#### Deactivating Virtual Environment
```bash
deactivate
```

#### Managing Multiple Environments
```bash
# Create environment for specific Python version
python3.9 -m venv venv39
python3.10 -m venv venv310
python3.11 -m venv venv311

# Activate specific environment
source venv311/bin/activate
```

### Environment Variables

#### Setting Environment Variables

**Windows (PowerShell):**
```powershell
# Set environment variables
$env:PYTHONPATH = "C:\path\to\congracnet"
$env:STREAMLIT_SERVER_PORT = "8501"

# Or set permanently
[Environment]::SetEnvironmentVariable("PYTHONPATH", "C:\path\to\congracnet", "User")
```

**Windows (Command Prompt):**
```cmd
# Set environment variables
set PYTHONPATH=C:\path\to\congracnet
set STREAMLIT_SERVER_PORT=8501

# Or set permanently
setx PYTHONPATH "C:\path\to\congracnet"
```

**macOS/Linux:**
```bash
# Set environment variables
export PYTHONPATH="/path/to/congracnet"
export STREAMLIT_SERVER_PORT="8501"

# Add to shell profile for persistence
echo 'export PYTHONPATH="/path/to/congracnet"' >> ~/.bashrc
echo 'export STREAMLIT_SERVER_PORT="8501"' >> ~/.bashrc
source ~/.bashrc
```

#### Environment File (.env)
```bash
# Create .env file
cat > .env << EOF
# Database Configuration
NEO4J_URL=http://localhost:7474
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# API Configuration
SKETCH_ENGINE_USERNAME=your_username
SKETCH_ENGINE_API_KEY=your_api_key

# Application Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
LOG_LEVEL=INFO
EOF

# Load environment variables
source .env
```

### Path Configuration

#### Adding to PATH

**Windows:**
```powershell
# Add to PATH temporarily
$env:PATH += ";C:\path\to\congracnet\venv\Scripts"

# Add to PATH permanently
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\path\to\congracnet\venv\Scripts", "User")
```

**macOS/Linux:**
```bash
# Add to PATH temporarily
export PATH="$PATH:/path/to/congracnet/venv/bin"

# Add to PATH permanently
echo 'export PATH="$PATH:/path/to/congracnet/venv/bin"' >> ~/.bashrc
source ~/.bashrc
```

## Configuration

### Database Configuration

#### 1. Create Configuration File
```bash
# Copy example configuration
cp authSettings.py.example authSettings.py

# Edit configuration file
nano authSettings.py  # or use your preferred editor
```

#### 2. Configure Neo4j Connection
```python
# authSettings.py
# Neo4j Database Configuration
graphURL = "http://localhost:7474"  # or your Neo4j server URL
graphUser = "neo4j"                 # default username
graphPass = "your_password"         # your database password

# For remote Neo4j server
# graphURL = "http://your-server:7474"
# graphUser = "your_username"
# graphPass = "your_password"
```

#### 3. Test Database Connection
```python
# Test connection in Python
from py2neo import Graph
import authSettings as aS

try:
    graph = Graph(aS.graphURL, auth=(aS.graphUser, aS.graphPass))
    result = graph.run("RETURN 1 as test").evaluate()
    print(f"Database connection successful: {result}")
except Exception as e:
    print(f"Database connection failed: {e}")
```

### API Configuration

#### 1. Configure Sketch Engine API
```python
# authSettings.py
# Sketch Engine API Configuration
userName = "your_sketch_engine_username"
apiKey = "your_sketch_engine_api_key"
```

#### 2. Test API Connection
```python
# Test API connection
import requests

def test_sketch_engine_api():
    """Test Sketch Engine API connectivity."""
    try:
        # Test basic connectivity
        response = requests.get("https://api.sketchengine.eu/health", timeout=5)
        if response.status_code == 200:
            print("Sketch Engine API is accessible")
        else:
            print(f"API returned status code: {response.status_code}")
    except Exception as e:
        print(f"API connection failed: {e}")

test_sketch_engine_api()
```

### Application Configuration

#### 1. Streamlit Configuration
```bash
# Create Streamlit configuration directory
mkdir -p ~/.streamlit

# Create configuration file
cat > ~/.streamlit/config.toml << EOF
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
fileWatcherType = "none"

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
EOF
```

#### 2. Logging Configuration
```python
# Create logging configuration
import logging
import logging.handlers
import os

def setup_logging():
    """Set up application logging."""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/congracnet.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('streamlit').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

setup_logging()
```

## Verification

### Installation Verification

#### 1. Check Python Environment
```bash
# Verify Python version
python --version  # Should be 3.9+

# Verify pip installation
pip --version

# Check virtual environment
echo $VIRTUAL_ENV  # Should show venv path
which python       # Should point to venv
```

#### 2. Verify Dependencies
```bash
# Check installed packages
pip list

# Verify key packages
python -c "
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import py2neo
print('All core dependencies installed successfully')
"
```

#### 3. Test Application Startup
```bash
# Test Streamlit application
streamlit run cgcnStream_0_3_6_withSBBLabel.py --server.port 8501

# Check if application starts without errors
# Look for any error messages in the console
```

### Functionality Verification

#### 1. Database Connection Test
```python
# Test database connectivity
from py2neo import Graph
import authSettings as aS

def test_database():
    """Test Neo4j database connection."""
    try:
        graph = Graph(aS.graphURL, auth=(aS.graphUser, aS.graphPass))
        
        # Test basic query
        result = graph.run("MATCH (n) RETURN count(n) as node_count").evaluate()
        print(f"Database connection successful. Node count: {result}")
        
        # Test specific corpus query
        result = graph.run("MATCH (n:Lemma) RETURN count(n) as lemma_count").evaluate()
        print(f"Lemma count: {result}")
        
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

test_database()
```

#### 2. API Connection Test
```python
# Test Sketch Engine API
import requests
import authSettings as aS

def test_api():
    """Test Sketch Engine API connectivity."""
    try:
        # Test basic connectivity
        response = requests.get("https://api.sketchengine.eu/health", timeout=10)
        print(f"API health check: {response.status_code}")
        
        # Test with credentials (if available)
        if hasattr(aS, 'userName') and hasattr(aS, 'apiKey'):
            print("API credentials configured")
        else:
            print("API credentials not configured")
            
        return True
    except Exception as e:
        print(f"API test failed: {e}")
        return False

test_api()
```

#### 3. Core Functionality Test
```python
# Test core functions
import cgcn_functions_3_6 as cgcn

def test_core_functions():
    """Test core functionality."""
    try:
        # Test corpus structure function
        corpus_structure = cgcn.struktura_korpusa()
        print(f"Corpus structure loaded: {len(corpus_structure)} corpora")
        
        # Test database query function
        db_structure = cgcn.databaza_izlistaj_strukturu_upita()
        print("Database structure query successful")
        
        return True
    except Exception as e:
        print(f"Core functions test failed: {e}")
        return False

test_core_functions()
```

## Troubleshooting

### Common Installation Issues

#### 1. Python Version Issues
**Problem**: Wrong Python version or multiple Python installations
```bash
# Check Python versions
python --version
python3 --version
python3.11 --version

# Use specific Python version for venv
python3.11 -m venv venv311
source venv311/bin/activate
```

**Solution**: Ensure using correct Python version and create venv with specific version

#### 2. Package Installation Failures
**Problem**: Package installation errors during pip install
```bash
# Upgrade pip
pip install --upgrade pip

# Install build tools (Windows)
# Install Visual C++ Build Tools

# Install system dependencies (Linux)
sudo apt install build-essential libssl-dev libffi-dev python3-dev

# Try alternative installation
pip install --no-cache-dir -r requirements.txt
```

**Solution**: Install system dependencies and upgrade pip

#### 3. Virtual Environment Issues
**Problem**: Virtual environment not activating or not found
```bash
# Check if venv exists
ls -la venv/

# Recreate venv if corrupted
rm -rf venv
python -m venv venv

# Activate with full path
source ./venv/bin/activate

# Check activation
which python
echo $VIRTUAL_ENV
```

**Solution**: Recreate virtual environment and verify activation

### Database Connection Issues

#### 1. Neo4j Connection Failed
**Problem**: Cannot connect to Neo4j database
```bash
# Check if Neo4j is running
# Windows: Check Services
# Linux: sudo systemctl status neo4j

# Test connection manually
curl http://localhost:7474

# Check firewall settings
# Windows: Windows Firewall
# Linux: sudo ufw status
```

**Solution**: Start Neo4j service and check firewall settings

#### 2. Authentication Errors
**Problem**: Wrong username/password
```bash
# Reset Neo4j password
sudo neo4j-admin set-initial-password new_password

# Check authSettings.py
cat authSettings.py

# Test with new credentials
python -c "
from py2neo import Graph
graph = Graph('http://localhost:7474', auth=('neo4j', 'new_password'))
print('Connection successful')
"
```

**Solution**: Reset password and update configuration

### API Connection Issues

#### 1. Sketch Engine API Unreachable
**Problem**: Cannot connect to Sketch Engine API
```bash
# Test internet connectivity
ping api.sketchengine.eu

# Test API endpoint
curl -I https://api.sketchengine.eu/health

# Check proxy settings if behind corporate firewall
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

**Solution**: Check internet connectivity and proxy settings

#### 2. API Authentication Errors
**Problem**: Invalid API credentials
```bash
# Verify credentials in authSettings.py
grep -E "userName|apiKey" authSettings.py

# Test credentials manually
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.sketchengine.eu/endpoint
```

**Solution**: Verify and update API credentials

### Performance Issues

#### 1. Slow Application Startup
**Problem**: Application takes long time to start
```bash
# Check system resources
# Windows: Task Manager
# Linux: htop or top

# Optimize virtual environment
pip install --upgrade pip
pip install --upgrade setuptools wheel

# Use faster package manager
pip install --upgrade pip-tools
```

**Solution**: Optimize system resources and package installation

#### 2. Memory Issues
**Problem**: Application runs out of memory
```bash
# Check memory usage
# Windows: Task Manager
# Linux: free -h

# Reduce network size limits in application
# Use smaller corpora for testing
# Enable data pruning options
```

**Solution**: Monitor memory usage and optimize application parameters

## Advanced Setup

### Development Environment

#### 1. Install Development Tools
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Install additional development tools
pip install black isort mypy pytest pytest-cov
```

#### 2. Configure Code Quality Tools
```bash
# Black configuration
cat > pyproject.toml << EOF
[tool.black]
line-length = 120
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
EOF

# isort configuration
cat > .isort.cfg << EOF
[settings]
profile = black
line_length = 120
multi_line_output = 3
include_trailing_comma = True
force_grid_wrap = 0
use_parentheses = True
ensure_newline_before_comments = True
EOF
```

#### 3. Set Up Testing Environment
```bash
# Create tests directory
mkdir -p tests

# Create test configuration
cat > pytest.ini << EOF
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    performance: marks tests as performance tests
EOF

# Run tests
pytest tests/
```

### Production Environment

#### 1. System Optimization
```bash
# Linux system optimization
# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize kernel parameters
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Optimize disk I/O
echo "vm.swappiness=1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### 2. Service Configuration
```bash
# Create systemd service
sudo tee /etc/systemd/system/congracnet.service << EOF
[Unit]
Description=ConGraCNet Streamlit Application
After=network.target

[Service]
Type=simple
User=congracnet
WorkingDirectory=/home/congracnet/app
Environment=PATH=/home/congracnet/app/venv/bin
ExecStart=/home/congracnet/app/venv/bin/streamlit run cgcnStream_0_3_6_withSBBLabel.py --server.port 7475 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable congracnet
sudo systemctl start congracnet
```

#### 3. Reverse Proxy Setup
```bash
# Install Nginx
sudo apt install nginx

# Configure Nginx
sudo tee /etc/nginx/sites-available/congracnet << EOF
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:7475;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/congracnet /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Production Deployment

### Deployment Checklist

#### 1. Pre-deployment
- [ ] System requirements met
- [ ] Dependencies installed
- [ ] Configuration files updated
- [ ] Database connection tested
- [ ] API credentials verified
- [ ] Security settings configured
- [ ] Monitoring tools installed
- [ ] Backup procedures established

#### 2. Deployment Steps
```bash
# 1. Create deployment user
sudo useradd -m -s /bin/bash congracnet
sudo usermod -aG sudo congracnet

# 2. Clone application
sudo -u congracnet git clone https://github.com/bperak/ConGraCNet.git /home/congracnet/app

# 3. Set up virtual environment
sudo -u congracnet python3.11 -m venv /home/congracnet/app/venv
sudo -u congracnet /home/congracnet/app/venv/bin/pip install -r requirements.txt

# 4. Configure application
sudo -u congracnet cp /home/congracnet/app/authSettings.py.example /home/congracnet/app/authSettings.py
# Edit configuration file with production settings

# 5. Set up service
sudo systemctl enable congracnet
sudo systemctl start congracnet

# 6. Configure firewall
sudo ufw allow 7475/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

#### 3. Post-deployment
- [ ] Service status verified
- [ ] Application accessible
- [ ] Database connectivity confirmed
- [ ] API functionality tested
- [ ] Performance metrics collected
- [ ] Monitoring alerts configured
- [ ] Documentation updated
- [ ] Team notified

### Monitoring and Maintenance

#### 1. Health Checks
```bash
# Check service status
sudo systemctl status congracnet

# Check application logs
sudo journalctl -u congracnet -f

# Check resource usage
htop
df -h
free -h

# Check network connectivity
netstat -tlnp | grep 7475
```

#### 2. Backup Procedures
```bash
# Database backup
sudo -u congracnet neo4j-admin dump --database=neo4j --to=/home/congracnet/backups/

# Application backup
sudo -u congracnet tar -czf /home/congracnet/backups/app_$(date +%Y%m%d).tar.gz /home/congracnet/app/

# Configuration backup
sudo -u congracnet cp /home/congracnet/app/authSettings.py /home/congracnet/backups/
```

## Uninstallation

### Complete Removal

#### 1. Stop Services
```bash
# Stop application service
sudo systemctl stop congracnet
sudo systemctl disable congracnet

# Remove service file
sudo rm /etc/systemd/system/congracnet.service
sudo systemctl daemon-reload
```

#### 2. Remove Application
```bash
# Remove application directory
sudo rm -rf /home/congracnet/app

# Remove user (optional)
sudo userdel -r congracnet

# Remove Nginx configuration
sudo rm /etc/nginx/sites-enabled/congracnet
sudo rm /etc/nginx/sites-available/congracnet
sudo nginx -t
sudo systemctl reload nginx
```

#### 3. Clean Up Dependencies
```bash
# Remove Python packages (if not needed elsewhere)
pip uninstall -r requirements.txt -y

# Remove virtual environment
rm -rf venv

# Remove configuration files
rm -f ~/.streamlit/config.toml
rm -f .env
```

### Partial Removal

#### 1. Remove Specific Components
```bash
# Remove only application files
rm -rf cgcnStream_0_3_6_withSBBLabel.py
rm -rf cgcn_functions_3_6.py
rm -rf sentiment_functions_3_6.py

# Keep virtual environment and dependencies
# Keep configuration files
```

#### 2. Reset Configuration
```bash
# Reset to default configuration
cp authSettings.py.example authSettings.py

# Clear application data
rm -rf logs/
rm -rf __pycache__/
```

---

*This installation guide provides comprehensive instructions for setting up ConGraCNet in various environments. For additional support, refer to the Troubleshooting section or contact the development team.*
