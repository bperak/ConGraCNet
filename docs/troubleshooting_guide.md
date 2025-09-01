# ConGraCNet Troubleshooting Guide

> **Complete guide to resolving common issues and problems**

## Table of Contents

1. [Quick Diagnosis](#quick-diagnosis)
2. [Common Error Messages](#common-error-messages)
3. [Installation Issues](#installation-issues)
4. [Database Connection Problems](#database-connection-problems)
5. [API and Network Issues](#api-and-network-issues)
6. [Application Performance Issues](#application-performance-issues)
7. [Visualization Problems](#visualization-problems)
8. [Data Processing Issues](#data-processing-issues)
9. [System Resource Issues](#system-resource-issues)
10. [Getting Help](#getting-help)

## Quick Diagnosis

### Problem Identification Checklist

Use this checklist to quickly identify the type of problem you're experiencing:

- [ ] **Application won't start**: Check installation and dependencies
- [ ] **Database connection failed**: Check Neo4j service and credentials
- [ ] **API errors**: Check internet connection and API credentials
- [ ] **Slow performance**: Check system resources and network size
- [ ] **Visualization issues**: Check browser compatibility and graphics drivers
- [ ] **Memory errors**: Check available RAM and application limits
- [ ] **Network construction fails**: Check corpus data and parameters

### Quick Fix Commands

```bash
# Check application status
streamlit --version
python --version
pip list | grep -E "(streamlit|pandas|networkx)"

# Check database connection
curl http://localhost:7474
telnet localhost 7474

# Check system resources
# Windows: Task Manager
# Linux: htop, free -h, df -h
# macOS: Activity Monitor
```

## Common Error Messages

### Python Import Errors

#### Error: `ModuleNotFoundError: No module named 'streamlit'`

**Cause**: Streamlit not installed or virtual environment not activated

**Solution**:
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# Install streamlit
pip install streamlit

# Verify installation
python -c "import streamlit; print('Streamlit installed')"
```

#### Error: `ModuleNotFoundError: No module named 'py2neo'`

**Cause**: Py2neo not installed

**Solution**:
```bash
# Install py2neo
pip install py2neo

# Verify installation
python -c "import py2neo; print('Py2neo installed')"
```

#### Error: `ModuleNotFoundError: No module named 'igraph'`

**Cause**: iGraph not installed

**Solution**:
```bash
# Install iGraph
pip install igraph

# If that fails, try:
pip install python-igraph

# Verify installation
python -c "import igraph; print('iGraph installed')"
```

### Database Connection Errors

#### Error: `Connection refused` or `Connection timeout`

**Cause**: Neo4j service not running or wrong port

**Solution**:
```bash
# Check if Neo4j is running
# Windows: Check Services app
# Linux: sudo systemctl status neo4j
# macOS: Check Activity Monitor

# Start Neo4j if stopped
# Windows: Start service from Services
# Linux: sudo systemctl start neo4j
# macOS: Start from Neo4j Desktop

# Check port availability
netstat -an | grep 7474
lsof -i :7474
```

#### Error: `Authentication failed` or `Invalid credentials`

**Cause**: Wrong username/password

**Solution**:
```bash
# Reset Neo4j password
sudo neo4j-admin set-initial-password new_password

# Update authSettings.py
# Edit the file and update:
# graphPass = "new_password"

# Test connection
python -c "
from py2neo import Graph
graph = Graph('http://localhost:7474', auth=('neo4j', 'new_password'))
print('Connection successful')
"
```

### API Connection Errors

#### Error: `Connection timeout` or `Network unreachable`

**Cause**: Internet connectivity issues or firewall blocking

**Solution**:
```bash
# Test internet connectivity
ping google.com
ping api.sketchengine.eu

# Test API endpoint
curl -I https://api.sketchengine.eu/health

# Check firewall settings
# Windows: Windows Firewall
# Linux: sudo ufw status
# macOS: System Preferences > Security & Privacy > Firewall

# If behind corporate proxy, set proxy environment variables
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

#### Error: `401 Unauthorized` or `403 Forbidden`

**Cause**: Invalid API credentials

**Solution**:
```bash
# Verify credentials in authSettings.py
grep -E "userName|apiKey" authSettings.py

# Test credentials manually
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.sketchengine.eu/endpoint

# Contact Sketch Engine support if credentials are invalid
```

## Installation Issues

### Virtual Environment Problems

#### Problem: Virtual environment not activating

**Symptoms**: `source venv/bin/activate` doesn't work, Python still points to system

**Solutions**:
```bash
# Check if venv exists
ls -la venv/

# Recreate virtual environment
rm -rf venv
python3.11 -m venv venv

# Activate with full path
source ./venv/bin/activate

# Verify activation
which python
echo $VIRTUAL_ENV
```

#### Problem: Wrong Python version in virtual environment

**Symptoms**: Virtual environment uses different Python version than expected

**Solutions**:
```bash
# Check Python versions
python --version
python3 --version
python3.11 --version

# Create venv with specific Python version
python3.11 -m venv venv311
source venv311/bin/activate

# Verify version
python --version
```

### Package Installation Failures

#### Problem: `Microsoft Visual C++ 14.0 is required` (Windows)

**Cause**: Missing Visual C++ build tools

**Solution**:
```bash
# Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or use Chocolatey
choco install visualstudio2019buildtools
choco install visualstudio2019-workload-vctools

# Restart terminal and try again
pip install -r requirements.txt
```

#### Problem: `fatal error: Python.h: No such file or directory` (Linux)

**Cause**: Missing Python development headers

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install python3.11-dev build-essential libssl-dev libffi-dev

# CentOS/RHEL
sudo yum install python3.11-devel gcc openssl-devel libffi-devel

# Try installation again
pip install -r requirements.txt
```

#### Problem: `Permission denied` during pip install

**Cause**: Installing packages globally instead of in virtual environment

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Verify you're using venv Python
which python  # Should show venv path

# If still having issues, use --user flag
pip install --user -r requirements.txt
```

### Dependency Conflicts

#### Problem: Package version conflicts

**Symptoms**: `ERROR: Cannot uninstall package` or version conflicts

**Solution**:
```bash
# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install with --force-reinstall
pip install --force-reinstall -r requirements.txt

# Or create fresh environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Database Connection Problems

### Neo4j Service Issues

#### Problem: Neo4j service won't start

**Symptoms**: Service fails to start, error messages in logs

**Solutions**:
```bash
# Check Neo4j logs
# Windows: Event Viewer
# Linux: sudo journalctl -u neo4j -f
# macOS: Check Neo4j Desktop logs

# Check system requirements
# Ensure sufficient memory (at least 2GB free)
# Check disk space (at least 1GB free)

# Reset Neo4j configuration
sudo neo4j stop
sudo rm -rf /var/lib/neo4j/data/databases/*
sudo neo4j start

# Check Java version (Neo4j 4.4+ requires Java 11+)
java -version
```

#### Problem: Neo4j port already in use

**Symptoms**: `Address already in use` error

**Solution**:
```bash
# Find process using port 7474
lsof -i :7474
netstat -tlnp | grep 7474

# Kill the process
sudo kill -9 <PID>

# Or change Neo4j port
# Edit neo4j.conf
# dbms.connector.http.listen_address=:7475

# Restart Neo4j
sudo systemctl restart neo4j
```

### Connection Configuration Issues

#### Problem: Wrong database URL format

**Symptoms**: Connection errors with specific URL patterns

**Solution**:
```python
# Correct URL formats for authSettings.py
# Local Neo4j
graphURL = "http://localhost:7474"

# Remote Neo4j with authentication
graphURL = "http://username:password@server:7474"

# Neo4j with custom protocol
graphURL = "bolt://localhost:7687"  # For Bolt protocol
```

#### Problem: Firewall blocking connection

**Symptoms**: Connection timeout from external machines

**Solution**:
```bash
# Check firewall status
sudo ufw status

# Allow Neo4j ports
sudo ufw allow 7474/tcp
sudo ufw allow 7687/tcp

# For Windows Firewall, add rules for ports 7474 and 7687
# Control Panel > System and Security > Windows Firewall > Advanced Settings
```

### Database Performance Issues

#### Problem: Slow database queries

**Symptoms**: Queries take long time to execute

**Solution**:
```bash
# Check Neo4j memory settings
# Edit neo4j.conf
# dbms.memory.heap.initial_size=1G
# dbms.memory.heap.max_size=2G

# Create database indexes
# Run in Neo4j browser:
CREATE INDEX ON :Lemma(lempos);
CREATE INDEX ON :Lemma(language);
CREATE INDEX ON :GramRel(type);

# Restart Neo4j after configuration changes
sudo systemctl restart neo4j
```

## API and Network Issues

### Sketch Engine API Problems

#### Problem: API rate limiting

**Symptoms**: `429 Too Many Requests` errors

**Solution**:
```python
# Implement exponential backoff
import time
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def api_call_with_retry(endpoint, **kwargs):
    """Make API call with retry logic."""
    response = requests.get(endpoint, **kwargs)
    response.raise_for_status()
    return response

# Use in your code
try:
    result = api_call_with_retry("https://api.sketchengine.eu/endpoint")
except Exception as e:
    print(f"API call failed after retries: {e}")
```

#### Problem: API authentication token expired

**Symptoms**: `401 Unauthorized` errors after working initially

**Solution**:
```bash
# Check token expiration
# Contact Sketch Engine support for new token

# Update authSettings.py with new token
# apiKey = "new_token_here"

# Test new token
curl -H "Authorization: Bearer new_token" \
     https://api.sketchengine.eu/endpoint
```

### Network Connectivity Issues

#### Problem: Corporate firewall blocking connections

**Symptoms**: Can't connect to external APIs from corporate network

**Solution**:
```bash
# Set proxy environment variables
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# Or configure pip to use proxy
pip install --proxy http://proxy.company.com:8080 package_name

# Contact IT department to whitelist required domains
# api.sketchengine.eu
# pypi.org
# files.pythonhosted.org
```

#### Problem: DNS resolution issues

**Symptoms**: Can't resolve hostnames

**Solution**:
```bash
# Test DNS resolution
nslookup api.sketchengine.eu
dig api.sketchengine.eu

# Try alternative DNS servers
# Google: 8.8.8.8, 8.8.4.4
# Cloudflare: 1.1.1.1, 1.0.0.1

# Add to /etc/resolv.conf (Linux/macOS)
nameserver 8.8.8.8
nameserver 8.8.4.4
```

## Application Performance Issues

### Slow Application Startup

#### Problem: Application takes long time to start

**Symptoms**: Long delay before Streamlit interface appears

**Solutions**:
```bash
# Check system resources
# Windows: Task Manager
# Linux: htop
# macOS: Activity Monitor

# Optimize virtual environment
pip install --upgrade pip setuptools wheel

# Use faster package manager
pip install --upgrade pip-tools

# Check for large files in project directory
du -sh *
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete
```

#### Problem: Slow database initialization

**Symptoms**: Long delay during database connection

**Solution**:
```python
# Implement connection pooling
from py2neo import Graph
import threading

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'graph'):
            self.graph = Graph(authSettings.graphURL, 
                             auth=(authSettings.graphUser, authSettings.graphPass))

# Use singleton pattern
db_manager = DatabaseManager()
```

### Memory Issues

#### Problem: Application runs out of memory

**Symptoms**: `MemoryError` or application crashes

**Solutions**:
```python
# Implement memory-efficient processing
def process_large_network_in_chunks(network, chunk_size=1000):
    """Process large networks in chunks to manage memory."""
    results = []
    nodes = list(network.nodes())
    
    for i in range(0, len(nodes), chunk_size):
        chunk = nodes[i:i + chunk_size]
        subgraph = network.subgraph(chunk)
        
        # Process chunk
        chunk_result = process_network_chunk(subgraph)
        results.extend(chunk_result)
        
        # Clear chunk from memory
        del subgraph
        gc.collect()
    
    return results

# Use in your code
if len(network.nodes) > 10000:
    results = process_large_network_in_chunks(network)
else:
    results = process_network(network)
```

#### Problem: Large network visualization consumes too much memory

**Symptoms**: Browser becomes unresponsive or crashes

**Solution**:
```python
# Implement network pruning
def prune_network_for_visualization(network, max_nodes=1000):
    """Prune network to manageable size for visualization."""
    if len(network.nodes) <= max_nodes:
        return network
    
    # Keep nodes with highest degree
    node_degrees = dict(network.degree())
    top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
    top_node_list = [node for node, _ in top_nodes]
    
    # Create subgraph
    pruned_network = network.subgraph(top_node_list)
    return pruned_network

# Use in visualization
visualization_network = prune_network_for_visualization(network, max_nodes=500)
```

### Network Construction Performance

#### Problem: Network construction takes too long

**Symptoms**: Long delays when building networks

**Solutions**:
```python
# Implement caching for repeated operations
from functools import lru_cache
import hashlib
import pickle

@lru_cache(maxsize=100)
def cached_network_construction(lemma, pos, corpus, gram_rel):
    """Cache network construction results."""
    return construct_network(lemma, pos, corpus, gram_rel)

# Use in your code
network = cached_network_construction(lemma, pos, corpus, gram_rel)
```

```python
# Implement progress indicators
import streamlit as st
from tqdm import tqdm

def construct_network_with_progress(lemma, pos, corpus, gram_rel):
    """Construct network with progress bar."""
    with st.spinner("Constructing network..."):
        progress_bar = st.progress(0)
        
        # Simulate progress steps
        steps = ["Initializing", "Fetching data", "Building graph", "Finalizing"]
        for i, step in enumerate(steps):
            st.text(f"Step {i+1}: {step}")
            progress_bar.progress((i + 1) / len(steps))
            time.sleep(0.5)  # Simulate work
        
        # Actual network construction
        network = construct_network(lemma, pos, corpus, gram_rel)
        
        progress_bar.progress(1.0)
        st.success("Network construction completed!")
        return network
```

## Visualization Problems

### Browser Compatibility Issues

#### Problem: Network visualization doesn't display

**Symptoms**: Blank area where visualization should appear

**Solutions**:
```bash
# Check browser console for JavaScript errors
# Press F12 to open Developer Tools

# Ensure browser supports WebGL
# Visit: https://get.webgl.org/

# Try different browsers
# Chrome/Edge: Best compatibility
# Firefox: Good compatibility
# Safari: May have issues with some features

# Check if JavaScript is enabled
# Ensure no ad blockers are interfering
```

#### Problem: 3D visualizations don't work

**Symptoms**: 3D network view shows error or doesn't render

**Solution**:
```python
# Check WebGL support
def check_webgl_support():
    """Check if browser supports WebGL."""
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=[1], y=[1], z=[1]))
        return True
    except Exception as e:
        st.warning("3D visualization not supported: WebGL required")
        return False

# Use in your code
if check_webgl_support():
    show_3d_visualization(network)
else:
    show_2d_visualization(network)
```

### Plotly Rendering Issues

#### Problem: Plotly charts don't display correctly

**Symptoms**: Charts appear but are malformed or incomplete

**Solutions**:
```python
# Update Plotly to latest version
pip install --upgrade plotly

# Check Plotly configuration
import plotly.io as pio
pio.renderers.default = "browser"  # or "notebook" for Jupyter

# Use explicit figure sizing
fig = go.Figure()
fig.update_layout(
    width=800,
    height=600,
    autosize=False
)

# Ensure proper data types
# Convert numpy arrays to lists if needed
data = [float(x) for x in numpy_array]
```

#### Problem: Interactive features not working

**Symptoms**: Can't zoom, pan, or interact with visualizations

**Solution**:
```python
# Enable interactive features
fig = go.Figure()
fig.update_layout(
    dragmode='pan',  # Enable panning
    hovermode='closest'  # Enable hovering
)

# Add interactive elements
fig.add_annotation(
    x=0.5,
    y=0.5,
    text="Click and drag to pan",
    showarrow=False
)

# Use FigureWidget for enhanced interactivity
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=1)
```

## Data Processing Issues

### Corpus Data Problems

#### Problem: Corpus not found or empty

**Symptoms**: Error message "Corpus not found" or empty results

**Solutions**:
```python
# Check corpus availability
def check_corpus_availability(corpus_id):
    """Check if corpus is available and accessible."""
    try:
        # Test corpus access
        corpus_info = get_corpus_info(corpus_id)
        if corpus_info and len(corpus_info) > 0:
            return True
        else:
            st.error(f"Corpus {corpus_id} is empty or not accessible")
            return False
    except Exception as e:
        st.error(f"Error accessing corpus {corpus_id}: {e}")
        return False

# Use in your code
if check_corpus_availability(corpus_id):
    # Proceed with corpus analysis
    pass
else:
    st.stop()
```

#### Problem: Lemma not found in corpus

**Symptoms**: "Lemma not found" error

**Solution**:
```python
# Implement lemma validation
def validate_lemma(lemma, pos, corpus_id):
    """Validate if lemma exists in corpus."""
    try:
        # Check if lemma exists
        result = source_lemma_freq(lemma, pos, corpus_id, corpus_id, "hr", "obj")
        if result.empty or result.iloc[0]['freq'] == 0:
            return False, "Lemma not found in corpus"
        return True, "Lemma found"
    except Exception as e:
        return False, f"Error validating lemma: {e}"

# Use in your code
is_valid, message = validate_lemma(lemma, pos, corpus_id)
if not is_valid:
    st.error(message)
    st.stop()
```

### Network Data Issues

#### Problem: Network construction returns empty results

**Symptoms**: Network has no nodes or edges

**Solutions**:
```python
# Check network construction parameters
def debug_network_construction(lemma, pos, corpus_id, gram_rel):
    """Debug network construction process."""
    st.write("Debugging network construction...")
    
    # Check lemma frequency
    freq_data = source_lemma_freq(lemma, pos, corpus_id, corpus_id, "hr", gram_rel)
    st.write(f"Lemma frequency: {freq_data}")
    
    # Check grammatical relations
    gram_rels = lemmaGramRels(lemma, pos, corpus_id, "hr")
    st.write(f"Available grammatical relations: {gram_rels}")
    
    # Check network data
    network_data = lemmaByGramRel("hr", lemma, pos, gram_rel, corpus_id, "score")
    st.write(f"Network data: {network_data}")
    
    return network_data

# Use in your code
if st.checkbox("Debug network construction"):
    network_data = debug_network_construction(lemma, pos, corpus_id, gram_rel)
```

#### Problem: Network data is corrupted or incomplete

**Symptoms**: Network has missing nodes or incorrect relationships

**Solution**:
```python
# Implement data validation
def validate_network_data(network_data):
    """Validate network data integrity."""
    errors = []
    
    # Check required columns
    required_columns = ['source', 'target', 'gramRel', 'count', 'score']
    for col in required_columns:
        if col not in network_data.columns:
            errors.append(f"Missing required column: {col}")
    
    # Check for empty values
    for col in network_data.columns:
        if network_data[col].isnull().any():
            errors.append(f"Column {col} contains null values")
    
    # Check data types
    if not network_data['count'].dtype in ['int64', 'float64']:
        errors.append("Count column should be numeric")
    
    if not network_data['score'].dtype in ['int64', 'float64']:
        errors.append("Score column should be numeric")
    
    return len(errors) == 0, errors

# Use in your code
is_valid, errors = validate_network_data(network_data)
if not is_valid:
    st.error("Network data validation failed:")
    for error in errors:
        st.error(f"  - {error}")
    st.stop()
```

## System Resource Issues

### Memory Management

#### Problem: Insufficient memory for large networks

**Symptoms**: `MemoryError` or system becomes unresponsive

**Solutions**:
```python
# Implement memory monitoring
import psutil
import gc

def check_memory_usage():
    """Check current memory usage."""
    memory = psutil.virtual_memory()
    st.write(f"Memory usage: {memory.percent}%")
    st.write(f"Available memory: {memory.available / (1024**3):.1f} GB")
    
    if memory.percent > 80:
        st.warning("High memory usage detected!")
        return False
    return True

# Use in your code
if not check_memory_usage():
    st.error("Insufficient memory for operation")
    st.stop()

# Force garbage collection
gc.collect()
```

#### Problem: Memory leaks during long sessions

**Symptoms**: Memory usage increases over time

**Solution**:
```python
# Implement memory cleanup
def cleanup_memory():
    """Clean up memory and clear caches."""
    import gc
    
    # Clear Streamlit cache
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # Force garbage collection
    gc.collect()
    
    # Clear large variables
    if 'large_network' in globals():
        del globals()['large_network']
    
    st.success("Memory cleanup completed")

# Add cleanup button
if st.button("Clean up memory"):
    cleanup_memory()
```

### CPU Performance

#### Problem: High CPU usage during network analysis

**Symptoms**: System becomes slow, high CPU utilization

**Solutions**:
```python
# Implement progress monitoring
import time
import psutil

def monitor_performance():
    """Monitor system performance during operations."""
    start_time = time.time()
    start_cpu = psutil.cpu_percent()
    
    # Your operation here
    # ...
    
    end_time = time.time()
    end_cpu = psutil.cpu_percent()
    
    st.write(f"Operation completed in {end_time - start_time:.2f} seconds")
    st.write(f"CPU usage: {end_cpu:.1f}%")

# Use multiprocessing for CPU-intensive tasks
from multiprocessing import Pool

def parallel_network_analysis(networks):
    """Run network analysis in parallel."""
    with Pool() as pool:
        results = pool.map(analyze_network, networks)
    return results
```

### Disk Space Issues

#### Problem: Insufficient disk space

**Symptoms**: `OSError: [Errno 28] No space left on device`

**Solution**:
```bash
# Check disk space
df -h

# Clean up temporary files
rm -rf /tmp/*
rm -rf ~/.cache/*
rm -rf __pycache__/
rm -rf *.pyc

# Clean up old log files
find . -name "*.log" -mtime +7 -delete

# Clean up old data files
find . -name "*.pkl" -mtime +30 -delete
```

## Getting Help

### Self-Diagnosis Tools

#### 1. Application Health Check
```python
def run_health_check():
    """Run comprehensive health check."""
    st.header("System Health Check")
    
    # Check Python environment
    st.subheader("Python Environment")
    st.write(f"Python version: {sys.version}")
    st.write(f"Virtual environment: {os.environ.get('VIRTUAL_ENV', 'Not activated')}")
    
    # Check dependencies
    st.subheader("Dependencies")
    try:
        import streamlit as st
        st.write("✅ Streamlit")
    except ImportError:
        st.write("❌ Streamlit")
    
    try:
        import pandas as pd
        st.write("✅ Pandas")
    except ImportError:
        st.write("❌ Pandas")
    
    # Check database connection
    st.subheader("Database Connection")
    try:
        graph = init_neo4j()
        if graph:
            st.write("✅ Neo4j connection")
        else:
            st.write("❌ Neo4j connection failed")
    except Exception as e:
        st.write(f"❌ Neo4j error: {e}")
    
    # Check system resources
    st.subheader("System Resources")
    memory = psutil.virtual_memory()
    st.write(f"Memory: {memory.percent}% used")
    st.write(f"Available: {memory.available / (1024**3):.1f} GB")

# Add health check button
if st.sidebar.button("Run Health Check"):
    run_health_check()
```

#### 2. Error Log Collection
```python
def collect_error_info():
    """Collect system information for error reporting."""
    import platform
    import sys
    
    error_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "streamlit_version": st.__version__,
        "memory_usage": psutil.virtual_memory()._asdict(),
        "disk_usage": psutil.disk_usage('/')._asdict(),
        "cpu_count": psutil.cpu_count(),
        "current_directory": os.getcwd(),
        "environment_variables": dict(os.environ)
    }
    
    st.json(error_info)
    
    # Export to file
    import json
    with open("error_report.json", "w") as f:
        json.dump(error_info, f, indent=2)
    
    st.success("Error report saved to error_report.json")

# Add error collection button
if st.sidebar.button("Collect Error Info"):
    collect_error_info()
```

### Support Resources

#### 1. Documentation
- **User Manual**: `docs/user_manual.md`
- **Developer Guide**: `docs/developer_guide.md`
- **API Reference**: `docs/api_reference.md`
- **Installation Guide**: `docs/installation_guide.md`

#### 2. Community Support
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share solutions
- **Wiki**: Community-maintained documentation

#### 3. Professional Support
- **Email Support**: For critical issues
- **Video Calls**: For complex problems
- **Training Sessions**: For organizations

### Reporting Issues

#### Issue Report Template
```markdown
## Issue Description
Brief description of the problem

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Error Messages
Copy any error messages or stack traces

## System Information
- Operating System:
- Python Version:
- ConGraCNet Version:
- Browser (if applicable):

## Additional Context
Any other information that might be helpful

## Attachments
- Screenshots
- Error logs
- Configuration files
```

#### Before Reporting
1. **Check existing issues** on GitHub
2. **Search documentation** for solutions
3. **Run health check** to gather system info
4. **Try minimal reproduction** case
5. **Include all relevant details** in report

---

*This troubleshooting guide provides solutions to common ConGraCNet issues. If your problem isn't covered here, please check the documentation or contact support with detailed information about your issue.*
