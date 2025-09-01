# ConGraCNet Developer Guide

> **Complete guide for developers contributing to and extending ConGraCNet**

## Table of Contents

1. [Development Environment](#development-environment)
2. [Project Structure](#project-structure)
3. [Architecture Overview](#architecture-overview)
4. [Core Components](#core-components)
5. [Database Schema](#database-schema)
6. [API Integration](#api-integration)
7. [Development Workflow](#development-workflow)
8. [Testing Strategy](#testing-strategy)
9. [Performance Optimization](#performance-optimization)
10. [Deployment](#deployment)
11. [Contributing Guidelines](#contributing-guidelines)

## Development Environment

### Prerequisites

#### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.9 or higher
- **Memory**: Minimum 8GB RAM, recommended 16GB+
- **Storage**: Minimum 10GB free space
- **Network**: Internet access for API calls and package installation

#### Development Tools
- **Code Editor**: VS Code, PyCharm, or Vim/Emacs
- **Version Control**: Git 2.20+
- **Package Manager**: pip 20.0+
- **Virtual Environment**: venv or conda
- **Database**: Neo4j 4.4+ (local or remote)

### Environment Setup

#### 1. Clone Repository
```bash
git clone https://github.com/bperak/ConGraCNet.git
cd ConGraCNet
```

#### 2. Create Virtual Environment
```bash
# Using venv (recommended)
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# Or using conda
conda create --name congracnet python=3.11
conda activate congracnet
```

#### 3. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### 4. Configure Development Tools
```bash
# Install development tools
pip install black isort mypy pytest pytest-cov

# Configure pre-commit
pre-commit install --hook-type pre-commit
pre-commit install --hook-type pre-push
```

### IDE Configuration

#### VS Code Settings
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm Configuration
- Set project interpreter to virtual environment
- Enable auto-import optimization
- Configure code style to PEP 8
- Enable type checking

## Project Structure

### Directory Layout
```
congracnet/
├── cgcnStream_0_3_6_withSBBLabel.py  # Main Streamlit application
├── cgcn_functions_3_6.py             # Core functionality module
├── sentiment_functions_3_6.py        # Sentiment analysis module
├── wordnet_functions.py              # WordNet integration
├── spacy_wordnet_functions.py        # spaCy-WordNet bridge
├── sketch2Neo2.py                    # Corpus management
├── authSettings.py                   # Configuration and credentials
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development dependencies
├── setup.py                         # Package configuration
├── README.md                         # Project overview
├── PLANNING.md                       # Architecture and planning
├── TASK.md                          # Task tracking
├── docs/                            # Documentation
│   ├── user_manual.md               # User documentation
│   ├── developer_guide.md            # This file
│   ├── api_reference.md             # API documentation
│   └── tutorials/                   # Step-by-step guides
├── tests/                           # Test suite
│   ├── test_core_functions.py       # Core function tests
│   ├── test_sentiment_functions.py  # Sentiment analysis tests
│   ├── test_network_functions.py    # Network construction tests
│   └── conftest.py                  # Test configuration
├── scripts/                         # Utility scripts
│   ├── setup_dev.py                 # Development setup
│   ├── run_tests.py                 # Test runner
│   └── deploy.py                    # Deployment script
└── venv/                            # Virtual environment
```

### File Organization Principles

#### 1. Single Responsibility
- Each file should have one primary purpose
- Functions should be grouped by functionality
- Avoid mixing UI, business logic, and data access

#### 2. Maximum File Size
- **Enforced Limit**: 500 lines per file
- **Target**: 300-400 lines for optimal maintainability
- **Refactoring**: Split large files into focused modules

#### 3. Import Organization
```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import pandas as pd
import numpy as np
import streamlit as st

# Local imports
from .core import network_functions
from .utils import helpers
```

## Architecture Overview

### System Architecture

#### 1. Layered Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                    │
│                 (Streamlit Interface)                   │
├─────────────────────────────────────────────────────────┤
│                    Business Logic Layer                 │
│              (Core Functions & Analysis)               │
├─────────────────────────────────────────────────────────┤
│                     Data Access Layer                   │
│                (Database & API Calls)                  │
├─────────────────────────────────────────────────────────┤
│                    External Services                    │
│              (Neo4j, Sketch Engine API)                │
└─────────────────────────────────────────────────────────┘
```

#### 2. Component Interaction
```
Streamlit App → Core Functions → Database Functions → Neo4j
     ↓              ↓              ↓              ↓
UI Components → Business Logic → Data Access → External APIs
```

### Design Patterns

#### 1. Factory Pattern
- **Purpose**: Create different types of network objects
- **Implementation**: Network factory for different corpus types
- **Benefits**: Easy extension for new corpus types

#### 2. Strategy Pattern
- **Purpose**: Interchangeable algorithms for analysis
- **Implementation**: Different centrality measures and clustering algorithms
- **Benefits**: Easy addition of new analysis methods

#### 3. Observer Pattern
- **Purpose**: Update UI when data changes
- **Implementation**: Streamlit state management
- **Benefits**: Responsive user interface

## Core Components

### Main Application (`cgcnStream_0_3_6_withSBBLabel.py`)

#### Structure
```python
# Imports and configuration
# Global variables and initialization
# Main application logic
# UI components and layout
```

#### Key Functions
- **`init_neo4j()`**: Database connection initialization
- **`get_state()`**: Persistent state management
- **`col_order()`**: DataFrame column organization

#### Refactoring Opportunities
- **UI Components**: Extract to separate modules
- **Business Logic**: Move to core functions
- **Configuration**: Externalize to config files

### Core Functions (`cgcn_functions_3_6.py`)

#### Database Operations
```python
def source_lemma_freq(lemma: str, pos: str, corpusID: str, 
                     corpus: str, language: str, gramRel: str) -> pd.DataFrame:
    """
    Fetch the frequency of the source lemma.
    
    Args:
        lemma: Base form of the word
        pos: Part of speech
        corpusID: Corpus identifier
        corpus: Corpus name
        language: Language code (hr/en)
        gramRel: Grammatical relation type
        
    Returns:
        DataFrame with frequency and relative frequency
        
    Raises:
        ConnectionError: If database connection fails
        ValueError: If parameters are invalid
    """
```

#### Network Construction
```python
def lemmaByGramRel(language: str, lemma: str, pos: str, 
                   gramRel: str, corpusID: str, measure: str) -> pd.DataFrame:
    """
    Build network based on grammatical relations.
    
    Args:
        language: Language code
        lemma: Source lemma
        pos: Part of speech
        gramRel: Grammatical relation
        corpusID: Corpus identifier
        measure: Analysis measure (score/freq)
        
    Returns:
        DataFrame with network nodes and edges
    """
```

### Sentiment Functions (`sentiment_functions_3_6.py`)

#### Dictionary Integration
```python
class SentimentAnalyzer:
    """Manages multiple sentiment dictionaries and calculations."""
    
    def __init__(self, language: str):
        self.language = language
        self.dictionaries = self._load_dictionaries()
    
    def calculate_network_sentiment(self, network: nx.Graph, 
                                  method: str = 'propagation') -> Dict[str, float]:
        """Calculate sentiment values for network nodes."""
```

#### Sentiment Propagation
```python
def propagate_sentiment(network: nx.Graph, initial_sentiment: Dict[str, float],
                       method: str = 'linear', decay: float = 0.5) -> Dict[str, float]:
    """
    Propagate sentiment through network structure.
    
    Args:
        network: NetworkX graph object
        initial_sentiment: Initial sentiment values
        method: Propagation method (linear/exponential)
        decay: Sentiment decay factor
        
    Returns:
        Updated sentiment values for all nodes
    """
```

### WordNet Integration (`wordnet_functions.py`)

#### Synset Management
```python
def get_hypernyms(lempos_list: List[str], language: str) -> pd.DataFrame:
    """
    Extract hypernym relationships for given lemmas.
    
    Args:
        lempos_list: List of lemma-pos combinations
        language: Language code
        
    Returns:
        DataFrame with source-target hypernym pairs
    """
```

#### Domain Classification
```python
def get_domains_for_word(word: str) -> List[str]:
    """
    Get semantic domains for a given word.
    
    Args:
        word: Target word
        
    Returns:
        List of semantic domain categories
    """
```

## Database Schema

### Neo4j Graph Structure

#### Node Labels
```cypher
// Lemma nodes
(:Lemma {
    lempos: String,           // lemma-pos combination
    language: String,         // language code
    freq_hrWaC22: Integer,   // frequency in specific corpus
    relFreq_hrWaC22: Float   // relative frequency
})

// Grammatical relation nodes
(:GramRel {
    type: String,             // relation type (obj, subj, mod)
    corpus: String,          // corpus identifier
    language: String         // language code
})
```

#### Relationship Types
```cypher
// Co-occurrence relationships
(:Lemma)-[:COOCCURS_WITH {
    count: Integer,          // co-occurrence count
    score: Float,            // association score
    corpus: String,          // corpus identifier
    gramRel: String          // grammatical relation
}]->(:Lemma)

// Grammatical relationships
(:Lemma)-[:HAS_GRAMREL {
    relation: String,        // relation type
    frequency: Integer,      // relation frequency
    corpus: String          // corpus identifier
}]->(:GramRel)
```

### Database Operations

#### Connection Management
```python
class DatabaseManager:
    """Manages Neo4j database connections and operations."""
    
    def __init__(self, url: str, user: str, password: str):
        self.graph = Graph(url, auth=(user, password))
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX ON :Lemma(lempos)",
            "CREATE INDEX ON :Lemma(language)",
            "CREATE INDEX ON :GramRel(type)"
        ]
        for index in indexes:
            try:
                self.graph.run(index)
            except Exception as e:
                print(f"Index creation failed: {e}")
```

#### Query Optimization
```python
def optimized_lemma_query(lemma: str, pos: str, corpusID: str) -> pd.DataFrame:
    """
    Optimized query for lemma information.
    
    Uses parameterized queries and proper indexing.
    """
    query = """
    MATCH (n:Lemma {lempos: $lempos, language: $language})
    WHERE n[$freq] IS NOT NULL
    RETURN n.lempos as lempos, n[$freq] as freq, n[$relFreq] as relFreq
    """
    
    params = {
        'lempos': f"{lemma}{pos}",
        'language': 'hr' if 'hr' in corpusID else 'en',
        'freq': f'freq_{corpusID}',
        'relFreq': f'relFreq_{corpusID}'
    }
    
    return self.graph.run(query, **params).to_data_frame()
```

## API Integration

### Sketch Engine API

#### Authentication
```python
class SketchEngineAPI:
    """Manages Sketch Engine API interactions."""
    
    def __init__(self, username: str, api_key: str):
        self.username = username
        self.api_key = api_key
        self.base_url = "https://api.sketchengine.eu/bonito/run.cgi"
    
    def _authenticate(self) -> Dict[str, str]:
        """Generate authentication headers."""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
```

#### Corpus Access
```python
def get_corpus_info(self, corpus_id: str) -> Dict[str, Any]:
    """
    Retrieve corpus information and metadata.
    
    Args:
        corpus_id: Corpus identifier
        
    Returns:
        Dictionary with corpus information
    """
    endpoint = f"{self.base_url}/corpus_info"
    params = {'corpus': corpus_id}
    
    response = requests.get(
        endpoint, 
        params=params, 
        headers=self._authenticate()
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise APIError(f"Failed to get corpus info: {response.status_code}")
```

#### Data Harvesting
```python
def harvest_word_sketch(self, corpus: str, words: List[str], 
                       pos: str, limit: int) -> pd.DataFrame:
    """
    Harvest word sketch data from Sketch Engine.
    
    Args:
        corpus: Target corpus
        words: List of words to analyze
        pos: Part of speech
        limit: Maximum number of results
        
    Returns:
        DataFrame with harvested data
    """
    endpoint = f"{self.base_url}/wsketch"
    
    data = {
        'corpus': corpus,
        'lemma': words,
        'pos': pos,
        'limit': limit
    }
    
    response = requests.post(
        endpoint,
        json=data,
        headers=self._authenticate()
    )
    
    return self._parse_sketch_response(response.json())
```

### Error Handling

#### API Error Classes
```python
class APIError(Exception):
    """Base class for API-related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass

class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass
```

#### Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def api_call_with_retry(self, endpoint: str, **kwargs) -> requests.Response:
    """
    Make API call with exponential backoff retry.
    
    Args:
        endpoint: API endpoint
        **kwargs: Request parameters
        
    Returns:
        API response
        
    Raises:
        APIError: If all retry attempts fail
    """
    try:
        response = requests.get(endpoint, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        raise APIError(f"API call failed: {str(e)}")
```

## Development Workflow

### Git Workflow

#### Branch Strategy
```bash
# Main branches
main                    # Production-ready code
develop                 # Integration branch
feature/*              # Feature development
bugfix/*               # Bug fixes
hotfix/*               # Critical production fixes
```

#### Commit Guidelines
```bash
# Commit message format
<type>(<scope>): <description>

# Examples
feat(network): add community detection algorithms
fix(sentiment): resolve memory leak in propagation
docs(api): update function documentation
refactor(core): split large functions into modules
test(network): add comprehensive test coverage
```

#### Pull Request Process
1. **Create Feature Branch**: `git checkout -b feature/new-feature`
2. **Develop Feature**: Implement with tests and documentation
3. **Run Tests**: Ensure all tests pass
4. **Create PR**: Submit pull request to develop branch
5. **Code Review**: Address review comments
6. **Merge**: Merge after approval

### Code Quality

#### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=120]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

#### Code Standards
- **Line Length**: Maximum 120 characters
- **Imports**: Organized and sorted
- **Type Hints**: Required for all functions
- **Docstrings**: Google style for all functions
- **Error Handling**: Comprehensive exception management

### Development Commands

#### Setup Commands
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run all quality checks
pre-commit run --all-files
```

#### Testing Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cgcn_functions --cov-report=html

# Run specific test file
pytest tests/test_core_functions.py

# Run tests in parallel
pytest -n auto
```

#### Quality Commands
```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy .

# Linting
flake8 .
```

## Testing Strategy

### Test Organization

#### Test Structure
```
tests/
├── conftest.py                    # Test configuration and fixtures
├── test_core_functions.py         # Core functionality tests
├── test_sentiment_functions.py    # Sentiment analysis tests
├── test_network_functions.py      # Network construction tests
├── test_ui_components.py          # Streamlit component tests
├── test_integration.py            # End-to-end tests
├── test_performance.py            # Performance tests
└── fixtures/                      # Test data and fixtures
    ├── sample_networks/           # Sample network data
    ├── mock_responses/            # Mock API responses
    └── test_corpora/              # Test corpus data
```

#### Test Categories

##### 1. Unit Tests
```python
def test_source_lemma_freq_success():
    """Test successful frequency retrieval."""
    # Arrange
    lemma = "kuca"
    pos = "N"
    corpusID = "hrWaC22"
    
    # Act
    result = source_lemma_freq(lemma, pos, corpusID, "hrWaC22", "hr", "obj")
    
    # Assert
    assert not result.empty
    assert 'freq' in result.columns
    assert 'relFreq' in result.columns
    assert result.iloc[0]['freq'] > 0
```

##### 2. Integration Tests
```python
def test_network_construction_integration():
    """Test complete network construction workflow."""
    # Arrange
    test_lemma = "kuca"
    test_pos = "N"
    
    # Act
    network = construct_network(test_lemma, test_pos, "hrWaC22")
    
    # Assert
    assert network is not None
    assert len(network.nodes) > 0
    assert len(network.edges) > 0
    assert test_lemma in network.nodes
```

##### 3. Performance Tests
```python
def test_large_network_performance():
    """Test performance with large networks."""
    # Arrange
    large_network = create_large_test_network(1000, 5000)
    
    # Act
    start_time = time.time()
    communities = detect_communities(large_network, algorithm='louvain')
    end_time = time.time()
    
    # Assert
    assert end_time - start_time < 5.0  # Should complete within 5 seconds
    assert len(communities) > 0
```

### Test Configuration

#### Pytest Configuration
```ini
# pytest.ini
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
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
```

#### Test Fixtures
```python
# conftest.py
import pytest
import pandas as pd
import networkx as nx
from unittest.mock import Mock

@pytest.fixture
def sample_network():
    """Create a sample network for testing."""
    G = nx.Graph()
    G.add_nodes_from(['node1', 'node2', 'node3'])
    G.add_edges_from([('node1', 'node2'), ('node2', 'node3')])
    return G

@pytest.fixture
def mock_neo4j_connection():
    """Mock Neo4j database connection."""
    mock_graph = Mock()
    mock_graph.run.return_value.to_data_frame.return_value = pd.DataFrame()
    return mock_graph

@pytest.fixture
def sample_corpus_data():
    """Sample corpus data for testing."""
    return {
        'corpus': 'hrWaC22',
        'language': 'hr',
        'corpusID': 'hrWaC22',
        'gramRel': 'obj',
        'initial_lexeme': 'kuca',
        'initial_pos': 'N'
    }
```

### Mocking and Stubbing

#### API Mocking
```python
@pytest.fixture
def mock_sketch_engine_api():
    """Mock Sketch Engine API responses."""
    with patch('sketch2Neo2.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'corpus': 'hrWaC22',
            'size': 1000000,
            'language': 'hr'
        }
        mock_get.return_value = mock_response
        yield mock_get
```

#### Database Mocking
```python
def test_database_query_with_mock():
    """Test database query using mocked connection."""
    # Arrange
    mock_graph = Mock()
    mock_result = Mock()
    mock_result.to_data_frame.return_value = pd.DataFrame({
        'lempos': ['kuca-N'],
        'freq': [100],
        'relFreq': [10.5]
    })
    mock_graph.run.return_value = mock_result
    
    # Act
    with patch('cgcn_functions.graph', mock_graph):
        result = source_lemma_freq('kuca', 'N', 'hrWaC22', 'hrWaC22', 'hr', 'obj')
    
    # Assert
    assert not result.empty
    assert result.iloc[0]['freq'] == 100
```

## Performance Optimization

### Memory Management

#### Large Network Handling
```python
def process_large_network(network: nx.Graph, chunk_size: int = 1000) -> List[Dict]:
    """
    Process large networks in chunks to manage memory.
    
    Args:
        network: NetworkX graph object
        chunk_size: Number of nodes to process at once
        
    Returns:
        List of processed results
    """
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
```

#### Data Structure Optimization
```python
def optimize_network_for_analysis(network: nx.Graph) -> nx.Graph:
    """
    Optimize network data structures for analysis.
    
    Args:
        network: Input network
        
    Returns:
        Optimized network
    """
    # Convert to undirected if possible
    if network.is_directed():
        network = network.to_undirected()
    
    # Remove self-loops
    network.remove_edges_from(nx.selfloop_edges(network))
    
    # Remove isolated nodes
    network.remove_nodes_from(list(nx.isolates(network)))
    
    # Convert to adjacency matrix for dense operations
    if len(network) < 1000:
        network.adj_matrix = nx.adjacency_matrix(network)
    
    return network
```

### Caching Strategies

#### Function Result Caching
```python
from functools import lru_cache
import hashlib
import pickle

def cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    # Create deterministic string representation
    key_data = (args, sorted(kwargs.items()))
    key_string = pickle.dumps(key_data)
    return hashlib.md5(key_string).hexdigest()

def cached_network_analysis(func):
    """Decorator for caching network analysis results."""
    cache = {}
    
    def wrapper(*args, **kwargs):
        key = cache_key(*args, **kwargs)
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            
            # Limit cache size
            if len(cache) > 100:
                # Remove oldest entries
                oldest_key = next(iter(cache))
                del cache[oldest_key]
        
        return cache[key]
    
    return wrapper
```

#### Streamlit Caching
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_network_data(lemma: str, pos: str, corpus: str) -> pd.DataFrame:
    """
    Get network data with caching for performance.
    
    Args:
        lemma: Source lemma
        pos: Part of speech
        corpus: Corpus identifier
        
    Returns:
        Cached network data
    """
    return fetch_network_data(lemma, pos, corpus)

@st.cache_resource
def get_network_visualization(network: nx.Graph, layout: str = 'kk'):
    """
    Cache network visualization objects.
    
    Args:
        network: NetworkX graph
        layout: Layout algorithm
        
    Returns:
        Cached visualization object
    """
    return create_network_visualization(network, layout)
```

### Algorithm Optimization

#### Network Analysis Algorithms
```python
def optimized_community_detection(network: nx.Graph, algorithm: str = 'louvain') -> List[Set]:
    """
    Optimized community detection with algorithm selection.
    
    Args:
        network: NetworkX graph
        algorithm: Detection algorithm
        
    Returns:
        List of community sets
    """
    if algorithm == 'louvain':
        return louvain.best_partition(network)
    elif algorithm == 'leiden':
        return leidenalg.find_partition(network, leidenalg.ModularityVertexPartition)
    elif algorithm == 'greedy':
        return nx.community.greedy_modularity_communities(network)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
```

#### Centrality Calculation
```python
def batch_centrality_calculation(network: nx.Graph, 
                                measures: List[str]) -> Dict[str, Dict]:
    """
    Calculate multiple centrality measures efficiently.
    
    Args:
        network: NetworkX graph
        measures: List of centrality measures
        
    Returns:
        Dictionary of centrality values by measure
    """
    results = {}
    
    # Calculate degree centrality (always needed)
    if 'degree' in measures or 'weighted_degree' in measures:
        degree_cent = nx.degree_centrality(network)
        results['degree'] = degree_cent
    
    # Calculate betweenness centrality
    if 'betweenness' in measures:
        betweenness_cent = nx.betweenness_centrality(network)
        results['betweenness'] = betweenness_cent
    
    # Calculate eigenvector centrality
    if 'eigenvector' in measures:
        eigenvector_cent = nx.eigenvector_centrality(network, max_iter=1000)
        results['eigenvector'] = eigenvector_cent
    
    return results
```

## Deployment

### Production Environment

#### System Requirements
- **Server**: Ubuntu 20.04+ or CentOS 8+
- **Python**: 3.9+ with virtual environment
- **Memory**: Minimum 16GB RAM
- **Storage**: 50GB+ for data and logs
- **Network**: Stable internet connection

#### Environment Setup
```bash
# System updates
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.9 python3.9-venv python3.9-dev

# Install system dependencies
sudo apt install build-essential libssl-dev libffi-dev

# Create application user
sudo useradd -m -s /bin/bash congracnet
sudo usermod -aG sudo congracnet
```

#### Application Deployment
```bash
# Clone application
sudo -u congracnet git clone https://github.com/bperak/ConGraCNet.git /home/congracnet/app

# Create virtual environment
sudo -u congracnet python3.9 -m venv /home/congracnet/app/venv

# Install dependencies
sudo -u congracnet /home/congracnet/app/venv/bin/pip install -r requirements.txt

# Set up configuration
sudo -u congracnet cp /home/congracnet/app/authSettings.py.example /home/congracnet/app/authSettings.py
```

### Service Configuration

#### Systemd Service
```ini
# /etc/systemd/system/congracnet.service
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
```

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/congracnet
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:7475;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Monitoring and Logging

#### Application Logging
```python
import logging
import logging.handlers
import os

def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Set up comprehensive logging configuration."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('streamlit').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
```

#### Health Monitoring
```python
def health_check() -> Dict[str, Any]:
    """Perform comprehensive health check."""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }
    
    # Database connectivity
    try:
        graph.run("RETURN 1").evaluate()
        health_status['checks']['database'] = 'healthy'
    except Exception as e:
        health_status['checks']['database'] = f'unhealthy: {str(e)}'
        health_status['status'] = 'unhealthy'
    
    # API connectivity
    try:
        response = requests.get('https://api.sketchengine.eu/health', timeout=5)
        health_status['checks']['api'] = 'healthy' if response.status_code == 200 else 'unhealthy'
    except Exception as e:
        health_status['checks']['api'] = f'unhealthy: {str(e)}'
        health_status['status'] = 'unhealthy'
    
    # Memory usage
    memory_info = psutil.virtual_memory()
    health_status['checks']['memory'] = {
        'used_percent': memory_info.percent,
        'available_gb': memory_info.available / (1024**3)
    }
    
    return health_status
```

## Contributing Guidelines

### Code Standards

#### Python Style Guide
- **PEP 8**: Follow Python style guide
- **Type Hints**: Use type annotations for all functions
- **Docstrings**: Google style docstrings
- **Line Length**: Maximum 120 characters
- **Imports**: Organized and sorted

#### Function Design
```python
def example_function(param1: str, param2: int, 
                    optional_param: Optional[bool] = None) -> Dict[str, Any]:
    """
    Brief description of what the function does.
    
    Longer description if needed, explaining the purpose,
    algorithm, or important details.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        optional_param: Description of optional parameter
        
    Returns:
        Description of return value and structure
        
    Raises:
        ValueError: When parameters are invalid
        ConnectionError: When external service is unavailable
        
    Example:
        >>> result = example_function("test", 42)
        >>> print(result['status'])
        'success'
    """
    # Function implementation
    pass
```

### Pull Request Process

#### Before Submitting
1. **Code Quality**: Ensure all tests pass
2. **Documentation**: Update relevant documentation
3. **Style Check**: Run pre-commit hooks
4. **Coverage**: Maintain or improve test coverage

#### PR Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Review Process

#### Review Criteria
1. **Functionality**: Does the code work as intended?
2. **Quality**: Is the code well-written and maintainable?
3. **Testing**: Are there adequate tests?
4. **Documentation**: Is the code well-documented?
5. **Performance**: Are there performance implications?

#### Review Comments
- **Constructive**: Provide specific, actionable feedback
- **Respectful**: Maintain professional tone
- **Thorough**: Check all aspects of the code
- **Timely**: Respond within 48 hours

---

*This developer guide provides comprehensive information for contributing to ConGraCNet. For additional resources, see the API Reference and Architecture Guide.*
