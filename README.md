# ConGraCNet - Construction Grammar Conceptual Networks

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.41+-red.svg)](https://streamlit.io/)
[![Neo4j](https://img.shields.io/badge/neo4j-4.4+-green.svg)](https://neo4j.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **A sophisticated corpus-based graph application for syntactic-semantic analysis of concepts using Construction Grammar principles.**

## 🌟 Overview

ConGraCNet is a powerful web application that extracts and visualizes conceptual networks from corpora based on syntactic-semantic constructions. It uses syntactic relations from syntactically tagged corpora to represent semantic relations in network structures, enabling researchers and linguists to explore language patterns and conceptual relationships.

### 🎯 Key Features

- **🔍 Corpus Analysis**: Multi-language support (Croatian, English) with Sketch Engine API integration
- **🌐 Network Visualization**: Interactive graph representations with Plotly and NetworkX
- **💭 Sentiment Analysis**: Multi-dictionary sentiment analysis (SenticNet, SentiWords, SentiWordNet)
- **🏷️ Community Detection**: Advanced clustering algorithms (Louvain, Leiden) for semantic domain identification
- **📊 Statistical Analysis**: Frequency analysis, centrality measures, and network metrics
- **🔗 WordNet Integration**: Comprehensive lexical-semantic relationships
- **⚡ Real-time Processing**: Dynamic network construction and analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Python Core   │    │   Neo4j Graph   │
│   Frontend      │◄──►│   Functions     │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────►│   Sketch Engine│◄─────────────┘
                         │   API          │
                         └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Neo4j Database** (accessible at your specified URL)
- **Sketch Engine API** credentials
- **Git** for cloning the repository

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/bperak/ConGraCNet.git
   cd ConGraCNet
   ```

2. **Create Virtual Environment**
   ```bash
   # Standard Python
   python -m venv venv
   
   # Activate (Windows)
   .\venv\Scripts\activate
   
   # Activate (macOS/Linux)
   source venv/bin/activate
   
   # Or using Conda
   conda create --name congracnet python=3.11
   conda activate congracnet
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```bash
   # Copy and edit the configuration file
   cp authSettings.py.example authSettings.py
   # Edit authSettings.py with your credentials
   ```

5. **Start the Application**
   ```bash
   streamlit run cgcnStream_0_3_6_withSBBLabel.py
   ```

The application will be available at `http://localhost:8501`

## 📖 Usage Guide

### 1. Corpus Selection
- Choose from available corpora in the dropdown menu
- Supported languages: Croatian (hr), English (en)
- Corpus data is fetched via Sketch Engine API

### 2. Lemma and Part-of-Speech Selection
- Enter a lemma (base form of a word)
- Select the appropriate part of speech (POS)
- The system will display available grammatical relations

### 3. Network Construction
- **Friend Co-occurrences**: Set the number of co-occurring words to analyze
- **Measure Selection**: Choose between 'score' (association strength) or 'freq' (frequency)
- **Grammatical Relations**: Select primary and secondary grammatical relations
- **Data Harvesting**: Enable to fetch additional data if needed

### 4. Network Analysis
- **Visualization Parameters**: Customize node size, edge thickness, layout algorithms
- **Centrality Measures**: Analyze node importance using various metrics
- **Community Detection**: Identify semantic clusters and domains
- **Pruning Options**: Filter networks based on frequency and score thresholds

### 5. Sentiment Analysis
- **Multi-Dictionary Support**: SenticNet 6, SentiWords 1.1, SentiWordNet
- **Original vs. Assigned Values**: Compare dictionary values with network-derived sentiment
- **Sentiment Propagation**: Analyze how sentiment flows through network structures

### 6. Advanced Features
- **3D Visualizations**: Interactive 3D network representations
- **Export Options**: Save networks and analysis results
- **Batch Processing**: Analyze multiple lemmas simultaneously
- **Custom Metrics**: Implement custom centrality and similarity measures

## 🔧 Configuration

### Database Connection
Edit `authSettings.py` to configure your Neo4j connection:

```python
# Neo4j Database Configuration
graphURL = "http://your-neo4j-server:7474"
graphUser = "neo4j"
graphPass = "your-password"

# Sketch Engine API Configuration
userName = "your-sketch-engine-username"
apiKey = "your-sketch-engine-api-key"
```

### Application Settings
- **Port Configuration**: Default port 7475 (configurable)
- **Memory Limits**: Adjust for large-scale graph processing
- **Caching**: Enable/disable result caching
- **Logging**: Configure log levels and output

## 📊 Features in Detail

### Network Construction
- **Syntactic Relations**: Map grammatical patterns to semantic structures
- **Co-occurrence Analysis**: Identify word associations and collocations
- **Multi-level Networks**: Build second-degree networks (friends of friends)
- **Dynamic Expansion**: Real-time network growth and modification

### Visualization Options
- **Layout Algorithms**: Force-directed, circular, hierarchical, random layouts
- **Node Customization**: Size by degree, weighted degree, or frequency
- **Edge Styling**: Thickness, color, and label customization
- **Interactive Features**: Zoom, pan, node selection, and highlighting

### Analysis Capabilities
- **Centrality Measures**: Degree, betweenness, closeness, eigenvector centrality
- **Community Detection**: Louvain, Leiden, and custom clustering algorithms
- **Network Metrics**: Density, clustering coefficient, average path length
- **Statistical Analysis**: Frequency distributions, correlation analysis

### Sentiment Integration
- **Dictionary Integration**: Multiple sentiment lexicons
- **Network-based Sentiment**: Propagate sentiment through graph structures
- **Comparative Analysis**: Original vs. network-derived sentiment values
- **Cross-lingual Support**: Sentiment analysis for multiple languages

## 🧪 Testing

### Running Tests
```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_core_functions.py
pytest tests/test_sentiment_functions.py
pytest tests/test_network_functions.py

# Run with coverage
pytest --cov=cgcn_functions tests/
```

### Test Structure
```
tests/
├── test_core_functions.py      # Core functionality tests
├── test_sentiment_functions.py # Sentiment analysis tests
├── test_network_functions.py   # Network construction tests
├── test_ui_components.py       # Streamlit component tests
└── test_integration.py         # End-to-end integration tests
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Verify Neo4j server is running
   - Check connection credentials in `authSettings.py`
   - Ensure firewall allows connection to database port

2. **API Errors**
   - Verify Sketch Engine credentials
   - Check API rate limits
   - Ensure internet connectivity

3. **Memory Issues**
   - Reduce network size limits
   - Enable data pruning
   - Use smaller corpora for testing

4. **Performance Problems**
   - Enable caching options
   - Reduce visualization complexity
   - Use smaller sample sizes

### Performance Optimization
- **Caching**: Enable result caching for repeated queries
- **Pruning**: Use frequency and score thresholds to limit network size
- **Batch Processing**: Process multiple items in single operations
- **Memory Management**: Monitor and optimize memory usage

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include comprehensive docstrings
- Write tests for new features
- Keep files under 500 lines

## 📚 Documentation

- **[User Manual](docs/user_manual.md)** - Comprehensive usage guide
- **[Developer Guide](docs/developer_guide.md)** - Development and contribution guidelines
- **[API Reference](docs/api_reference.md)** - Function documentation and examples
- **[Architecture Guide](docs/architecture.md)** - System design and components
- **[Tutorials](docs/tutorials/)** - Step-by-step examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sketch Engine** for corpus data and API access
- **Neo4j** for graph database technology
- **Streamlit** for the web application framework
- **Academic Community** for linguistic research and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/bperak/ConGraCNet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bperak/ConGraCNet/discussions)
- **Documentation**: [Project Wiki](https://github.com/bperak/ConGraCNet/wiki)
- **Email**: [Contact Information]

## 🔄 Version History

- **v3.6.2** - Current version with SLI-based labeling
- **v3.6** - Enhanced sentiment analysis and 3D visualizations
- **v3.5** - Improved community detection and performance
- **v3.0** - Major rewrite with Streamlit interface

---

**ConGraCNet** - Exploring the conceptual structure of language through construction grammar and network analysis.

*Built with ❤️ for linguistic research and natural language processing.*

