# ConGraCNet - Project Planning & Architecture

## Project Overview
**ConGraCNet** (Construction Grammar Conceptual Networks) is a sophisticated corpus-based graph application for syntactic-semantic analysis of concepts. The application uses syntactic relations from syntactically tagged corpora to represent semantic relations in network structures.

## Core Architecture

### 1. Technology Stack
- **Frontend**: Streamlit web application
- **Backend**: Python 3.9+ with Neo4j graph database
- **Data Processing**: Pandas, NumPy, NetworkX, iGraph
- **Visualization**: Plotly, Matplotlib
- **NLP**: spaCy, NLTK, WordNet integration
- **Sentiment Analysis**: SenticNet, SentiWords, SentiWordNet

### 2. System Architecture
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

### 3. Core Components
- **Main Application**: `cgcnStream_0_3_6_withSBBLabel.py`
- **Core Functions**: `cgcn_functions_3_6.py`
- **Sentiment Analysis**: `sentiment_functions_3_6.py`
- **WordNet Integration**: `wordnet_functions.py`, `spacy_wordnet_functions.py`
- **Database Connection**: `authSettings.py`
- **Corpus Management**: `sketch2Neo2.py`

## Project Goals

### Primary Objectives
1. **Conceptual Network Analysis**: Extract and visualize conceptual networks from corpora
2. **Syntactic-Semantic Mapping**: Map syntactic relations to semantic structures
3. **Sentiment Propagation**: Analyze sentiment flow through network structures
4. **Community Detection**: Identify semantic domains and communities
5. **Multi-language Support**: Support for Croatian and English corpora

### Secondary Objectives
1. **Interactive Visualization**: Provide intuitive network exploration tools
2. **Performance Optimization**: Efficient handling of large-scale graph data
3. **Extensibility**: Modular architecture for easy feature additions
4. **Documentation**: Comprehensive user and developer documentation

## Code Style & Conventions

### Python Standards
- **PEP 8** compliance with 120 character line limit
- **Type hints** for all function parameters and return values
- **Google-style docstrings** for all functions and classes
- **Black formatting** for consistent code style

### Naming Conventions
- **Functions**: `snake_case` with descriptive names
- **Variables**: `snake_case` for local variables, `camelCase` for global
- **Constants**: `UPPER_SNAKE_CASE`
- **Files**: `snake_case` with descriptive names

### Code Organization
- **Maximum file length**: 500 lines (enforced)
- **Modular structure**: Separate files for different functionalities
- **Clear separation**: UI, business logic, and data access layers
- **Consistent imports**: Grouped and ordered imports

## File Structure & Organization

### Root Directory
```
congracnet/
├── cgcnStream_0_3_6_withSBBLabel.py  # Main Streamlit application
├── cgcn_functions_3_6.py             # Core functionality
├── sentiment_functions_3_6.py        # Sentiment analysis
├── wordnet_functions.py              # WordNet integration
├── spacy_wordnet_functions.py        # spaCy-WordNet bridge
├── sketch2Neo2.py                    # Corpus management
├── authSettings.py                   # Database configuration
├── requirements.txt                  # Dependencies
├── README.md                         # Project overview
├── PLANNING.md                       # This file
├── TASK.md                          # Task tracking
├── markdown/                         # Documentation fragments
└── venv/                            # Virtual environment
```

### Documentation Structure
- **User Documentation**: Installation, usage, features
- **Developer Documentation**: Architecture, API, development setup
- **API Documentation**: Function signatures, parameters, examples
- **Tutorials**: Step-by-step usage examples

## Development Constraints

### Technical Constraints
- **Python 3.9+** compatibility required
- **Neo4j database** must be accessible
- **Streamlit 1.41+** for web interface
- **Memory efficient** processing for large graphs
- **Cross-platform** compatibility (Windows, Linux, macOS)

### Performance Constraints
- **Response time**: < 5 seconds for standard queries
- **Memory usage**: < 2GB for typical operations
- **Scalability**: Support for corpora with 1M+ tokens
- **Caching**: Implement intelligent caching strategies

### Security Constraints
- **API key management**: Secure storage and usage
- **Database access**: Restricted access controls
- **Input validation**: Sanitize all user inputs
- **Error handling**: No sensitive information in error messages

## Testing Strategy

### Test Coverage
- **Unit tests**: 90%+ coverage for core functions
- **Integration tests**: Database and API interactions
- **UI tests**: Streamlit component functionality
- **Performance tests**: Large dataset handling

### Test Organization
- **Location**: `/tests` directory mirroring main structure
- **Naming**: Descriptive test names indicating purpose
- **Categories**: Expected use, edge cases, failure scenarios
- **Environment**: Use `.env` variables for configuration

## Deployment & Distribution

### Environment Setup
- **Virtual environment**: Always use `venv` for isolation
- **Dependencies**: Pin versions in requirements.txt
- **Configuration**: Environment-specific settings
- **Ports**: Configurable port settings (default: 7475)

### Distribution
- **Git repository**: Version control with semantic versioning
- **Documentation**: Comprehensive README and API docs
- **Installation**: Simple pip install process
- **Docker**: Optional containerization support

## Future Development

### Planned Features
1. **Advanced Visualization**: 3D network representations
2. **Machine Learning**: Automated community detection
3. **API Endpoints**: RESTful API for external access
4. **Plugin System**: Extensible architecture for custom analyses
5. **Real-time Updates**: Live corpus data integration

### Technical Debt
1. **Code Refactoring**: Reduce file sizes below 500 lines
2. **Error Handling**: Comprehensive exception management
3. **Logging**: Structured logging throughout application
4. **Monitoring**: Performance metrics and health checks
5. **Documentation**: Auto-generated API documentation

## Contact & Support

### Development Team
- **Lead Developer**: [To be specified]
- **Contributors**: [To be specified]
- **Contact**: [To be specified]

### Resources
- **Repository**: [GitHub URL]
- **Documentation**: [Documentation URL]
- **Issues**: [Issue tracker URL]
- **Wiki**: [Wiki URL]

---

*This document should be updated as the project evolves. Last updated: [Current Date]*
