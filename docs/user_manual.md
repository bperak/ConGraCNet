# ConGraCNet User Manual

> **Complete guide to using ConGraCNet for linguistic research and network analysis**

## Table of Contents

1. [Getting Started](#getting-started)
2. [Application Interface](#application-interface)
3. [Corpus Selection](#corpus-selection)
4. [Lemma and POS Selection](#lemma-and-pos-selection)
5. [Network Construction](#network-construction)
6. [Network Analysis](#network-analysis)
7. [Sentiment Analysis](#sentiment-analysis)
8. [Visualization Options](#visualization-options)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

## Getting Started

### First Launch

1. **Start the Application**
   ```bash
   streamlit run cgcnStream_0_3_6_withSBBLabel.py
   ```

2. **Access the Web Interface**
   - Open your browser to `http://localhost:8501`
   - The application will load with default settings

3. **Verify Database Connection**
   - Check the sidebar for connection status
   - Ensure Neo4j database is accessible

### Interface Overview

The ConGraCNet interface is organized into several main sections:

- **Header**: Application title and version information
- **Main Content Area**: Dynamic content based on current operation
- **Sidebar**: Configuration options and parameters
- **Tabs**: Organized feature sections (Semantic Dictionaries, Sentiment Analysis, etc.)

## Application Interface

### Main Components

#### 1. Header Section
- **Application Title**: "ConGraCNet"
- **Version Information**: Current version (3.6.2)
- **Introduction**: Markdown content explaining the application

#### 2. Corpus Selection
- **Corpus Dropdown**: Select from available corpora
- **Language Support**: Croatian (hr) and English (en)
- **API Integration**: Automatic data fetching from Sketch Engine

#### 3. Lemma and POS Selection
- **Lemma Input**: Text field for entering base word forms
- **POS Selection**: Part-of-speech specification
- **Combined Form**: Automatic combination (lemma+pos)

#### 4. Sidebar Configuration
- **Network Parameters**: Core network construction settings
- **Visualization Options**: Display and layout preferences
- **Analysis Settings**: Sentiment and clustering parameters

## Corpus Selection

### Available Corpora

ConGraCNet supports multiple corpora with different characteristics:

#### Croatian Corpora
- **hrWaC22**: Croatian Web Corpus 2022
- **hrTenTen13**: Croatian TenTen Corpus 2013
- **hrWaC**: Croatian Web Corpus

#### English Corpora
- **enTenTen13**: English TenTen Corpus 2013
- **enWaC**: English Web Corpus

### Selection Process

1. **Open Corpus Dropdown**
   - Located at the top of the main content area
   - Shows all available corpora for your account

2. **Choose Appropriate Corpus**
   - Consider language requirements
   - Select based on domain relevance
   - Consider corpus size and coverage

3. **Verify Selection**
   - Check that corpus information loads correctly
   - Verify language detection
   - Confirm grammatical relation availability

### Corpus Information

After selection, the system displays:
- **Corpus Name**: Full corpus identifier
- **Language**: Primary language (hr/en)
- **Corpus ID**: Unique identifier for database queries
- **Initial GramRel**: Default grammatical relation type
- **Initial Lexeme**: Suggested starting word
- **Initial POS**: Suggested part of speech

## Lemma and POS Selection

### Lemma Input

1. **Enter Base Form**
   - Use the dictionary form of the word
   - Avoid inflected forms (e.g., "run" not "running")
   - Ensure correct spelling

2. **Examples**
   - **Croatian**: "kuca" (house), "auto" (car), "grad" (city)
   - **English**: "house", "car", "city", "computer"

### Part of Speech Selection

1. **POS Categories**
   - **N**: Noun
   - **V**: Verb
   - **ADJ**: Adjective
   - **ADV**: Adverb
   - **PREP**: Preposition

2. **Selection Process**
   - Choose from dropdown menu
   - System automatically adds "-" prefix
   - Combined form: lemma + "-" + pos

3. **Validation**
   - System checks if lemma+pos exists in corpus
   - Error message if combination not found
   - Suggests alternatives if available

### Combined Form

The system automatically creates a combined form:
- **Format**: `lemma-pos`
- **Example**: "kuca-N" (house-noun)
- **Usage**: Used throughout the application for queries

## Network Construction

### Basic Parameters

#### 1. Friend Co-occurrences
- **Purpose**: Number of co-occurring words to analyze
- **Range**: 1-100 (default: 15)
- **Impact**: Larger values create more complex networks
- **Consideration**: Balance between detail and performance

#### 2. Measure Selection
- **Score**: Association strength (recommended for analysis)
- **Frequency**: Raw occurrence count
- **Usage**: Choose based on analysis goals

#### 3. Data Harvesting Options

##### Source Harvest
- **Purpose**: Fetch initial co-occurrence data
- **When to Use**: First time analyzing a lemma
- **Process**: Automatic API calls to Sketch Engine
- **Status**: Shows progress in spinner

##### Friend Harvest
- **Purpose**: Fetch friend-of-friend data
- **When to Use**: Building second-degree networks
- **Impact**: Expands network coverage
- **Consideration**: Increases processing time

### Grammatical Relations

#### Primary Relation (gr1)
- **Definition**: Relationship between source lemma and friends
- **Selection**: Choose from available relations
- **Examples**: 
  - **Croatian**: "obj" (object), "subj" (subject), "mod" (modifier)
  - **English**: "obj", "subj", "mod", "prep"

#### Secondary Relation (gr2)
- **Definition**: Relationship between friends and their friends
- **Selection**: Choose from available relations
- **Process**: System calculates available options
- **Impact**: Determines network depth

### Network Structure

```
Source Lemma (gr1) → Friends (gr2) → Friends of Friends
     ↓                    ↓                    ↓
   "kuca-N"           "velika-ADJ"        "stara-ADJ"
                      "mala-ADJ"          "nova-ADJ"
                      "lijepa-ADJ"        "moderna-ADJ"
```

## Network Analysis

### Frequency Analysis

#### Source Lemma Statistics
- **Frequency**: Raw count in corpus
- **Relative Frequency**: Per million tokens
- **Grammatical Relations**: Available relation types

#### Network Statistics
- **Node Count**: Total number of nodes
- **Edge Count**: Total number of edges
- **Density**: Network connectivity measure
- **Average Degree**: Mean connections per node

### Centrality Measures

#### Available Metrics
1. **Degree Centrality**: Number of direct connections
2. **Weighted Degree**: Sum of edge weights
3. **Betweenness Centrality**: Bridge importance
4. **Eigenvector Centrality**: Influence based on neighbors
5. **SLI (Semi-Local Importance)**: Custom centrality measure

#### Interpretation
- **High Degree**: Well-connected nodes
- **High Betweenness**: Bridge nodes between communities
- **High Eigenvector**: Influential nodes in important neighborhoods

### Community Detection

#### Algorithms Available
1. **Louvain**: Fast community detection
2. **Leiden**: Improved community quality
3. **CPM (Constant Potts Model)**: Resolution-based clustering

#### Parameters
- **Resolution**: Controls community size (0=large, 1=small)
- **Minimal Size**: Minimum community size
- **Quality Threshold**: Community quality measure

#### Results
- **Community Count**: Number of detected communities
- **Community Sizes**: Distribution of community sizes
- **Modularity**: Quality measure of community structure

## Sentiment Analysis

### Dictionary Integration

#### SenticNet 6
- **Coverage**: Multi-lingual sentiment lexicon
- **Values**: Polarity scores and labels
- **Integration**: Original vs. network-derived values

#### SentiWords 1.1
- **Coverage**: English sentiment lexicon
- **Values**: Positive/negative scores
- **Features**: Continuous sentiment values

#### SentiWordNet
- **Coverage**: English synset-based lexicon
- **Values**: Positive/negative/objective scores
- **Features**: WordNet integration

### Sentiment Calculation

#### Original Dictionary Values (ODV)
- **Source**: Direct dictionary lookup
- **Usage**: Baseline sentiment scores
- **Display**: Tabular format with polarity information

#### Assigned Dictionary Values (ADV)
- **Source**: Network-based calculation
- **Method**: Sentiment propagation through graph
- **Parameters**: 
  - Limit F: Number of friends to consider
  - Limit FoF: Number of friends-of-friends
  - Centrality Measure: For sentiment propagation

#### Comparison Analysis
- **Difference Calculation**: ODV vs. ADV comparison
- **Visualization**: Gradient coloring for easy comparison
- **Interpretation**: Network influence on sentiment

### Sentiment Propagation

#### Process Flow
1. **Source Lemma**: Initial sentiment assignment
2. **Friend Network**: Sentiment spread to first-degree neighbors
3. **FoF Network**: Sentiment spread to second-degree neighbors
4. **Aggregation**: Combined sentiment scores

#### Parameters
- **Centrality Measure**: How to weight node importance
- **Propagation Method**: Linear vs. exponential decay
- **Threshold**: Minimum sentiment for propagation

## Visualization Options

### Network Layouts

#### Available Algorithms
1. **Force-Directed (kk)**: Spring-based layout
2. **Fruchterman-Reingold (fr)**: Energy-based layout
3. **Large Graph Layout (lgl)**: Scalable layout
4. **Circular**: Ring-based arrangement
5. **Directed Layout (drl)**: Hierarchical arrangement
6. **Random**: Stochastic positioning
7. **Tree**: Hierarchical tree structure

#### Selection Criteria
- **Network Size**: Large networks benefit from lgl
- **Structure**: Hierarchical networks use tree/drl
- **Aesthetics**: Force-directed for general use
- **Performance**: Random for quick preview

### Node Customization

#### Size Parameters
- **Size Type**: degree, weighted_degree, freq
- **Size Range**: 0.0 to 10.0 (default: 5.0)
- **Scaling**: Linear or logarithmic scaling

#### Label Parameters
- **Label Size**: 1.0 to 40.0 (default: 10.0)
- **Label Type**: degree or weighted_degree based
- **Visibility**: Show/hide labels

### Edge Customization

#### Visual Properties
- **Edge Size**: 0.0 to 10.0 (default: 1.0)
- **Edge Labels**: Show/hide relationship labels
- **Label Size**: 0.0 to 10.0 (default: 1.0)
- **Color Scheme**: Automatic or custom coloring

### Interactive Features

#### Navigation
- **Zoom**: Mouse wheel or zoom controls
- **Pan**: Click and drag to move view
- **Reset**: Return to default view

#### Selection
- **Node Selection**: Click to select nodes
- **Edge Selection**: Click to select edges
- **Multiple Selection**: Shift+click for multiple
- **Information Panel**: Details for selected elements

## Advanced Features

### 3D Visualizations

#### Activation
- **Checkbox**: Enable 3D network view
- **Rendering**: WebGL-based 3D graphics
- **Performance**: Hardware acceleration required

#### Controls
- **Rotation**: Mouse drag to rotate
- **Zoom**: Mouse wheel for depth
- **Pan**: Right-click and drag
- **Reset**: Return to default orientation

### Export Options

#### Network Data
- **GraphML**: Standard graph format
- **CSV**: Node and edge tables
- **JSON**: Structured data export
- **PNG/SVG**: Static image export

#### Analysis Results
- **Statistics**: Network metrics summary
- **Community Data**: Clustering results
- **Sentiment Analysis**: Sentiment scores and comparisons

### Batch Processing

#### Multiple Lemma Analysis
1. **Input List**: Upload CSV with lemma-pos pairs
2. **Processing**: Automatic analysis of all items
3. **Results**: Aggregated analysis report
4. **Export**: Combined results in various formats

#### Performance Considerations
- **Memory Usage**: Monitor system resources
- **Processing Time**: Estimate based on network sizes
- **Storage**: Plan for result storage

### Custom Metrics

#### Implementation
- **Function Definition**: Python function with specific signature
- **Integration**: Add to centrality measures list
- **Parameters**: Configurable metric parameters
- **Validation**: Input validation and error handling

#### Examples
- **Custom Centrality**: Domain-specific importance measures
- **Similarity Metrics**: Custom similarity calculations
- **Clustering Quality**: Custom community quality measures

## Troubleshooting

### Common Issues

#### 1. Database Connection Problems
**Symptoms**: Error messages about Neo4j connection
**Solutions**:
- Verify Neo4j server is running
- Check connection credentials in authSettings.py
- Ensure firewall allows database port access
- Test connection with simple query

#### 2. API Errors
**Symptoms**: Sketch Engine API failures
**Solutions**:
- Verify API credentials are correct
- Check API rate limits and quotas
- Ensure internet connectivity
- Verify corpus access permissions

#### 3. Memory Issues
**Symptoms**: Application crashes or slow performance
**Solutions**:
- Reduce network size limits
- Enable data pruning options
- Use smaller corpora for testing
- Monitor system memory usage

#### 4. Performance Problems
**Symptoms**: Slow response times
**Solutions**:
- Enable caching options
- Reduce visualization complexity
- Use smaller sample sizes
- Optimize network parameters

### Error Messages

#### Common Error Types
1. **Connection Errors**: Database or API connectivity issues
2. **Validation Errors**: Invalid input parameters
3. **Processing Errors**: Computation or memory issues
4. **Display Errors**: Visualization or rendering problems

#### Error Resolution
1. **Read Error Message**: Note specific error details
2. **Check Parameters**: Verify input values and settings
3. **Reduce Complexity**: Simplify network or analysis parameters
4. **Check Logs**: Review application logs for details
5. **Restart Application**: Clear cached data and restart

### Performance Optimization

#### Network Size Management
- **Limit Co-occurrences**: Start with smaller values (5-10)
- **Enable Pruning**: Use frequency and score thresholds
- **Selective Harvesting**: Only harvest necessary data
- **Batch Processing**: Process multiple items efficiently

#### Visualization Optimization
- **Reduce Node Count**: Limit displayed nodes
- **Simplify Layouts**: Use faster layout algorithms
- **Disable Features**: Turn off unnecessary visual elements
- **Update Frequency**: Reduce automatic updates

#### Memory Management
- **Clear Cache**: Regularly clear cached results
- **Monitor Usage**: Watch memory consumption
- **Restart Application**: Periodic restarts for long sessions
- **Optimize Parameters**: Balance detail vs. performance

### Getting Help

#### Documentation Resources
1. **This Manual**: Comprehensive usage guide
2. **API Reference**: Function documentation
3. **Tutorials**: Step-by-step examples
4. **Video Guides**: Visual demonstrations

#### Support Channels
1. **GitHub Issues**: Bug reports and feature requests
2. **Discussions**: Community support and questions
3. **Documentation**: Self-service help resources
4. **Email Support**: Direct contact for complex issues

#### Reporting Problems
When reporting issues, include:
- **Error Message**: Exact error text
- **Steps to Reproduce**: Detailed reproduction steps
- **System Information**: OS, Python version, dependencies
- **Expected vs. Actual**: What you expected vs. what happened
- **Screenshots**: Visual evidence of the problem

---

*This user manual covers the core functionality of ConGraCNet. For advanced topics and developer information, please refer to the Developer Guide and API Reference.*
