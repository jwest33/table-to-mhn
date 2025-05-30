# Hopfield Table Memory

An interactive similarity search engine for structured data, built on Hopfield networks with an intuitive Streamlit interface. Upload any CSV dataset and perform intelligent similarity searches using neural associative memory.


![App example](streamlit-app-example.jpg)

## Overview

Transform any CSV dataset into a powerful associative memory system that supports:

- **CSV Upload Support**: Automatically detects column types and creates dynamic query interfaces
- **Default Dataset**: Loads `employees.csv` by default for immediate use
- **Sparse Querying**: Match using only the fields you specify, ignore the rest
- **Similarity Search**: Results ranked by distance and confidence scores
- **Mixed Data Types**: Seamlessly handles numeric and categorical features
- **Dynamic Interface**: Form inputs automatically adapt to your dataset structure
- **Export Results**: Download search results as CSV files

## Quick Start

### 1. Install Requirements

```bash
pip install torch pandas scikit-learn matplotlib streamlit numpy
pip install git+https://github.com/ml-jku/hopfield-layers
```

### 2. Run the Application

```bash
streamlit run query.py
```

The app will automatically load `employees.csv` if available, or you can upload your own CSV file.

## Features

### Intelligent Data Processing
- **Automatic Type Detection**: Numeric and categorical columns are identified automatically
- **Smart Input Generation**: Creates appropriate form inputs (number inputs, selectboxes, text fields) based on your data
- **Data Validation**: Handles missing values and provides data quality insights
- **Memory Statistics**: Shows pattern dimensions and encoding details

### Advanced Search Capabilities
- **Sparse Search Mode**: Only match specified fields (default)
- **Dense Search Mode**: Compare against all features
- **Configurable Results**: Choose number of top matches to return
- **Confidence Scoring**: Results include both distance metrics and confidence percentages

### User-Friendly Interface
- **Dataset Overview**: Displays record counts, column types, and sample data
- **Dynamic Query Forms**: Input fields automatically adapt to your CSV structure
- **Comprehensive Results**: All matches displayed in a single, sortable dataframe
- **Debug Mode**: Inspect internal query processing and pattern encoding

## How It Works

### Data Encoding Process
1. **Numeric Features**: Scaled using MinMaxScaler to [0,1] range
2. **Categorical Features**: One-hot encoded with unknown value handling
3. **Pattern Storage**: Combined feature vectors stored in Hopfield network memory
4. **Query Encoding**: Search parameters converted to same feature space

### Similarity Search
1. **Query Processing**: Input values encoded using same preprocessing pipeline
2. **Distance Calculation**: Euclidean distance computed between query and stored patterns
3. **Sparse Matching**: Only specified fields used for comparison (when enabled)
4. **Result Ranking**: Top-N matches ranked by distance with confidence scores

### Confidence Scoring
- **Formula**: `Confidence = 1 / (1 + distance)`
- **Range**: 0.0 to 1.0 (higher values indicate better matches)
- **Interpretation**: Values > 0.8 are high confidence, 0.6-0.8 medium, < 0.6 low

## File Structure

```
├── hopfield_table.py           # Core Hopfield memory implementation
├── query.py                    # Streamlit web interface
├── employees.csv               # Default dataset (optional)
└── README.md                   # This file
```

## Core Components

### `hopfield_table.py`
- `HopfieldTableMemory` class with encoding, querying, and pattern storage
- Support for mixed data types and sparse querying
- Memory expansion with `add_patterns()` method
- Built-in debugging and statistics

### `query.py`
- Streamlit web interface with CSV upload functionality
- Dynamic form generation based on dataset structure
- Results visualization and export capabilities
- Real-time dataset analysis and memory statistics

## Usage Examples

### Basic Search
1. Load the app with your CSV data
2. Fill in any combination of fields you want to match
3. Click "Search Similar Records" to find matches
4. Review results with confidence scores and distances

### Advanced Configuration
- **Top Matches**: Adjust slider to show more/fewer results
- **Sparse Query**: Toggle to match only specified vs. all fields
- **Debug Info**: Enable to see internal query processing details

## Technical Details

### Memory Requirements
- **Pattern Storage**: ~100MB for 100k records with 20 features
- **Query Processing**: Real-time similarity search (< 1 second for most datasets)
- **Scalability**: Tested with datasets up to 100k records

### Supported Data Types
- **Numeric**: Integers, floats, boolean values
- **Categorical**: Strings, limited cardinality categories
- **Mixed**: Datasets with combination of numeric and categorical columns

### Performance Characteristics
- **Search Speed**: O(n) where n is number of stored patterns
- **Memory Usage**: Linear with dataset size and feature dimensions
- **Preprocessing**: One-time cost during initial data loading
