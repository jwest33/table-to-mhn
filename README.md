# Hopfield Table Memory

A neural associative memory system for intelligent database querying and pattern completion using Hopfield networks.

## Overview

Transform any pandas DataFrame into a Hopfield-based associative memory that supports:
- **Sparse Querying**: Find matches using only partial information
- **Pattern Completion**: Intelligently fill in missing data
- **Mixed Data Types**: Seamless handling of numeric and categorical columns
- **Similarity Search**: Find similar records based on learned patterns

## Key Features

**Partial Queries**: Search with incomplete information
```python
# Find tech employees with salary around 80k (ignoring other columns)
results = memory.query(dept='tech', salary=80000, sparse=True)
```

**Smart Completion**: Fill in missing information based on learned patterns
```python
# Complete a profile: age=33, dept='sales' → find similar sales employee
completed = memory.complete_pattern(age=33, dept='sales')
```

**Mixed Data Support**: Handles numeric, categorical, and sparse data automatically

## Installation

```bash
pip install torch pandas scikit-learn matplotlib hflayers
```

## Quick Start

```python
import pandas as pd
from hopfield_table import HopfieldTableMemory

# Your data
df = pd.DataFrame({
    'age': [34, 28, 45, 31],
    'salary': [70000, 80000, 60000, 90000], 
    'dept': ['sales', 'tech', 'hr', 'tech']
})

# Create memory
memory = HopfieldTableMemory(df)

# Example 1) Sparse query - match only specified features
results = memory.query(dept='tech', salary=80000, top_n=2, sparse=True)

#  Example 2) Dense query - match all features  
results = memory.query(age=30, salary=85000, sparse=False)

#  Example 3) Pattern completion
completed = memory.complete_pattern(age=35, dept='sales')
```

## Example 1 Results

```
Query: dept='tech', salary=80000
Match 1: age=28, salary=80000, dept='tech' (Distance: 0.0000)
Match 2: age=35, salary=85000, dept='tech' (Distance: 0.0769)
```

## Why Use This?

Unlike traditional databases that require exact matches, this system:
- Finds "similar" records even with partial information
- Learns patterns from your data to make intelligent suggestions
- Handles missing or imperfect queries gracefully
- Provides confidence scores and similarity distances

## Files

- `hopfield_table.py` - Core HopfieldTableMemory class
- `query.py` - Usage examples and testing
- Requires `hflayers` library for Hopfield network implementation5

Perfect for recommendation systems, data completion, and intelligent search!
