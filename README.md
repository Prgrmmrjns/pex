# PatternFinder 🔍

A powerful tool for discovering and analyzing patterns in spatial data and time series, PatternFinder harnesses the Force of machine learning to predict future outcomes based on discovered patterns.

## 🌟 Features

- Pattern detection in time series data
- Correlation-based pattern matching
- Support for both spatial and temporal data analysis
- Visualization tools for pattern analysis
- Machine learning integration for prediction

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

The project includes several key components:

1. **Pattern Detection**: Use `functions.py` to detect patterns in your data:
   - Pattern matching with correlation coefficients
   - Visualization of detected patterns
   - Efficient sliding window operations

2. **Time Series Analysis**: Examples provided in Jupyter notebooks:
   - `mitbih.ipynb`: ECG pattern analysis
   - `d1namo.ipynb`: Glucose level prediction

## 🧪 Example

```python
from functions import get_match_counts, visualize_pattern, locate_pattern

# Define your pattern parameters
params = [pattern_length, start, end, stride, correlation_threshold, *pattern_values]

# Find pattern matches
matches = get_match_counts(params, your_data)

# Visualize the results
visualize_pattern(params)
```

## Example dataset

The Mitbih dataset was downloaded from https://www.kaggle.com/datasets/mondejar/mitbih-database

## 🤝 Contributing

The Force is strong with this one, but contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

