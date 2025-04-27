# Root Cause Detection Framework

A machine learning framework for detecting and analyzing root causes in business process event logs, with a focus on terminology variations and workload patterns.

## Overview

This framework analyzes business process event logs to detect various patterns and root causes of terminology variations. It implements multiple analysis approaches including:

- Frequency-based feature analysis
- Classification of terminology patterns
- Workload pattern detection
- Probability calculation for root causes

## Features

- Data preprocessing and feature engineering
- Multiple pattern detection algorithms:
  - System vs Human terminology usage (P1)
  - Peak time analysis (P2)
  - Individual typo detection (P3)
  - Overall workload impact (P4)
  - Individual convention analysis (P5)
  - Monthly workload patterns (P6)
- Feature importance visualization
- Probability-based root cause analysis
- Comprehensive metric calculations including:
  - Edit distance similarity
  - Variance metrics
  - Workload patterns
  - Terminology diversity

## Installation

```bash
git clone https://github.com/ShokoufehGh/RootCauseDetectionFramework.git
cd RootCauseDetectionFramework
pip install -r requirements.txt
```

## Usage

1. Prepare your input CSV file with event log data
2. Configure the input parameters in `main.py`
3. Run the analysis:

```bash
python main.py
```

## Project Structure

- `main.py` - Main entry point and orchestration
- `classification.py` - Classification models and pattern detection
- `frequency_features.py` - Frequency-based analysis
- `filter_data.py` - Data preprocessing and filtering
- `probability_calculation.py` - Root cause probability calculations
- `probability_chart.py` - Visualization utilities

## Data Format

Input CSV should contain the following columns:
- event concept:name
- timestamp
- resource
- Additional workload-related features

## Configuration

Modify the input parameters in `main.py`:

```python
input_data_pattern = {
    'pattern': 'Pattern Name',
    'synonyms': ['Term1', 'Term2', ...],
    'configuration': [[param1, param2]]
}
```

## Output

The framework generates:
- Feature importance charts
- ROC curves
- Pattern analysis reports
- Probability distributions
- Metric calculations

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@thesis{YourName2025,
    title={Root Cause Detection Framework for Business Process Event Logs},
    author={Your Name},
    year={2025},
    school={Your University}
}
```

## Contact

Your Name - your.email@example.com

Project Link: https://github.com/yourusername/RootCauseDetectionFramework