# US State Health Rankings - Machine Learning Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://health-rankings-analysis-2025.streamlit.app/)

A data science application analyzing health outcomes across all 50 US states using clustering algorithms, predictive modeling, and interactive visualizations.

## Overview

This project applies machine learning techniques to public health data, identifying patterns in state health outcomes and enabling evidence-based comparisons. The analysis uses unsupervised learning to group states with similar health profiles and supervised learning to predict which factors most influence overall health rankings.

## Technical Implementation

### Machine Learning Pipeline

**K-Means Clustering**
- Groups states into distinct health profiles based on six health categories
- Cluster optimization via silhouette analysis and elbow method
- PCA dimensionality reduction for 2D visualization
- Explains 89% of variance with first two principal components

**Random Forest Classification**
- Predicts health cluster membership from category scores
- Feature importance analysis identifies key health determinants
- 5-fold cross-validation ensures model robustness
- Achieves 90% test accuracy, 98% cross-validation accuracy

**Anomaly Detection**
- Isolation Forest algorithm identifies states with atypical health patterns
- 10% contamination threshold
- Useful for detecting unique policy outcomes or measurement anomalies

**Similarity Analysis**
- Cosine similarity measures state-to-state comparability
- Enables peer group identification for benchmarking

### Interactive Dashboard

Built with Streamlit, the application features:
- Interactive scatter plots with state-level exploration
- Cluster comparison and profiling tools
- Individual state analysis with radar charts
- Real-time filtering and data exploration

## Technical Stack

**Language**: Python 3.9+

**Core Libraries**:
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning algorithms
- plotly - Interactive visualizations
- streamlit - Web application framework

## Installation

### Requirements
- Python 3.9 or higher
- pip package manager

### Setup Instructions

1. Clone repository
```bash
git clone https://github.com/olneyjR/health-rankings-ml.git
cd health-rankings-ml
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run Analysis Pipeline
```bash
python health_ml_analysis.py
```
Outputs cluster assignments, model metrics, and anomaly detection results.

### Launch Dashboard
```bash
streamlit run streamlit_app_modern.py
```
Access at http://localhost:8501

## Data Source

**America's Health Rankings 2025 Annual Report**
- Publisher: United Health Foundation
- Coverage: All 50 US states, 99 health indicators
- Data structure: 82,054 records across 1,578 measures

**Categories Analyzed**:
- Social and Economic Factors
- Physical Environment
- Clinical Care
- Behaviors
- Health Outcomes
- Overall composite score

## Methodology

### 1. Data Preparation
- Filtered 82,054 records to 6 primary health categories
- Transformed data structure from long to wide format (states × categories)
- Applied StandardScaler normalization for clustering algorithms

### 2. Cluster Analysis
- Evaluated optimal cluster count using silhouette scores (k=2 to k=8)
- Applied K-Means algorithm with k=4 clusters
- Used PCA to reduce dimensionality for visualization
- Assigned interpretive labels based on average health rankings

### 3. Predictive Modeling
- Split data: 80% training, 20% testing
- Trained Random Forest classifier (100 estimators, max_depth=5)
- Performed 5-fold cross-validation
- Extracted and ranked feature importances

### 4. Model Validation
- Test set accuracy: 90%
- Cross-validation mean: 98% (std: ±4%)
- Confirmed model generalizability across different data splits

## Results

### Cluster Profiles

Four distinct health groups emerged:

**High Performers** (23 states, avg rank: 12)
- Consistently strong performance across all categories
- Examples: New Hampshire, Vermont, Massachusetts
- Strongest in clinical care access

**Above Average** (varies by configuration)
- Balanced health outcomes
- Moderate performance across categories

**Mixed Outcomes** (varies by configuration)
- Strong physical environment scores
- Challenges in behavioral health metrics

**Facing Challenges** (27 states, avg rank: 37)
- Lower rankings across most categories
- Primary weaknesses in behaviors and social/economic factors
- Examples: Mississippi, Louisiana, Arkansas

### Feature Importance

Most predictive factors (Random Forest):
1. Overall health score - 40.5%
2. Social and Economic Factors - 29.0%
3. Behaviors - 11.6%
4. Clinical Care - 10.2%
5. Physical Environment - 5.1%
6. Health Outcomes - 4.6%

### Anomalies Identified

Five states demonstrated unique health patterns not captured by standard clusters:
California, Louisiana, Massachusetts, New Hampshire, Vermont

These anomalies may indicate distinctive policy environments, demographic compositions, or data quality considerations.

## Project Structure

```
health-ml-analysis/
├── streamlit_app_modern.py      # Dashboard application
├── health_ml_analysis.py         # ML pipeline and utilities
├── us_health_2025.csv            # Source data (17MB)
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
└── .gitignore                    # Version control exclusions
```

## Deployment Options

### Streamlit Cloud (Public)
1. Push repository to GitHub
2. Connect at share.streamlit.io
3. Configure: `streamlit_app_modern.py`
4. Deploy

### Posit Connect (Enterprise)
```bash
rsconnect deploy streamlit \
  --server https://connect.company.com \
  --api-key API_KEY \
  streamlit_app_modern.py
```

## Applications

**Public Health Policy**
- Benchmark state performance against similar peers
- Identify policy interventions from high-performing states
- Allocate resources based on cluster-specific needs

**Research**
- Explore relationships between health determinants
- Test hypotheses about social/economic impacts on health
- Visualize complex multidimensional health data

**Data Science Portfolio**
- Demonstrates end-to-end ML project workflow
- Shows practical application of multiple algorithms
- Exhibits data visualization and communication skills

## Extensions and Future Work

Potential enhancements:
- Time-series analysis with historical rankings data
- Integration of demographic and geographic variables
- Predictive modeling for future health trends
- SHAP values for enhanced model interpretability
- Automated reporting and email distribution

## Technical Notes

**Performance Considerations**:
- Streamlit caching applied to data loading and ML computations
- PCA reduces computational complexity for visualization
- Model serialization possible for production deployment

**Data Quality**:
- Missing values handled via exclusion (DC, aggregate "ALL" records)
- Standardization ensures fair comparison across different measurement scales
- Cross-validation guards against overfitting

## License and Attribution

**Data**: America's Health Rankings, United Health Foundation  
**Code**: Available for educational and portfolio purposes  
**Attribution**: Required when using or adapting this analysis

## Contact

Jeffrey Olney  
[GitHub](https://github.com/olneyjR) | [LinkedIn](https://www.linkedin.com/in/jeffrey-olney/)

## Acknowledgments

This project demonstrates applied machine learning in public health analytics. The analysis framework is generalizable to other state-level comparison studies across education, economics, or environmental metrics.

---

**Stack**: Python | scikit-learn | Streamlit | Plotly | pandas | NumPy
