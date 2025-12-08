# FDA FAERS Drug Safety Analysis - Unsupervised Learning Framework

## Project Overview
An advanced pharmacovigilance system that applies unsupervised machine learning techniques to FDA's Adverse Event Reporting System (FAERS) data, automatically discovering hidden drug safety patterns to prevent patient harm.

## Key Achievements
- **Processed:** 2.85 million adverse event reports across 7 quarters (Q1 2024 - Q3 2025)
- **Discovered:** 1,900 clinically significant drug-reaction associations
- **Identified:** 8 distinct patient risk clusters with silhouette score of 0.4698
- **Achieved:** 70% data compression while preserving 99.4% of clinical signals
- **Detected:** Safety signals 2-3 quarters before FDA alerts

## Quick Start

### Prerequisites
- Python 3.8+
- Google Colab (recommended for large dataset processing)
- 16GB RAM minimum

### Installation
```bash
# Clone repository
git clone https://github.com/orgs/2025-F-CS6220/teams/pharmacovigilance-system.git
cd pharmacovigilance-system

# Install dependencies
pip install -r requirements.txt
```

## Dataset Information

### Source
FDA Adverse Event Reporting System (FAERS)
- **URL:** https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
- **Coverage:** 7 quarters (2024 Q1 - 2025 Q3)
- **Raw Size:** ~15GB uncompressed
- **Processed Size:** 618.87MB pickle file

### Data Structure
- **DEMO:** Patient demographics (age, sex, country)
- **DRUG:** Drug names, doses, role codes (PS/SS)
- **REAC:** Adverse reactions (MedDRA terms)
- **OUTC:** Patient outcomes (Death, Hospitalization)
- **INDI:** Drug indications

### Final Dataset
```python
Shape: (2,847,862 cases × 19 features)
Key Features:
- suspect_drugs (list)
- all_reactions (list)
- age_years
- polypharmacy_category
- reaction_severity
- is_serious_outcome
```

## Methodology

### 1. Data Preprocessing
```python
# Process quarterly FAERS data
python src/preprocessing.py --quarters "2024Q1" "2024Q2" "2024Q3" "2024Q4" "2025Q1" "2025Q2" "2025Q3"

# Output: 2,847,862 clean records
# Removed: 132 duplicates, 46 outliers (age>120)
# Handled: 11.88M missing values
```

### 2. Association Rule Mining (FP-Growth)
```python
# Discover drug-reaction patterns
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Parameters
min_support = 0.001
min_confidence = 0.6
min_lift = 2.0

# Results: 1,900 rules discovered
# Top pattern: METHOTREXATE → PNEUMONIA (Confidence: 0.72, Lift: 8.4)
```

### 3. Patient Clustering (K-means)
```python
# Identify risk groups
from sklearn.cluster import KMeans

# Optimal K=8 (Silhouette: 0.4698)
kmeans = KMeans(n_clusters=8, random_state=42)

# Discovered clusters:
# - Elderly high-mortality (234,521 patients, 18.4% death rate)
# - Pediatric vaccine group (89,456 patients)
# - Oncology severe (67,234 patients, 42.1% death rate)
```

### 4. Dimensionality Reduction
```python
# Reduce 10,000+ items to 48 while preserving signals
correlation_drugs = 0.9994
correlation_reactions = 0.9969
```

## Key Results

### Association Rules
| Pattern | Support | Confidence | Lift | Clinical Significance |
|---------|---------|------------|------|----------------------|
| METHOTREXATE → PNEUMONIA | 0.018 | 0.72 | 8.4 | Immunosuppression risk |
| METHOTREXATE + Age≥65 → DEATH | 0.009 | 0.68 | 15.2 | Elderly toxicity |
| RITUXIMAB + Female → INFECTION | 0.012 | 0.81 | 6.9 | Gender-specific risk |

### Patient Clusters
| Cluster | Size | Death Rate | Profile |
|---------|------|------------|---------|
| 0 | 234,521 | 18.4% | Elderly polypharmacy |
| 1 | 89,456 | 0.3% | Pediatric vaccines |
| 2 | 67,234 | 42.1% | Oncology severe |
| 3 | 112,345 | 2.1% | Women reproductive |

## Usage Examples

### Load Processed Dataset
```python
import pandas as pd

# Load the final processed dataset
df = pd.read_pickle('data/final_dataset/Final Dataset.pkl')
print(f"Dataset shape: {df.shape}")
print(f"Total cases: {df.shape[0]:,}")
```

### Run Association Mining
```python
from src.mining import run_fp_growth

# Mine patterns
rules = run_fp_growth(df, min_support=0.001, min_confidence=0.6)
print(f"Rules discovered: {len(rules)}")
```

### Perform Clustering
```python
from src.clustering import patient_clustering

# Cluster patients
clusters = patient_clustering(df, n_clusters=8)
print(f"Silhouette score: {clusters['silhouette_score']:.4f}")
```

## Validation

### Resampling Validation
- **Method:** Two independent 100K samples (seeds: 42, 99)
- **Data Overlap:** Only 6%
- **Rule Consistency:** 44.6% identical rules
- **Conclusion:** Patterns are dataset-wide, not sample artifacts

### External Validation
- **FDA MedWatch:** 62% of rules confirmed
- **Literature:** 31/50 top rules have PubMed support
- **Early Detection:** 2-3 quarters before FDA alerts

## Impact

### Clinical Significance
- Identified 226,449 cases with serious reactions
- Discovered 76,434 patients on dangerous drug combinations
- Flagged 670,552 elderly patients at elevated risk

### Novel Findings
1. SGLT2 inhibitors + ACE inhibitors → Hyperkalemia
2. Biologics + Age<18 → Growth abnormalities
3. PPI + Immunotherapy → Reduced efficacy markers

## Team Members
- **Nidhi Patel**
- **Moumita Baidya**

## Dependencies
```txt
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
mlxtend==0.22.0
matplotlib==3.7.1
seaborn==0.12.2
```

## Future Work
1. **Real-time Streaming:** Implement Apache Kafka for continuous FAERS monitoring
2. **Deep Learning:** LSTM networks for 30-60 day adverse event prediction
3. **Causal Inference:** Propensity score matching to distinguish true drug effects

## License
This project is for educational purposes under CS 6220: Data Mining course.

## Acknowledgments
- FDA for maintaining the public FAERS database
- CS 6220 teaching team for guidance on unsupervised learning techniques

## Contact
For questions or collaboration, please open an issue or contact the team members through the GitHub repository or email us:
  - baidya.m@northeastern.edu
  - patel.nidhi@northeastern.edu

---
**Note:** This project demonstrates the application of unsupervised learning techniques for drug safety analysis. All findings should be validated by medical professionals before clinical application.
