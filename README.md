# Intelligent Customer Churn Prevention System

> End-to-end ML system combining Gradient Boosting predictions with GPT-4-powered retention strategy generation

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-Gradient_Boosting-green.svg)]()
[![GenAI](https://img.shields.io/badge/GenAI-GPT--4-orange.svg)]()

![Project Banner](docs/images/project_banner.png) <!-- Optional: Screenshot vom Dashboard später -->

---

## Project Overview

This project builds a production-ready machine learning system that helps telecommunications companies reduce customer churn by:

1. **Predicting** which customers will churn with 69.5% recall
2. **Explaining** why customers are at risk using feature importance
3. **Generating** personalized AI-powered retention strategies
4. **Optimizing** campaign ROI with data-driven success rate estimates

---

## Business Problem

Customer churn costs telecom companies millions annually:

- **Average acquisition cost:** €300-500 per customer
- **Lost CLV per churned customer:** €2,000-3,000
- **Industry churn rate:** 15-25% annually

**Traditional approach:** Generic retention offers, reactive campaigns, low success rates

**This system:** Personalized, AI-generated strategies with 1,453% ROI on high-risk segments

---

## Key Features

### 1. **ML-Powered Churn Prediction**

- **Model:** Gradient Boosting with optimized threshold (0.35)
- **Performance:** 69.5% Recall, 55.1% Precision, 634% ROI
- **Innovation:** Business-driven threshold optimization (not just accuracy maximization)

### 2. **Business-Driven Feature Engineering**

- `charges_per_service`: Value perception metric (€10 difference between churners/retainers)
- `is_month_to_month`: Contract risk indicator (strongest predictor!)
- `lifecycle_stage`: Customer maturity (for personalized strategies)

### 3. **GenAI Strategy Generator**

- **Technology:** OpenAI GPT-4o-mini
- **Output:** Personalized retention strategies with reasoning, success rates, and costs
- **Innovation:** Not just "will they churn?" but "HOW do we save them?"
- **Cost:** ~€0.05 per strategy (economical for scale)

### 4. **Data-Driven Success Rate Estimation**

- Success rates adjust based on customer risk level:
  - Low risk (20-40%): 55-65% success rate
  - Medium risk (40-70%): 40-50% success rate
  - High risk (70-95%): 30-45% success rate
- Prevents over-optimistic forecasting

---

## Results & Impact

### **Model Performance**

| Metric    | Value |
| --------- | ----- |
| Recall    | 69.5% |
| Precision | 55.1% |
| F2-Score  | 0.661 |
| ROI       | 634%  |

### **Strategy Generation**

- **Batch Processed:** 20 high-risk customers in ~2 minutes
- **Average Cost:** €68 per strategy
- **Expected Success Rate:** 42.5% for extreme high-risk (85-95% churn)
- **Campaign ROI:** 1,453%

### **Critical Business Insight** 💡

> 100% of extreme high-risk customers (85-95% churn probability) are in their first 12 months. This indicates a **systematic onboarding problem**, not individual retention issues. Recommendation: Invest in proactive onboarding optimization rather than reactive retention for this segment.

---

## 🛠️ Tech Stack

**Machine Learning:**

- `scikit-learn` - Model training & evaluation
- `XGBoost` - Gradient Boosting implementation
- `imbalanced-learn` - SMOTE testing
- `SHAP` - Model explainability (planned)

**GenAI:**

- `OpenAI GPT-4o-mini` - Strategy generation
- `python-dotenv` - API key management

**Data & Visualization:**

- `pandas`, `numpy` - Data processing
- `plotly`, `matplotlib`, `seaborn` - Visualizations

**Deployment (planned):**

- `Streamlit` - Interactive dashboard

---

## Project Structure

```
customer-churn-ai/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned & engineered features
├── notebooks/
│   ├── 01_EDA_Business_Perspective.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Baseline_Models.ipynb
│   └── 04_GenAI_Strategy_Generator.ipynb
├── src/
│   └── (modular code - planned)
├── models/
│   ├── final_churn_model.pkl
│   └── scaler.pkl
├── results/
│   ├── retention_strategies.csv
│   └── strategy_analysis.png
├── docs/
│   ├── Business_Context.md
│   └── Key_Insights.md
└── README.md
```

---

## Getting Started

### **Prerequisites**

- Python 3.9+
- OpenAI API Key (for strategy generation)

### **Installation**

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/customer-churn-ai.git
cd customer-churn-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env
```

### **Quick Demo**

```python
# Load model and generate strategy for a customer
import joblib
from src.strategy_generator import ImprovedStrategyGenerator

# Load model
model = joblib.load('models/final_churn_model.pkl')

# Initialize generator
generator = ImprovedStrategyGenerator()

# Generate strategy
strategy = generator.generate_strategy(
    customer_profile={...},
    churn_probability=0.85,
    top_drivers=[...]
)

print(strategy)
```

---

## What I Learned

### **Technical Skills**

- Handling imbalanced datasets (SMOTE vs threshold optimization)
- Business-driven ML optimization (ROI > Accuracy)
- GenAI API integration and prompt engineering
- End-to-end ML pipeline development

### **Business Skills**

- Trade-off analysis (Recall vs Precision)
- Cost-benefit analysis and ROI calculation
- Strategic problem identification (onboarding vs retention)

### **Key Decisions**

1. **No SMOTE in final model** - Simpler solution achieved same performance
2. **Threshold = 0.35** - Optimized for business metrics, not defaults
3. **Keep lifecycle_stage despite low ML importance** - Needed for GenAI prompts
4. **Data-driven success rates** - Realistic estimates over AI hallucination

---

## Future Improvements

- [ ] **Dashboard:** Streamlit interface for business users
- [ ] **SHAP Explainability:** Visual explanation of individual predictions
- [ ] **Feedback Loop:** Track actual success rates, retrain model
- [ ] **A/B Testing Framework:** Compare strategy effectiveness
- [ ] **Multi-channel Integration:** Email, SMS, call scripts
- [ ] **Real-time Scoring API:** FastAPI deployment

---

## 👤 Author

**Filip Kucharko**

---

## 🙏 Acknowledgments

- Dataset: [Telco Customer Churn - IBM Sample Data](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Inspiration: Real-world consulting challenges in customer retention

```
DEALER:     [10♠] [??]  → Punkte: 10
SPIELER:    [7♥] [K♣]   → Punkte: 17

[H] Hit | [S] Stand | [Q] Quit
Ihre Wahl: _
```
