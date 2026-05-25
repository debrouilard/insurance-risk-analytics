# Insurance Risk Analytics & Predictive Modeling

**End-to-End Risk Analysis and Dynamic Pricing System for AlphaCare Insurance Solutions (ACIS)**
## 📋 Project Overview

This project analyses 18 months of historical auto-insurance claims data (Feb 2014 – Aug 2015) to help **AlphaCare Insurance Solutions** optimize marketing strategy and implement risk-based pricing in South Africa.

**Main Objectives:**
- Identify low-risk customer segments for competitive premium pricing
- Statistically validate risk drivers (province, gender, zip code)
- Build predictive models for claim severity
- Provide actionable, data-backed business recommendations

---

## 🛠️ Project Structure

```bash
insurance-risk-analytics/
├── .github/workflows/      # CI/CD pipeline
├── data/                   # DVC tracked data (raw + cleaned)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_loader.py
│   ├── eda_utils.py
│   ├── hypothesis_tests.py
│   └── modeling.py
├── reports/
│   └── final_report.md
├── .dvc/                   # DVC configuration
├── dvc.yaml
├── requirements.txt
└── README.md
