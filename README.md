# 🛍️ Customer Purchase Propensity & Recommender System for E-Commerce

## 📌 Overview

This project builds a powerful **Machine Learning-based recommender system** for an e-commerce platform. It predicts the **purchase propensity** of users for various products and delivers **real-time product recommendations** based on user behavior and item similarity.

> 🔍 Use Case: Improve conversion rates and personalization by predicting user intent and offering data-driven product suggestions.

---

## 🚀 Project Highlights

### 🔢 Module 1: Data Preprocessing & EDA
- Cleaned ~1M+ user-item interaction logs
- Handled missing values, encoded categorical variables
- Reduced dimensionality using feature importance

### 🧠 Module 2: Purchase Propensity Modeling
- Modeled using **LightGBM Classifier**
- Achieved high performance using `ROC AUC`, `MAP@5`, and `MRR@5`
- Interpretable predictions using **SHAP values**

### 🎯 Module 3: Recommendation Engine
- Built a **hybrid recommender system** using:
  - Predicted purchase probabilities
  - Item similarity (content-based)
- Personalized top-N product recommendations

### 📊 Module 4: Model Explainability
- Visualized SHAP feature importance
- Interpreted key features influencing purchase behavior

### 🖥️ Module 5: Real-Time Streamlit Dashboard
- User-friendly web app with:
  - User ID selector
  - Purchase predictions + top-5 recommendations
  - SHAP explainability
  - Similar item suggestions

---

## 🗃️ Dataset

> Source: User-Item interaction logs from a simulated E-commerce platform  
> Size: ~1M rows  
> Features: `visitorid`, `itemid`, `category`, `price`, `brand`, `timestamp`, `event`, and more

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Pandas, NumPy, Scikit-learn**
- **LightGBM**
- **SHAP**
- **Streamlit**
- **Matplotlib, Seaborn**
- **Jupyter Notebook**

---

## 📈 Model Performance

| Metric                 | Score      |
|------------------------|------------|
| ROC AUC                | `0.88+`    |
| MAP@5                  | `0.67+`    |
| MRR@5                  | `0.72+`    |
| Top SHAP Features      | Price, Category, Popularity |

---

## 🧪 Run Locally

### 📦 Clone the Repo
```bash
git clone https://github.com/arpitsingh4297/ecommerce-purchase-recommender.git
cd ecommerce-purchase-recommender
