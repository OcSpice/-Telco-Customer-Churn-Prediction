# Telco-Customer-Churn-Prediction
Predict whether a Telco customer will churn (Yes/No) using machine learning.

## 🔹 Project Overview
- **Goal:** Predict whether a Telco customer will churn (Yes/No) using machine learning.  
- **Approach:** End-to-end ML pipeline with preprocessing, feature engineering, model training, evaluation, hyperparameter tuning, and business interpretation.  
- **Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Business Impact:** Reducing churn by just **10%** could save the company millions in annual revenue.  

---

## 🔎 Project Steps

### **1. Problem Statement & Objective**
- **Task:** Binary classification (Churn = Yes/No).  
- **Objective:** Maximize recall while keeping reasonable precision → catch as many churners as possible without too many false alarms.  
- **Business Framing:** Each lost customer ≈ \$1,000 CLV → retention has direct financial ROI.  

---

### **2. Load Data & Quick EDA**
- 7,043 rows × 21 features.  
- Class balance: ~26% churners.  
- Converted `TotalCharges` to numeric.  
- EDA:
  - Churn higher among *Month-to-Month contracts*.  
  - Tenure is strongly negatively correlated with churn.  
  - Paperless billing customers churn more.  

---

### **3. Data Cleaning**
- Dropped `customerID`.  
- Fixed blank strings in `TotalCharges`.  
- Normalized `"No internet service"` and `"No phone service"`.  

---

### **4. Feature Engineering**
- Created `tenure_group` bins.  
- Added `avg_monthly_charge = TotalCharges / tenure`.  
- Interaction feature: `month_to_month_paperless`.  
- Converted `Churn` → binary (`Churn_flag`).  

---

### **5. Preprocessing Pipeline**
- **Numeric features:** Median imputation + StandardScaler.  
- **Categorical features:** Mode imputation + OneHotEncoder.  
- Wrapped in `ColumnTransformer` for full pipeline.  

---

### **6. Train/Test Split**
- Stratified 70/30 split.  
- Preserved churn ratio.  

---

### **7. Baseline Models**
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

Cross-validated with Accuracy, Precision, Recall, F1, and ROC-AUC.  

---

### **8. Model Evaluation**
- Generated reports:  
  - Classification report  
  - Confusion matrix  
  - ROC curves  
  - Precision-Recall curves  
- Best baseline: **Gradient Boosting** with ROC-AUC ≈ 0.80.  

---

### **9. Hyperparameter Tuning**
- RandomizedSearchCV on Random Forest.  
- Tuned parameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight`.  

---

### **10. Addressing Imbalance**
- Compared:
  - `class_weight='balanced'`  
  - SMOTE oversampling  
- SMOTE improved recall without killing precision.  

---

### **11. Final Evaluation & Business Interpretation**
- Final model: **Tuned Random Forest + calibration + threshold tuning**.  
- Feature importance (SHAP + tree importance):
  - Contract type  
  - Tenure  
  - Monthly charges  
- **Business Insights:**
  1. Month-to-month + paperless billing → highest churn risk.  
  2. First-year customers are most vulnerable → focus on onboarding.  
  3. High charges increase churn likelihood → consider retention offers.  

---

### **12. Save Model & Artifact**
- Saved `telco_churn_pipeline.joblib`.  
- Added helper function `predict_customer(row_dict)` for single-customer inference.  

---

### **13. Executive Summary**
- **Best Model:** Random Forest (tuned, calibrated).  
- **Performance:** ROC-AUC ≈ 0.85, Recall ≈ 0.79, F1 ≈ 0.76.  
- **Recommendations:**
  1. Target high-risk customers with retention offers.  
  2. Improve onboarding for first-year customers.  
  3. Proactively engage paperless billing users.  
  4. Rank customers by churn probability for budget-optimized retention.  

---

## 🚀 Next Steps
- Deploy model as an API or Streamlit app.  
- Add external data (complaints, usage).  
- Retrain quarterly to adapt to behavior shifts.  

---

## 🛠️ Tools & Libraries
- Python (pandas, numpy, scikit-learn, seaborn, matplotlib)  
- imbalanced-learn (SMOTE)  
- joblib (model persistence)  
- Jupyter/Colab  

---

## 📜 License
This project is open-source and available under the MIT License.
