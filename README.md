# 🧠 Alzheimer’s Disease Predictor

**AI-Powered Clinical Decision Support System**

A machine learning-based tool to predict the risk of Alzheimer’s Disease using clinical, demographic, and lifestyle inputs. Built to assist early detection and enable individuals to consult professionals based on AI-based screening results.

---

## 🚀 Overview

Alzheimer’s Disease is a progressive neurodegenerative disorder affecting millions globally. Early detection significantly increases the chances of slowing disease progression. This project provides a **low-cost, accessible ML solution** to evaluate a person's risk using simple inputs—**no invasive tests or scans required**.

✅ Designed for **individual use** and **medical practitioners** in **low-resource settings**
✅ Offers **symptom-based risk prediction** with partial inputs
✅ Integrated into an easy-to-use **web-based interface**

---

## 💡 Key Features

* 🔍 **Risk Prediction** using clinical and lifestyle data
* 🤖 **XGBoost Model** – highest accuracy among all evaluated ML algorithms
* 🧪 **Handles missing inputs** using intelligent statistical imputation
* 📊 Real-time performance insights via **F1-score, ROC, confusion matrix**
* 🌐 **Web app interface** usable by non-experts, including rural/remote users

---

## 🛠 Tech Stack

| Component     | Tech Used                                                                |
| ------------- | ------------------------------------------------------------------------ |
| Language      | **Python**                                                               |
| Libraries     | **scikit-learn**, **XGBoost**, **Pandas**, **NumPy**, **Matplotlib**     |
| Serialization | **Pickle**                                                               |
| Interface     | Streamlit                                                                |
| Dataset       | Clinical, lifestyle, and demographic dataset (2149 records, 35 features) |

---

## 🧪 Models Compared

| Model               | Accuracy | F1-Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 71%      | 0.68     |
| SVM                 | 67%      | 0.73     |
| KNN                 | 63%      | 0.55     |
| Decision Tree       | 93%      | 0.91     |
| Random Forest       | 94%      | 0.93     |
| **XGBoost**         | **95%**  | **0.95** |

✅ **XGBoost** selected as the final model due to highest performance across all metrics.

---

## ⚙️ How It Works

1. **User Inputs** basic health and lifestyle information
2. Model processes data with **statistical handling for missing fields**
3. Returns **Alzheimer’s risk prediction**
4. User is **encouraged to seek medical help** based on results

---

## 🔧 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/your-username/alzheimers-predictor.git
cd alzheimers-predictor

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py  # or `streamlit run app.py` if using Streamlit
```

---

## 📌 Use Case

This tool can be used by:

* 🧓 **Elderly individuals** to check for symptoms
* 👨‍⚕️ **Doctors/Clinics** in remote or low-resource areas
* 🧬 **Researchers & ML practitioners** in healthcare AI

---
