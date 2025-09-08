# 🎯 Predictive Modeling of Student Performance – ML Dashboard & Analytics

[![Open in Jupyter](https://img.shields.io/badge/Jupyter-Launch%20Notebook-orange?style=for-the-badge&logo=jupyter)](https://github.com/Saisohithk/Predictive-Modeling-of-Student-Performance/)

> An interactive data science project to explore, predict, and visualize student academic outcomes using **machine learning**, **feature engineering**, and **visual analytics**.

---

## 🚀 Project Overview

This repository presents a comprehensive **Predictive Modeling of Student Performance**, fusing **EDA, feature engineering, supervised ML models, and visual analytics** to deliver actionable insights into academic outcomes.

Leveraging open datasets of student attributes, the project uncovers drivers of performance and provides robust predictive models for:

- 📊 **Exploratory analysis** of grades and influencing factors  
- 🧠 **Feature engineering** from raw academic & demographic data  
- 🔍 **Supervised learning** (classification/regression) to predict pass/fail or grades  
- 📈 **Model evaluation & comparison** (accuracy, ROC, error metrics)  
- 🤖 **Notebook-based prediction tools** for experimenting with new student data  
- 📸 **Exportable visuals** and cleaned datasets for further analysis

---

## 🧠 Key Features

- 📖 Interactive EDA on student characteristics, grades, and relationships  
- 🛠️ Automated feature engineering and cleaning pipelines  
- 🧑‍🏫 Multiple ML models (Logistic Regression, Random Forest, SVM, etc.)  
- 🏆 Model comparison and metric reporting  
- 📉 Visualization of distributions, correlations, and predictions  
- 🧪 Jupyter notebook-based prediction tools for new data  
- 💾 Exportable charts, tables, and datasets

---

## 📂 Dataset Source

The project is based on public student performance datasets, such as:

🔗 

This dataset includes:

- Demographic info (age, gender, family background)
- Academic history (past grades, study time)
- School-related factors (support, absences, activities)
- Target variables: final grades, pass/fail status

---

## 📁 Project Structure

Below is the project structure as reflected in your repository:

```bash
Predictive-Modeling-of-Student-Performance/
├── student-data.csv
├── README.md
├── requirements.txt
├── Predictive.py
└── Predictive.ipynb
```

---

## 🔧 Tech Stack

### 🧑‍💻 Data Analysis & Visualization
- **Pandas, NumPy:** Data manipulation and analysis
- **Matplotlib, Seaborn:** Charts and plots
- **Plotly:** Interactive visualizations

### 🧠 Machine Learning
- **Scikit-learn:** Model training and evaluation (classification, regression)
- **Pipeline:** For preprocessing and model chaining
- **Joblib:** Model persistence
- **Imbalanced-learn:** (Optional) For handling class imbalance

### 🛠️ Utilities
- **Jupyter Notebook:** Interactive analysis and experimentation
- **Regex, OS, IO:** Data loading and cleaning

---

## 🛠️ How to Run Locally

### ✅ Clone Repository

```bash
git clone https://github.com/Saisohithk/Predictive-Modeling-of-Student-Performance.git
cd Predictive-Modeling-of-Student-Performance
```

### ✅ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### ✅ Install Requirements

```bash
pip install -r requirements.txt
```

### ✅ Launch Jupyter Notebooks

```bash
jupyter notebook
# Open notebooks from the notebooks/ directory
```

---

## 🏗️ Example EDA & Visualization Workflow

Below is a step-by-step summary of the main code-based workflow and analyses you can expect in the notebooks:

1. **Data Loading & Cleaning:**  
   - Read `student-data.csv`  
   - Map categorical variables (e.g., school, sex, address, jobs) to numeric values

2. **Feature Scaling:**  
   - Normalize features to a common scale for optimal model training

3. **Exploratory Data Analysis (EDA):**  
   - Pie charts of pass/fail rates  
   - Correlation heatmaps between features  
   - Bar charts and KDE plots:  
     - Going out frequency  
     - Romantic relationship status  
     - Mother’s job & education  
     - Age, failures, address, alcohol usage, internet access, study time, health

4. **Feature Engineering:**  
   - Selection and transformation of relevant predictors

5. **Model Training:**  
   - Train multiple classification models (Logistic Regression, KNN, SVM, etc.)  
   - Evaluate via accuracy, ROC-AUC, F1, and confusion matrices

6. **Visualization:**  
   - Exportable and publication-quality charts for all insights

---

## 🧾 Conclusion

This project demonstrates a full data science workflow for student performance prediction — from EDA and feature engineering to ML model deployment and interactive analytics. It enables both exploratory analysis and robust predictive modeling for academic data.

---

## 🤝 Connect
- 📫 Email: Saisohithkommana@gmail.com
- 💻 GitHub: [Saisohithk](https://github.com/Saisohithk)
- 🔗 LinkedIn: [Sai Sohith](www.linkedin.com/in/sai-sohith-410s62s11)
