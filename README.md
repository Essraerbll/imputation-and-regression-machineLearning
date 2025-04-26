# Missing Value Imputation and Regression Modeling with Machine Learning

---

## 📚 Project Overview

This project explores **methods for handling missing data** in structured datasets through **random imputation** and **regression-based imputation** strategies.  
After imputation, the project evaluates and compares the predictive performance of **Neural Network (MLP Regressor)** models trained on the completed datasets.

The primary objective is to analyze the impact of different imputation techniques on the predictive accuracy of regression models, with a focus on **Mean Squared Error (MSE)** as the evaluation metric.

---

## 🛠 Methods and Tools

- **Random Imputation:**  
  Missing values are replaced with randomly sampled values from the observed feature range.

- **Regression Imputation:**  
  Missing values are predicted using a **Ridge Regression** model, with hyperparameter tuning via **GridSearchCV**.

- **Regression Modeling:**  
  - **Multi-Layer Perceptron (MLPRegressor)** neural networks are trained to predict the target variable.
  - **StandardScaler** and **MinMaxScaler** are utilized for feature scaling.

- **Evaluation Metrics:**  
  - **Mean Squared Error (MSE)** for training and testing performance.
  - Scatter plots for visual comparison of true vs predicted values.

- **Data Processing Pipelines:**  
  `Pipeline` objects are used to encapsulate preprocessing and modeling steps.

---

## 📦 Files Used

| File Name           | Purpose                                    |
|---------------------|--------------------------------------------|
| `original_data.csv`  | Original dataset without missing values   |
| `random_imp.csv`     | Dataset after random imputation           |
| `regression_imp.csv` | Dataset after regression-based imputation |

---

## 🚀 How to Run

1. Install the required packages:
```bash
pip install numpy pandas scikit-learn matplotlib
```

2. Execute the main script:
```bash
python Task1_2.py
```

The script will:
- Generate visualizations (scatter plots).
- Print MSE values for models trained on different imputed datasets.
- Export intermediate datasets (`original_data.csv`, `random_imp.csv`, `regression_imp.csv`).

---

## 📈 Key Results

- **Regression imputation** significantly outperforms **random imputation** in terms of predictive accuracy.
- **Neural Networks** trained on data imputed through regression achieve **lower MSE** scores.
- Visual analysis using scatter plots reveals tighter clustering around true values for regression-imputed datasets.

---

## ✨ Motivation

Missing data is a pervasive challenge in real-world datasets.  
Effective imputation is crucial to preserve the underlying structure and ensure model reliability.  
This project highlights the practical differences between naive (random) and intelligent (regression-based) imputation strategies when building machine learning models.

---

## 🧠 Future Work

- Extend the study to include advanced imputation techniques (e.g., KNN Imputation, Iterative Imputer).
- Evaluate model performance across different datasets and domains.
- Explore ensemble methods combining multiple imputation strategies.

---

## 📢 Acknowledgements

This work was inspired by the need to develop **robust preprocessing pipelines** capable of handling incomplete data in **machine learning workflows**.

---

# 🎯 Final Note

In machine learning, dealing with imperfection is often more important than modeling perfection.  
Through rigorous evaluation, this project demonstrates that **smart imputation** can dramatically enhance model performance.

