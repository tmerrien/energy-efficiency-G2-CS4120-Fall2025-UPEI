# CS-4120 Proposal – Energy Efficiency Dataset
**Course:** CS-4120, Fall 2025  
**Team:** Tanguy Merrien & Alex Deboer

---
## 1. Problem and Motivation
In our modern world, the concept of sustainable energy is at the 
forefront of engineering developments. Normally, when someone mentions 
sustainable energy, they mean energy from natural, renewable sources 
such as solar or hydroelectric. However, that isn’t the only method 
for making our energy usage more sustainable. One way to do this would be to 
maximize the efficiency of the power we use, a concept known as energy efficiency. 
Using Machine Learning, it’s possible to discover trends in structural design to 
construct better energy-efficient buildings and reduce energy consumption. 
This would not only benefit consumers, who can maintain their lifestyle while using 
less energy, but also manufacturers who would effectively generate more energy 
for the same cost.

---
## 2. Dataset Description
For the purposes of this project, we will be using the public domain energy
efficiency dataset from the University of California, Irvine. 
This dataset is made up of 768 samples, each with 8 features (X1-8) 
used to calculate the Heating Load (Y1) and the Cooling Load (Y2). 
This dataset should be effective in providing clean working 
samples supporting both classification and regression.

 - **Source:** [UCI Repository](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
 - **Data Types:** Integer, Real
 - **Rows:** 768 
 - **Columns:** 10
 - **Missing Values:** None
 - **Sensitive Attributes:** None

| Column                         | Desciption                       | Type        | % Missing | Role             |
|--------------------------------|----------------------------------|-------------|-----------|------------------|
| Relative Compactness (X1)      | Shape Efficiency Ratio           | Continuous  | 0         | Feature          |
| Surface Area (X2)              | Total Exterior Surface Area      | Continuous  | 0         | Feature          |
| Wall Area (X3)                 | Total Wall Surface Area          | Continuous  | 0         | Feature          |
| Roof Area (X4)                 | Total Roof Surface Area          | Continuous  | 0         | Feature          |
| Overall Height (X5)            | Height Of The Building           | Continuous  | 0         | Feature          |
| Orientation (X6)               | Position Relative To Cardinality | Integer     | 0         | Feature          |
| Glazing Area (X7)              | Window-To-Wall Ratio             | Continuous  | 0         | Feature          |
| Glazing Area Distribution (X8) | Window Distribution Across Faces | Integer     | 0         | Feature          |
| Heating Load (Y1)              | Energy Required For Heating      | Continuous  | 0         | Target           |
| Cooling Load (Y2)              | Energy Required For Cooling      | Continuous  | 0         | Auxiliary Target |

---

## 3. Task Definitions
- **Classification:** Heating load → High vs Low. Threshold = dataset median (≈18.95). Balanced distribution: 384/384.  
  - Leakage-safe threshold (train-only) ≈18.92.  
  - Train/Val/Test splits balanced (230/230, 77/77, 77/77).
- **Regression:** Heating load as continuous variable. Range = 6.01–43.10. Mean ≈22.31.

---

## 4. Metrics Plan
- **Classification:** Accuracy + F1 (macro).  
- **Regression:** MAE + RMSE.  
Metrics reported on validation and test sets.

---

## 5. Baseline Plan
- **Classification:** Logistic Regression, Decision Tree.  
- **Regression:** Linear Regression, Decision Tree Regressor.  

### Planned Models and Metrics
| Task | Baseline Models | Planned Metrics |
|------|-----------------|-----------------|
| Classification | Logistic Regression, Decision Tree | Accuracy, F1 |
| Regression | Linear Regression, Decision Tree Regressor | MAE, RMSE |
