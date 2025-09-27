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
In this project, we work with the Heating Load (HL) column as our target for both tasks.

- **Classification Task:** We transform Heating Load into a binary classification problem with two groups: *High* and *Low*. The split is made at the dataset's median value (≈18.95), creating an even balance of 384 buildings in each group. To ensure methodological soundness, the threshold is calculated using the **training set only** during actual experiments. This approach keeps a balanced distributions across all splits (Train 230/230, Val 77/77, Test 77/77).
- **Regression Task:** For regression, Heating Load is maintained as a continuous variable to predict directly. Values in the dataset range from 6.01 to 43.10 kWh/m², with a mean of approximately 22.31. This allows us to predict the exact energy requirement rather than just the category.

The dataset features straightforward numeric and categorical variables (surface area, wall area, roof area, overall height, glazing ratio) that directly influence building heating requirements. These characteristics make both the classification (high vs. low) and regression (exact value) tasks realistic and feasible.

---

## 4. Metrics Plan
For the classification task, we will use Accuracy and F1 (macro). Accuracy is easy to understand since it tells us the overall percentage of correct predictions. We also add F1 because it balances precision and recall, which is useful even when the classes are close to balanced.

For the regression task, we will use MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error). MAE shows the average difference between predicted and real values, which is easy to interpret. RMSE also measures error but gives more weight to larger mistakes, so it warns us when the model makes a few very bad predictions.

All metrics will be reported on both the validation and test sets so that we can compare performance fairly.

---

## 5. Baseline Plan
For the classification task, we will start with Logistic Regression and a Decision Tree. Logistic Regression gives us a simple linear model that is easy to understand and provides a good starting point. The Decision Tree adds a non-linear method that can capture interactions between features.

For the regression task, we will use Linear Regression and a Decision Tree Regressor. Linear Regression gives us a straightforward model that predicts Heating Load as a weighted sum of the features. The Decision Tree Regressor can capture more complex, non-linear relationships that the linear model might miss.

These baseline models are simple but strong enough to give us a reference point.

### Planned Models and Metrics
| Task | Baseline Models | Planned Metrics |
|------|-----------------|-----------------|
| Classification | Logistic Regression, Decision Tree | Accuracy, F1 |
| Regression | Linear Regression, Decision Tree Regressor | MAE, RMSE |
