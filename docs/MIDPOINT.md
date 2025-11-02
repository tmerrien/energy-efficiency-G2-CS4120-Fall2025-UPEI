# CS-4120 Midpoint – Energy Efficiency Dataset
**Course:** CS-4120, Fall 2025  
**Team:** Tanguy Merrien & Alex Deboer

---
### Updated Dataset Description and Cleaning Notes
We are using the public domain energy efficiency dataset from the University of California, Irvine. 
This dataset counts 8 (X1-8) features across 768 samples, with the Heating Load (Y1) as the primary 
target and the Cooling Load (Y2) as the auxiliary. This is a complete dataset, meaning there are 
no missing values.

In terms of preprocessing, we identified which features were categorical and which were numeric. 
Then, we one-hot encode the categorical features, standardize the numeric features using a scaler,
concatenate them together and simultaneously split the data into training, validation, and test sets.
All of this preprocessing is to ensure consistency across splits and to make it more easily reproducible.

| Column                         | Desciption                       | Type        | % Missing | Role             |
|--------------------------------|----------------------------------|-------------|-----------|------------------|
| Relative Compactness (X1)      | Shape Efficiency Ratio           | Numeric     | 0         | Feature          |
| Surface Area (X2)              | Total Exterior Surface Area      | Numeric     | 0         | Feature          |
| Wall Area (X3)                 | Total Wall Surface Area          | Numeric     | 0         | Feature          |
| Roof Area (X4)                 | Total Roof Surface Area          | Numeric     | 0         | Feature          |
| Overall Height (X5)            | Height Of The Building           | Numeric     | 0         | Feature          |
| Orientation (X6)               | Position Relative To Cardinality | Categorical | 0         | Feature          |
| Glazing Area (X7)              | Window-To-Wall Ratio             | Numeric     | 0         | Feature          |
| Glazing Area Distribution (X8) | Window Distribution Across Faces | Categorical | 0         | Feature          |
| Heating Load (Y1)              | Energy Required For Heating      | Numeric     | 0         | Target           |
| Cooling Load (Y2)              | Energy Required For Cooling      | Numeric     | 0         | Auxiliary Target |

---
### Exploratory Data Analysis (EDA)
Plot 1 is the target distribution plot for classification. The x axis being the two classes 
and the y axis being the number of samples belonging to each class. The bar graph is also 
further divided between the “true” class of each sample and the class predicted by our algorithm. 
Here we can see that our model slightly over-predicted the number of samples belonging to class 1,
while under-predicting the number of samples belonging to class 0, however, the number of 
misclassified samples remains extremely low.

Plot 2 is the correlation heatmap of numeric features. The two axes are composed of each of the 
six numeric features of this dataset, with each cell being the degree of similarity between each 
feature in a range of -1 to 1. The higher the number is, the more similar the two features are. 
We can observe that comparing a feature with itself results in a value of 1, which is to be expected. 
The features with the highest observed degree of separation are x1 and x2 with -0.99.

Plot 1: Target Distribution
![target_dist_plot](../src/outputs/plots/target_distribution_decision_tree.png)

Plot 2: Correlation Heatmap
![corr_heatmap_plot](../src/outputs/plots/correlation_heatmap_decision_tree_regressor.png)
---
### Classical Models and Training Setup

---
### Results and Evaluation

---
### Discussion and Neural Network Plan