## 3. Comparative Analysis – Classical vs Neural Network

### Classification Results

Contrary to what we originally expected, we found that the Neural Networks couldn't improve on
the Decision Tree Model in either case. In terms of classification models, the Decision Tree
outperformed it in both Accuracy and F1 score across the validation and test sets. The Neural Network
classified a larger number of samples as "low" than the Decision Tree did, which points to one of its
limiting factor being tied to the decision boundary. Furthermore, the samples that were misclassified 
were often samples where the calculated heating load was much higher than the average for this data set,
namely, buildings with either an extremely large surface area or volume.

#### Table 1 – Classification comparison
| Model          | Val_Accuracy | Val_F1 | Test_Accuracy | Test_F1 |
|----------------|--------------|--------|--------|----------|
| Decision Tree  | 0.9870       | 0.9870 | 0.9740 | 0.9739 |
| Neural Network | 0.9416 | 0.9415 | 0.9610 | 0.9610 |


#### Plot 3: Confusion Matrix (Decision Tree)
![confusion_matrix](../outputs/plots/confusion_matrix_decision_tree.png)


### Regression Results

The Decision Tree Regressor model also proves itself to be most the effective regression
model of those tested. In this case, the Decision Tree model outperformed the Neural Network 
in all but one category, Validation set RMSE. From what we observed, the Neural Network
seemed to down weigh samples, leading to both a lower ceiling and lower floor than other models.
Furthermore, the Neural Network resulted in a wider residual spread than other models. This time, however,
the model appeared to have difficulty with samples containing lower heating loads.

#### Table 2 – Regression comparison

| Model                   | Val_MAE | Val_RMSE | Test_MAE | Test_RMSE |
|-------------------------|---------|---------|----------|---------|
| Decision Tree Regressor | 0.3741  | 0.6009  | 0.3867   | 0.5801  |
| Neural Network          | 0.4342  | 0.5869  | 0.4360   | 0.6002  |



#### Plot 4: Residuals Plot (Decision Tree Regressor)
![residuals](../outputs/plots/residuals_decision_tree_regressor.png)


## 5. Risks, Ethics, and Limitations


Among the risks and limitations of these models are issues that arise with generalization and overfitting
due to regional architecture and non-standard designs. From the dataset, only 12
building shaped were simulated, meaning that it is highly probable that any new building
you attempt to plug into these models might not resemble one that exists within the
dataset and could cause the model's predictions to be highly inaccurate. Along this
line of thinking, it's likely that these results would be extremely region dependent.
Architecture and building shapes change to fit the needs of their respective cultures
and communities, meaning that a model trained on Canadian buildings could experience difficulty
in attempting to predict the heating load for buildings with european architecture.

In a similar vein, misuse of these predictions in cases of profiling or discrimination based on energy consumption.
Consider the following hypothetical situation: lets say the government implements
a new tax to discourage their citizens from wasting energy, any home that uses more than 20% over its predicted heating load in power
to heat their house must pay into this "green tax". However, if the models were only trained on certain building shapes, people
who live in irregularly shaped homes or apartments could face unjust taxation due to the lack of diversity in the training of the models.

The next steps for modeling energy efficiency would be to increase the number of samples
in the dataset, more specifically implementing some sort of cross-regional tests to ensure
diversity in the shape and design of structure samples.

