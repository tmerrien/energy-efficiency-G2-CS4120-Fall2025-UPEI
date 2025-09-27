# CS-4120 Proposal – Energy Efficiency Dataset
**Course:** CS-4120, Fall 2025  
**Team:** [Your Names Here]

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
