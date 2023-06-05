# Warfarin_dose-Prediction
This is a regression model used for warfarin dose prediction. Using Linear regression model, Poly linear regression model, and Random forest regression.
The data used is modified from the source below.

Original dataset source:

International Warfarin Pharmacogenetics Consortium (IWPC)

Data from Estimation of the warfarin dose with clinical and pharmacogenetic data (PMID:19228618):
https://api.pharmgkb.org/v1/download/submission/553247439

Reference

Truda, G., & Marais, P. (2021). Evaluating warfarin dosing models on multiple datasets with a novel software framework and evolutionary optimisation. Journal of Biomedical Informatics, 113, 103634. https://doi.org/10.1016/j.jbi.2020.103634

---
# To Run
1. Run command ```pip install -r requirements.txt```
2. Go to UI directory
3. Run ```main.py```

---
# Models
To change the model, you need to replace the model path in main.py line 45 (with open(model_path, 'rb') as filo:).

Modedls are in models directory:
1. RFR.pkl - Random Forest Regression
2. lin.pkl - Linear Regression
3. poly.pkl - Polynomial Linear Regression
